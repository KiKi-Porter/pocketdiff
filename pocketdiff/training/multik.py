"""Small CPU clean-condition training with independently resampled graph times."""

from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path

import torch

from pocketdiff.geometry.bridge import apply_fractional_update
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.models import PocketDiffModel
from .bridge import build_bridge_batch
from .clean import masked_remaining_motion_loss, masked_bridge_rate_loss


CHECKPOINT_FORMAT = 'pocketdiff-multik-clean-v1'


def endpoint_fingerprint(clean):
    """Bind a continuation to the exact ordered endpoint data and metadata."""
    digest = hashlib.sha256()
    for field in fields(clean):
        value = getattr(clean, field.name)
        digest.update(field.name.encode())
        if isinstance(value, torch.Tensor):
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


class MultiKCleanTrainer:
    """One optimizer step uses one freshly sampled k for every endpoint graph.

    This MVP trainer is CPU-only; checkpoints include CPU dropout RNG, the
    independent k generator, Adam state and exact data identity for replay.
    """

    def __init__(self, clean, *, model_config=None, seed=17, learning_rate=1e-3, max_grad_norm=10.0):
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('learning_rate must be finite and positive')
        if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError('max_grad_norm must be finite and positive')
        if any(getattr(clean, f.name).device.type != 'cpu' for f in fields(clean)
               if isinstance(getattr(clean, f.name), torch.Tensor)):
            raise ValueError('MultiKCleanTrainer currently supports CPU batches only')
        build_bridge_batch(clean, torch.zeros(len(clean.sample_ids), dtype=torch.long))
        self.clean = clean
        self.data_fingerprint = endpoint_fingerprint(clean)
        self.model_config = dict(model_config or {'encoder_layers': 1, 'knn': 8, 'sigma_translation': 1.0})
        self.model_config.setdefault('motion_parameterization', 'remaining')
        self.config = dict(seed=seed, learning_rate=learning_rate, max_grad_norm=max_grad_norm)
        torch.manual_seed(seed)
        self.model = PocketDiffModel(**self.model_config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.generator = torch.Generator().manual_seed(seed + 1)
        self.step_count = 0
        self.k_histogram = torch.zeros(20, dtype=torch.long)

    def step(self):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch = build_bridge_batch(self.clean, generator=self.generator)
        prediction = self.model(**batch.model_kwargs())
        raw_loss = masked_remaining_motion_loss(prediction, batch.target_translation_local,
                                            batch.target_rotvec_local, batch.frame_valid)
        loss = raw_loss
        if self.model.motion_parameterization == 'bridge_rate':
            loss = masked_bridge_rate_loss(prediction, batch.target_translation_local,
                                           batch.target_rotvec_local, batch.frame_valid,
                                           batch.pocket_k, batch.batch_residue)
        if not torch.isfinite(loss.loss):
            raise FloatingPointError('non-finite multi-k loss before optimizer step')
        loss.loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'],
                                             error_if_nonfinite=True)
        self.optimizer.step()
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):
            raise FloatingPointError('non-finite multi-k parameters after optimizer step')
        self.step_count += 1
        self.k_histogram += torch.bincount(batch.pocket_k, minlength=20)
        return dict(step=self.step_count, loss=float(loss.loss.detach()),
                    loss_parameterization=self.model.motion_parameterization,
                    remaining_loss=float(raw_loss.loss.detach()),
                    translation_loss=float(loss.translation_loss.detach()),
                    rotation_loss=float(loss.rotation_loss.detach()),
                    gradient_norm_before_clip=float(norm), pocket_k=batch.pocket_k.tolist())

    def save_checkpoint(self, path, *, metadata=None):
        # Import here to avoid a cycle: cache loading also uses training helpers.
        from pocketdiff.preprocessing import GEOMETRY_VERSION
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            format=CHECKPOINT_FORMAT, geometry_version=GEOMETRY_VERSION,
            model_config=self.model_config, trainer_config=self.config,
            model_state_dict=self.model.state_dict(), optimizer_state_dict=self.optimizer.state_dict(),
            step=self.step_count, k_histogram=self.k_histogram,
            k_generator_state=self.generator.get_state(), torch_rng_state=torch.get_rng_state(),
            sample_ids=list(self.clean.sample_ids), endpoint_fingerprint=self.data_fingerprint,
            metadata=dict(metadata or {}),
        )
        temporary = path.with_name(path.name + '.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def from_checkpoint(cls, path, clean):
        from pocketdiff.preprocessing import GEOMETRY_VERSION
        payload = torch.load(path, map_location='cpu')
        if payload.get('format') != CHECKPOINT_FORMAT or payload.get('geometry_version') != GEOMETRY_VERSION:
            raise ValueError('incompatible multi-k checkpoint format or geometry version')
        if payload['sample_ids'] != clean.sample_ids or payload['endpoint_fingerprint'] != endpoint_fingerprint(clean):
            raise ValueError('checkpoint endpoint data or sample order mismatch')
        trainer = cls(clean, model_config=payload['model_config'], **payload['trainer_config'])
        trainer.model.load_state_dict(payload['model_state_dict'], strict=True)
        trainer.optimizer.load_state_dict(payload['optimizer_state_dict'])
        trainer.step_count = payload['step']
        trainer.k_histogram = payload['k_histogram'].clone()
        trainer.generator.set_state(payload['k_generator_state'])
        torch.set_rng_state(payload['torch_rng_state'])
        return trainer


@torch.no_grad()
def evaluate_multik_clean(model, clean):
    """Enumerate every k, reporting graph-equal teacher-forced observations.

    Endpoint RMSD applies the full predicted remaining transform once. It is
    not a 20-step autoregressive rollout. Evaluation consumes no random draws.
    """
    was_training = model.training
    model.eval()
    rows = []
    try:
        for k in range(20):
            batch = build_bridge_batch(clean, torch.full((len(clean.sample_ids),), k, dtype=torch.long))
            prediction = model(**batch.model_kwargs())
            valid = prediction.frame_valid & batch.frame_valid
            frames = build_residue_frames(batch.protein_pos, batch.atom_to_residue_global,
                                          batch.protein_atom_name, num_residues=batch.residue_type.numel())
            args = (batch.protein_pos, batch.atom_to_residue_global, frames.origins, frames.frames,
                    prediction.remaining_translation_local, prediction.remaining_rotvec_local)
            next_pos = apply_fractional_update(*args, remaining_steps=batch.remaining_steps[batch.batch_residue],
                                               frame_valid=valid)
            endpoint = apply_fractional_update(*args, remaining_steps=1, frame_valid=valid)
            tr_error = (prediction.remaining_translation_local - batch.target_translation_local).square().mean(-1)
            rot_error = (prediction.remaining_rotvec_local - batch.target_rotvec_local).square().mean(-1)
            for graph, sample_id in enumerate(batch.sample_ids):
                residues = valid & (batch.batch_residue == graph)
                atoms = valid[batch.atom_to_residue_global] & (batch.batch_protein == graph)
                if not bool(residues.any()):
                    raise ValueError('cannot evaluate graph without valid residues: ' + sample_id)
                tr, rot = float(tr_error[residues].mean()), float(rot_error[residues].mean())
                next_rmsd = float((next_pos[atoms] - batch.protein_pos_next_target[atoms]).square().sum(-1).mean().sqrt())
                endpoint_rmsd = float((endpoint[atoms] - batch.protein_pos_holo[atoms]).square().sum(-1).mean().sqrt())
                current_rmsd = float((batch.protein_pos[atoms] - batch.protein_pos_holo[atoms]).square().sum(-1).mean().sqrt())
                numbers = (tr, rot, next_rmsd, endpoint_rmsd, current_rmsd)
                if not all(math.isfinite(n) for n in numbers):
                    raise FloatingPointError('non-finite evaluation for ' + sample_id)
                rows.append(dict(sample_id=sample_id, k=k, t=199-10*k, loss=tr+rot,
                                 translation_loss=tr, rotation_loss=rot,
                                 next_bridge_rmsd=next_rmsd, endpoint_holo_rmsd=endpoint_rmsd,
                                 current_holo_rmsd=current_rmsd))
    finally:
        model.train(was_training)
    keys = ('loss', 'translation_loss', 'rotation_loss', 'next_bridge_rmsd', 'endpoint_holo_rmsd', 'current_holo_rmsd')
    per_k = []
    for k in range(20):
        subset = [r for r in rows if r['k'] == k]
        per_k.append(dict(k=k, t=199-10*k, **{key: sum(r[key] for r in subset)/len(subset) for key in keys}))
    return dict(mode='teacher_forced_clean', reduction='mean_over_graphs_then_20_times',
                mean={key: sum(r[key] for r in per_k)/20 for key in keys}, per_k=per_k, per_graph=rows)


__all__ = ['MultiKCleanTrainer', 'evaluate_multik_clean', 'endpoint_fingerprint']
