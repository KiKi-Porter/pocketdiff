"""Small, reproducible training on detached autonomous self-states."""

from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path

import torch

from pocketdiff.geometry.bridge import apply_fractional_update
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.models import PocketDiffModel

from .clean import CleanBatch
from .self_state import build_self_state_batch, masked_self_state_motion_loss
from .multik import endpoint_fingerprint


CHECKPOINT_FORMAT = 'pocketdiff-self-state-v1'


def trajectory_fingerprint(positions):
    if not isinstance(positions, torch.Tensor):
        raise TypeError('trajectory_positions must be a tensor')
    digest = hashlib.sha256()
    digest.update(str((positions.dtype, tuple(positions.shape))).encode())
    digest.update(positions.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _validate_trajectory(clean, positions):
    if type(clean) is not CleanBatch:
        raise TypeError('clean must be an endpoint CleanBatch')
    if positions.device.type != 'cpu' or positions.ndim != 3 or positions.shape[1:] != clean.protein_pos.shape:
        raise ValueError('self-state trainer requires CPU trajectory [21, Np, 3]')
    if positions.shape[0] != 21 or not positions.is_floating_point() or not torch.isfinite(positions).all():
        raise ValueError('trajectory must contain exactly 21 finite floating states')
    if not torch.equal(positions[0], clean.apo_pos_ref):
        raise ValueError('trajectory step 0 must equal clean apo coordinates')
    # This also checks current frames and all reference geometry before training.
    build_self_state_batch(clean, positions, torch.zeros(len(clean.sample_ids), dtype=torch.long))


class SelfStateTrainer:
    """Train remaining-motion prediction on fixed detached autonomous states."""

    def __init__(self, clean, trajectory_positions, *, model_config=None, seed=17,
                 learning_rate=1e-3, max_grad_norm=10.0):
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('learning_rate must be finite and positive')
        if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError('max_grad_norm must be finite and positive')
        if any(getattr(clean, f.name).device.type != 'cpu' for f in fields(clean)
               if isinstance(getattr(clean, f.name), torch.Tensor)):
            raise ValueError('SelfStateTrainer currently supports CPU batches only')
        _validate_trajectory(clean, trajectory_positions)
        self.clean = clean
        self.trajectory_positions = trajectory_positions.detach().clone()
        self.data_fingerprint = endpoint_fingerprint(clean)
        self.trajectory_fingerprint = trajectory_fingerprint(self.trajectory_positions)
        self.model_config = dict(model_config or {'encoder_layers': 1, 'knn': 8, 'sigma_translation': 1.0})
        self.model_config.setdefault('motion_parameterization', 'remaining')
        if self.model_config['motion_parameterization'] != 'remaining':
            raise ValueError('SelfStateTrainer requires physical remaining parameterization')
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
        pocket_k = torch.randint(20, (len(self.clean.sample_ids),), generator=self.generator)
        batch = build_self_state_batch(self.clean, self.trajectory_positions, pocket_k)
        prediction = self.model(**batch.model_kwargs())
        loss = masked_self_state_motion_loss(
            prediction, batch.target_translation_local, batch.target_rotvec_local, batch.frame_valid)
        if not torch.isfinite(loss.loss):
            raise FloatingPointError('non-finite self-state loss before optimizer step')
        loss.loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'],
                                             error_if_nonfinite=True)
        self.optimizer.step()
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):
            raise FloatingPointError('non-finite self-state parameters after optimizer step')
        self.step_count += 1
        self.k_histogram += torch.bincount(pocket_k, minlength=20)
        return dict(step=self.step_count, loss=float(loss.loss.detach()),
                    translation_loss=float(loss.translation_loss.detach()),
                    rotation_loss=float(loss.rotation_loss.detach()),
                    gradient_norm_before_clip=float(norm), pocket_k=pocket_k.tolist(),
                    loss_parameterization='remaining')

    def save_checkpoint(self, path, *, metadata=None):
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
            trajectory_fingerprint=self.trajectory_fingerprint, metadata=dict(metadata or {}),
        )
        temporary = path.with_name(path.name + '.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def from_checkpoint(cls, path, clean, trajectory_positions):
        from pocketdiff.preprocessing import GEOMETRY_VERSION
        payload = torch.load(path, map_location='cpu')
        if payload.get('format') != CHECKPOINT_FORMAT or payload.get('geometry_version') != GEOMETRY_VERSION:
            raise ValueError('incompatible self-state checkpoint format or geometry version')
        if (payload['sample_ids'] != clean.sample_ids or
                payload['endpoint_fingerprint'] != endpoint_fingerprint(clean) or
                payload['trajectory_fingerprint'] != trajectory_fingerprint(trajectory_positions)):
            raise ValueError('checkpoint endpoint, trajectory or sample order mismatch')
        trainer = cls(clean, trajectory_positions, model_config=payload['model_config'],
                      **payload['trainer_config'])
        trainer.model.load_state_dict(payload['model_state_dict'], strict=True)
        trainer.optimizer.load_state_dict(payload['optimizer_state_dict'])
        trainer.step_count = payload['step']
        trainer.k_histogram = payload['k_histogram'].clone()
        trainer.generator.set_state(payload['k_generator_state'])
        torch.set_rng_state(payload['torch_rng_state'])
        return trainer


@torch.no_grad()
def evaluate_self_state(model, clean, trajectory_positions):
    """Evaluate physical remaining loss on all 20 detached autonomous states."""
    _validate_trajectory(clean, trajectory_positions)
    was_training = model.training
    model.eval()
    rows = []
    try:
        for k in range(20):
            pocket_k = torch.full((len(clean.sample_ids),), k, dtype=torch.long)
            batch = build_self_state_batch(clean, trajectory_positions, pocket_k)
            prediction = model(**batch.model_kwargs())
            valid = prediction.frame_valid & batch.frame_valid
            frames = build_residue_frames(batch.protein_pos, batch.atom_to_residue_global,
                                          batch.protein_atom_name, num_residues=batch.residue_type.numel())
            endpoint = apply_fractional_update(
                batch.protein_pos, batch.atom_to_residue_global, frames.origins, frames.frames,
                prediction.remaining_translation_local, prediction.remaining_rotvec_local,
                remaining_steps=1, frame_valid=valid)
            tr_error = (prediction.remaining_translation_local-batch.target_translation_local).square().mean(-1)
            rot_error = (prediction.remaining_rotvec_local-batch.target_rotvec_local).square().mean(-1)
            for graph, sample_id in enumerate(batch.sample_ids):
                residues = valid & (batch.batch_residue == graph)
                atoms = valid[batch.atom_to_residue_global] & (batch.batch_protein == graph)
                if not bool(residues.any()) or not bool(atoms.any()):
                    raise ValueError('self-state evaluation graph has no valid geometry: ' + sample_id)
                values = dict(sample_id=sample_id, k=k, t=199-10*k,
                              loss=float(tr_error[residues].mean()+rot_error[residues].mean()),
                              translation_loss=float(tr_error[residues].mean()),
                              rotation_loss=float(rot_error[residues].mean()),
                              current_holo_rmsd=float((batch.protein_pos[atoms]-batch.protein_pos_holo[atoms]).square().sum(-1).mean().sqrt()),
                              predicted_endpoint_holo_rmsd=float((endpoint[atoms]-batch.protein_pos_holo[atoms]).square().sum(-1).mean().sqrt()))
                if not all(math.isfinite(value) for key, value in values.items() if key not in ('sample_id', 'k', 't')):
                    raise FloatingPointError('non-finite self-state evaluation')
                rows.append(values)
    finally:
        model.train(was_training)
    keys = ('loss', 'translation_loss', 'rotation_loss', 'current_holo_rmsd', 'predicted_endpoint_holo_rmsd')
    per_k = []
    for k in range(20):
        subset = [row for row in rows if row['k'] == k]
        per_k.append(dict(k=k, t=199-10*k,
                          **{key: sum(row[key] for row in subset)/len(subset) for key in keys}))
    return dict(mode='self_state_teacher_forced', reduction='mean_over_graphs_then_20_times',
                mean={key: sum(row[key] for row in per_k)/20 for key in keys},
                per_k=per_k, per_graph=rows)


__all__ = ['CHECKPOINT_FORMAT', 'SelfStateTrainer', 'evaluate_self_state', 'trajectory_fingerprint']
