"""Online autonomous-state training with exact checkpoint replay."""

from dataclasses import fields
import math
from pathlib import Path

import torch

from pocketdiff.models import PocketDiffModel

from .clean import CleanBatch
from .multik import endpoint_fingerprint
from .online_state import OnlineTrajectory, refresh_online_trajectory, build_latest_self_state
from .self_state import masked_self_state_motion_loss
from .self_state_trainer import trajectory_fingerprint


CHECKPOINT_FORMAT = 'pocketdiff-online-self-state-v1'


def _validate_clean(clean):
    if type(clean) is not CleanBatch:
        raise TypeError('clean must be an endpoint CleanBatch')
    if any(getattr(clean, field.name).device.type != 'cpu'
           for field in fields(clean)
           if isinstance(getattr(clean, field.name), torch.Tensor)):
        raise ValueError('OnlineSelfStateTrainer currently supports CPU batches only')
    if not torch.equal(clean.protein_pos, clean.apo_pos_ref):
        raise ValueError('clean must contain apo endpoint coordinates')
    if bool((clean.pocket_k != 0).any()) or bool((clean.targetdiff_t != 199).any()):
        raise ValueError('clean must be at k=0/t=199')
    if len(clean.sample_ids) == 0 or len(set(clean.sample_ids)) != len(clean.sample_ids):
        raise ValueError('clean sample_ids must be non-empty and unique')


def _validate_trajectory(clean, trajectory):
    if not isinstance(trajectory, OnlineTrajectory):
        raise TypeError('trajectory must be an OnlineTrajectory')
    if trajectory.positions.shape[1:] != clean.protein_pos.shape:
        raise ValueError('trajectory atom shape differs from clean batch')
    if trajectory.positions.shape[0] != 21:
        raise ValueError('online trajectory must contain exactly 21 states')
    if not torch.equal(trajectory.positions[0], clean.apo_pos_ref):
        raise ValueError('trajectory step 0 must equal clean apo coordinates')


def _copy_trajectory(trajectory):
    return OnlineTrajectory(
        trajectory.positions.detach().cpu().clone(),
        trajectory.frame_valid.detach().cpu().clone(),
        trajectory.update_valid.detach().cpu().clone(),
    )


class OnlineSelfStateTrainer:
    """Train physical remaining motion on states refreshed by the current model.

    The train trajectory is refreshed immediately after every
    refresh_interval-th optimizer update. Therefore a checkpoint saved at
    step N contains exactly the state that the next step will consume.
    """

    def __init__(
        self,
        clean,
        *,
        model_config=None,
        seed=17,
        learning_rate=1e-3,
        max_grad_norm=10.0,
        refresh_interval=10,
        initial_state_dict=None,
        initial_trajectory=None,
    ):
        _validate_clean(clean)
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('learning_rate must be finite and positive')
        if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError('max_grad_norm must be finite and positive')
        if not isinstance(refresh_interval, int) or refresh_interval <= 0:
            raise ValueError('refresh_interval must be a positive integer')
        self.clean = clean
        self.data_fingerprint = endpoint_fingerprint(clean)
        self.model_config = dict(model_config or {
            'encoder_layers': 1, 'knn': 8, 'sigma_translation': 1.0,
        })
        self.model_config.setdefault('motion_parameterization', 'remaining')
        if self.model_config['motion_parameterization'] != 'remaining':
            raise ValueError('online self-state training requires physical remaining parameterization')
        self.config = dict(seed=seed, learning_rate=learning_rate,
                           max_grad_norm=max_grad_norm,
                           refresh_interval=refresh_interval)
        torch.manual_seed(seed)
        self.model = PocketDiffModel(**self.model_config)
        if initial_state_dict is not None:
            self.model.load_state_dict(initial_state_dict, strict=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.generator = torch.Generator().manual_seed(seed + 1)
        if initial_trajectory is None:
            self.trajectory = _copy_trajectory(refresh_online_trajectory(self.model, clean))
        else:
            _validate_trajectory(clean, initial_trajectory)
            self.trajectory = _copy_trajectory(initial_trajectory)
        self.step_count = 0
        self.refresh_count = 0
        self.last_refresh_step = 0
        self.k_histogram = torch.zeros(20, dtype=torch.long)

    @classmethod
    def from_initial_checkpoint(cls, path, clean, *, refresh_interval=10,
                                seed=None, learning_rate=None,
                                max_grad_norm=None):
        """Initialize a new online run from a Phase 17 model checkpoint.

        The old optimizer and RNG are deliberately not continued. The
        checkpoint contributes only the model weights and architecture; the
        online experiment starts a fresh fixed seed-17 optimizer stream.
        """
        from pocketdiff.preprocessing import GEOMETRY_VERSION
        payload = torch.load(path, map_location='cpu')
        if payload.get('geometry_version') != GEOMETRY_VERSION:
            raise ValueError('incompatible initial checkpoint geometry version')
        if payload.get('format') != 'pocketdiff-multik-clean-v1':
            raise ValueError('initial checkpoint must be a Phase 17 multi-k checkpoint')
        if payload.get('sample_ids') != clean.sample_ids:
            raise ValueError('initial checkpoint sample order differs from clean batch')
        if payload.get('endpoint_fingerprint') != endpoint_fingerprint(clean):
            raise ValueError('initial checkpoint endpoint fingerprint differs from clean batch')
        config = dict(payload['model_config'])
        config['motion_parameterization'] = 'remaining'
        trainer_config = payload['trainer_config']
        return cls(
            clean,
            model_config=config,
            seed=trainer_config['seed'] if seed is None else seed,
            learning_rate=trainer_config['learning_rate'] if learning_rate is None else learning_rate,
            max_grad_norm=trainer_config['max_grad_norm'] if max_grad_norm is None else max_grad_norm,
            refresh_interval=refresh_interval,
            initial_state_dict=payload['model_state_dict'],
        )

    @property
    def trajectory_fingerprint(self):
        return trajectory_fingerprint(self.trajectory.positions)

    def refresh_train_trajectory(self):
        """Refresh the train state and return its new fingerprint."""
        self.trajectory = _copy_trajectory(refresh_online_trajectory(self.model, self.clean))
        self.refresh_count += 1
        self.last_refresh_step = self.step_count
        return self.trajectory_fingerprint

    def step(self):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        pocket_k = torch.randint(
            20, (len(self.clean.sample_ids),), generator=self.generator,
        )
        batch = build_latest_self_state(self.clean, self.trajectory, pocket_k)
        prediction = self.model(**batch.model_kwargs())
        loss = masked_self_state_motion_loss(
            prediction, batch.target_translation_local,
            batch.target_rotvec_local, batch.frame_valid,
        )
        if not torch.isfinite(loss.loss):
            raise FloatingPointError('non-finite online self-state loss before optimizer step')
        loss.loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config['max_grad_norm'],
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        if not all(torch.isfinite(parameter).all() for parameter in self.model.parameters()):
            raise FloatingPointError('non-finite online self-state parameters after optimizer step')
        self.step_count += 1
        self.k_histogram += torch.bincount(pocket_k, minlength=20)
        refreshed = False
        if self.step_count % self.config['refresh_interval'] == 0:
            self.refresh_train_trajectory()
            refreshed = True
        return dict(
            step=self.step_count,
            loss=float(loss.loss.detach()),
            translation_loss=float(loss.translation_loss.detach()),
            rotation_loss=float(loss.rotation_loss.detach()),
            gradient_norm_before_clip=float(norm),
            pocket_k=pocket_k.tolist(),
            trajectory_fingerprint=self.trajectory_fingerprint,
            refreshed=refreshed,
            loss_parameterization='remaining',
        )

    def save_checkpoint(self, path, *, metadata=None):
        from pocketdiff.preprocessing import GEOMETRY_VERSION
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            format=CHECKPOINT_FORMAT,
            geometry_version=GEOMETRY_VERSION,
            model_config=self.model_config,
            trainer_config=self.config,
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            step=self.step_count,
            refresh_count=self.refresh_count,
            last_refresh_step=self.last_refresh_step,
            k_histogram=self.k_histogram,
            k_generator_state=self.generator.get_state(),
            torch_rng_state=torch.get_rng_state(),
            sample_ids=list(self.clean.sample_ids),
            endpoint_fingerprint=self.data_fingerprint,
            trajectory_fingerprint=self.trajectory_fingerprint,
            trajectory_positions=self.trajectory.positions,
            trajectory_frame_valid=self.trajectory.frame_valid,
            trajectory_update_valid=self.trajectory.update_valid,
            metadata=dict(metadata or {}),
        )
        temporary = path.with_name(path.name + '.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def from_checkpoint(cls, path, clean):
        from pocketdiff.preprocessing import GEOMETRY_VERSION
        payload = torch.load(path, map_location='cpu')
        if payload.get('format') != CHECKPOINT_FORMAT:
            raise ValueError('incompatible online self-state checkpoint format')
        if payload.get('geometry_version') != GEOMETRY_VERSION:
            raise ValueError('incompatible online self-state checkpoint geometry version')
        if payload.get('sample_ids') != clean.sample_ids:
            raise ValueError('checkpoint sample order differs from clean batch')
        if payload.get('endpoint_fingerprint') != endpoint_fingerprint(clean):
            raise ValueError('checkpoint endpoint fingerprint differs from clean batch')
        trajectory = OnlineTrajectory(
            payload['trajectory_positions'].detach().cpu().clone(),
            payload['trajectory_frame_valid'].detach().cpu().clone(),
            payload['trajectory_update_valid'].detach().cpu().clone(),
        )
        _validate_trajectory(clean, trajectory)
        if payload.get('trajectory_fingerprint') != trajectory_fingerprint(trajectory.positions):
            raise ValueError('checkpoint trajectory fingerprint mismatch')
        trainer = cls(
            clean,
            model_config=payload['model_config'],
            **payload['trainer_config'],
            initial_trajectory=trajectory,
        )
        trainer.model.load_state_dict(payload['model_state_dict'], strict=True)
        trainer.optimizer.load_state_dict(payload['optimizer_state_dict'])
        trainer.step_count = int(payload['step'])
        trainer.refresh_count = int(payload.get('refresh_count', 0))
        trainer.last_refresh_step = int(payload.get('last_refresh_step', 0))
        trainer.k_histogram = payload['k_histogram'].clone()
        trainer.generator.set_state(payload['k_generator_state'])
        torch.set_rng_state(payload['torch_rng_state'])
        return trainer


__all__ = ['CHECKPOINT_FORMAT', 'OnlineSelfStateTrainer']
