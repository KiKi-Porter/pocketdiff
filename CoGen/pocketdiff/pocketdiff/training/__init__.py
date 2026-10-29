"""Training helpers for the staged PocketDiff implementation."""

from .clean import (
    CleanBatch,
    CleanExample,
    MotionLoss,
    ChiLoss,
    collate_clean_examples,
    make_clean_example,
    masked_remaining_motion_loss,
    masked_bridge_rate_loss,
    masked_periodic_chi_loss,
    masked_chi_loss,
)
from .trainer import TrainRecord, save_checkpoint, train_clean_batch
from .split import CleanSplit, deterministic_clean_split
from .bridge import BridgeBatch, build_bridge_batch
from .multik import MultiKCleanTrainer, evaluate_multik_clean
from .self_state import SelfStateBatch, build_self_state_batch, masked_self_state_motion_loss
from .self_state_trainer import SelfStateTrainer, evaluate_self_state
from .online_state import OnlineTrajectory, build_latest_self_state, refresh_online_trajectory
from .online_self_state_trainer import OnlineSelfStateTrainer
from .joint import (
    JointLoss,
    JointTrainRecord,
    evaluate_joint_clean,
    masked_joint_clean_loss,
    save_joint_checkpoint,
    train_joint_clean_batch,
)

__all__ = [
    "masked_bridge_rate_loss",
    "MultiKCleanTrainer",
    "evaluate_multik_clean",
    "BridgeBatch",
    "build_bridge_batch",
    "CleanBatch",
    "CleanExample",
    "CleanSplit",
    "MotionLoss",
    "ChiLoss",
    "TrainRecord",
    "collate_clean_examples",
    "make_clean_example",
    "masked_remaining_motion_loss",
    "masked_periodic_chi_loss",
    "masked_chi_loss",
    "save_checkpoint",
    "train_clean_batch",
    "deterministic_clean_split",
    "build_self_state_batch",
    "SelfStateBatch",
    "masked_self_state_motion_loss",
    "SelfStateTrainer",
    "evaluate_self_state",
    "OnlineTrajectory",
    "build_latest_self_state",
    "refresh_online_trajectory",
    "OnlineSelfStateTrainer",
    "JointLoss",
    "JointTrainRecord",
    "evaluate_joint_clean",
    "masked_joint_clean_loss",
    "save_joint_checkpoint",
    "train_joint_clean_batch",
]
from .coordinate import NextXYZLoss, masked_next_xyz_loss
from .coordinate_step import (
    NextXYZBatch, NextXYZObjective, NextXYZTrainRecord,
    next_xyz_objective, train_next_xyz_step,
)

__all__ += ['NextXYZLoss', 'masked_next_xyz_loss', 'NextXYZBatch', 'NextXYZObjective',
            'NextXYZTrainRecord', 'next_xyz_objective', 'train_next_xyz_step']
from .joint_trajectory import JointReferenceTrajectory, build_joint_reference

__all__ += ['JointReferenceTrajectory', 'build_joint_reference']
from .diffusion import (
    DiffusionTrainConfig,
    LoadedDiffusionCheckpoint,
    SelectedComplexes,
    coordinate_rmsd,
    evaluate_diffusion_sampler,
    load_apo2mol_slices,
    load_diffusion_checkpoint,
    run_diffusion_training,
    sample_checkpoint_on_complex,
    save_diffusion_checkpoint,
    select_convertible_complexes,
    train_diffusion_model,
    build_scheduled_self_state,
)

__all__ += [
    'DiffusionTrainConfig',
    'LoadedDiffusionCheckpoint',
    'SelectedComplexes',
    'coordinate_rmsd',
    'evaluate_diffusion_sampler',
    'load_apo2mol_slices',
    'load_diffusion_checkpoint',
    'run_diffusion_training',
    'sample_checkpoint_on_complex',
    'save_diffusion_checkpoint',
    'select_convertible_complexes',
    'train_diffusion_model',
    'build_scheduled_self_state',
]
