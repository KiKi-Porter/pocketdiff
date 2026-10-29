"""Read-only diagnostics for online autonomous-state training."""

import math

import torch

from pocketdiff.training.online_state import OnlineTrajectory, build_latest_self_state
from pocketdiff.training.self_state_trainer import trajectory_fingerprint


def _as_trajectory(value):
    if isinstance(value, OnlineTrajectory):
        return value
    raise TypeError('diagnostics require an OnlineTrajectory')


def _rms(values):
    return float(values.square().mean().sqrt())


@torch.no_grad()
def summarize_online_trajectory(clean, trajectory):
    """Summarize fixed-apo and holo distances for all 21 states."""
    trajectory = _as_trajectory(trajectory)
    if trajectory.positions.shape[1:] != clean.protein_pos.shape:
        raise ValueError('trajectory atom shape differs from clean batch')
    if not torch.equal(trajectory.positions[0], clean.apo_pos_ref):
        raise ValueError('trajectory step 0 must equal apo coordinates')
    atom_valid = clean.frame_valid[clean.atom_to_residue_global]
    rows = []
    graph_count = len(clean.sample_ids)
    for step, positions in enumerate(trajectory.positions):
        for graph, sample_id in enumerate(clean.sample_ids):
            atoms = atom_valid & (clean.batch_protein == graph)
            residues = clean.batch_residue == graph
            if not bool(atoms.any()):
                raise ValueError('trajectory graph has no valid atoms: ' + sample_id)
            cumulative = torch.linalg.vector_norm(
                positions[atoms] - clean.apo_pos_ref[atoms], dim=-1,
            )
            rows.append(dict(
                sample_id=sample_id,
                step=step,
                apo_rmsd=_rms(positions[atoms] - clean.apo_pos_ref[atoms]),
                holo_rmsd=_rms(positions[atoms] - clean.protein_pos_holo[atoms]),
                max_apo_displacement=float(cumulative.max()),
                frame_invalid_count=int((~trajectory.frame_valid[step, residues]).sum()),
                scored_atom_count=int(atoms.sum()),
            ))
    per_step = []
    for step in range(21):
        subset = [row for row in rows if row['step'] == step]
        per_step.append(dict(
            step=step,
            apo_rmsd=sum(row['apo_rmsd'] for row in subset) / graph_count,
            holo_rmsd=sum(row['holo_rmsd'] for row in subset) / graph_count,
            max_apo_displacement=max(row['max_apo_displacement'] for row in subset),
            frame_invalid_count=sum(row['frame_invalid_count'] for row in subset),
        ))
    values = [row[key] for row in per_step
              for key in ('apo_rmsd', 'holo_rmsd', 'max_apo_displacement')]
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError('non-finite trajectory summary')
    return dict(
        trajectory_fingerprint=trajectory_fingerprint(trajectory.positions),
        step_count=21,
        per_step=per_step,
        final=per_step[-1],
        per_graph=rows,
    )


@torch.no_grad()
def summarize_online_self_state_scales(model, clean, trajectory):
    """Measure target/prediction amplitudes on all latest self-state times."""
    trajectory = _as_trajectory(trajectory)
    was_training = model.training
    rng_before = torch.get_rng_state().clone()
    model.eval()
    rows = []
    try:
        for k in range(20):
            pocket_k = torch.full((len(clean.sample_ids),), k, dtype=torch.long)
            batch = build_latest_self_state(clean, trajectory, pocket_k)
            prediction = model(**batch.model_kwargs())
            valid = prediction.frame_valid & batch.frame_valid
            if not bool(valid.any()):
                raise ValueError('no valid self-state residue at k=%d' % k)
            target_translation = batch.target_translation_local[valid]
            target_rotation = batch.target_rotvec_local[valid]
            predicted_translation = prediction.remaining_translation_local[valid]
            predicted_rotation = prediction.remaining_rotvec_local[valid]
            target_translation_rms = _rms(target_translation)
            target_rotation_rms = _rms(target_rotation)
            prediction_translation_rms = _rms(predicted_translation)
            prediction_rotation_rms = _rms(predicted_rotation)
            translation_error = _rms(predicted_translation - target_translation)
            rotation_error = _rms(predicted_rotation - target_rotation)
            values = (
                target_translation_rms, target_rotation_rms,
                prediction_translation_rms, prediction_rotation_rms,
                translation_error, rotation_error,
            )
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError('non-finite self-state scale at k=%d' % k)
            rows.append(dict(
                k=k,
                t=199 - 10 * k,
                valid_residue_count=int(valid.sum()),
                target_translation_rms=target_translation_rms,
                target_rotation_rms=target_rotation_rms,
                prediction_translation_rms=prediction_translation_rms,
                prediction_rotation_rms=prediction_rotation_rms,
                translation_error_rms=translation_error,
                rotation_error_rms=rotation_error,
                translation_prediction_target_ratio=prediction_translation_rms /
                max(target_translation_rms, 1e-12),
                rotation_prediction_target_ratio=prediction_rotation_rms /
                max(target_rotation_rms, 1e-12),
                loss=translation_error ** 2 + rotation_error ** 2,
            ))
    finally:
        model.train(was_training)
    if not torch.equal(rng_before, torch.get_rng_state()):
        raise RuntimeError('self-state scale diagnostics consumed torch RNG')
    return dict(
        mode='online_self_state_scale',
        trajectory_fingerprint=trajectory_fingerprint(trajectory.positions),
        per_k=rows,
        mean_loss=sum(row['loss'] for row in rows) / len(rows),
        max_translation_target_rms=max(row['target_translation_rms'] for row in rows),
        max_rotation_target_rms=max(row['target_rotation_rms'] for row in rows),
        max_translation_prediction_rms=max(
            row['prediction_translation_rms'] for row in rows
        ),
        max_rotation_prediction_rms=max(
            row['prediction_rotation_rms'] for row in rows
        ),
    )


__all__ = ['summarize_online_self_state_scales', 'summarize_online_trajectory']
