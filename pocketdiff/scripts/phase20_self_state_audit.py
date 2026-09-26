"""Audit labels and bridge_rate range on autonomous Phase19 states."""

import argparse
import json
from pathlib import Path

import torch

from pocketdiff.preprocessing import load_cached_clean_examples
from pocketdiff.scripts.phase18_late_step_diagnosis import load_baseline_data
from pocketdiff.training import build_self_state_batch


def load_trace(path):
    payload = torch.load(path, map_location='cpu')
    if payload.get('format') != 'pocketdiff-clean-rollout-v1':
        raise ValueError('unexpected trajectory format: ' + str(path))
    return payload


def audit_variant(clean, trace):
    positions = trace['positions']
    if positions.shape[0] != 21:
        raise ValueError('expected 21 autonomous states')
    rows, per_k = [], []
    for k in range(20):
        state = build_self_state_batch(clean, positions, torch.full_like(clean.pocket_k, k))
        valid = state.frame_valid
        fraction = (20-k)/20.0
        tr = state.target_translation_local[valid]
        rot = state.target_rotvec_local[valid]
        rot_norm = torch.linalg.vector_norm(rot, dim=-1)
        normalized_tr = tr / fraction
        normalized_rot = rot / fraction
        current = state.protein_pos
        atom_valid = valid[state.atom_to_residue_global] & (state.batch_protein >= 0)
        holo_rmsd = ((current[atom_valid]-state.protein_pos_holo[atom_valid]).square().sum(-1).mean().sqrt())
        row = dict(k=k, remaining_fraction=fraction,
                   valid_residue_count=int(valid.sum()), invalid_residue_count=int((~valid).sum()),
                   translation_target_rms=float(tr.square().mean().sqrt()),
                   rotation_target_rms=float(rot.square().mean().sqrt()),
                   translation_rate_rms=float(normalized_tr.square().mean().sqrt()),
                   rotation_rate_rms=float(normalized_rot.square().mean().sqrt()),
                   rotation_norm_max=float(rot_norm.max()) if rot.numel() else 0.0,
                   normalized_rotation_norm_max=float((rot_norm/fraction).max()) if rot.numel() else 0.0,
                   pi_rate_bound=float(torch.pi),
                   rotation_bound_violations=int((rot_norm > torch.pi*fraction + 1e-5).sum()),
                   holo_rmsd=float(holo_rmsd))
        rows.append(row)
        per_k.append(row)
    k0 = build_self_state_batch(clean, positions, torch.zeros_like(clean.pocket_k))
    clean_k0_error = max(float((k0.target_translation_local-clean.target_translation_local).abs().max()),
                         float((k0.target_rotvec_local-clean.target_rotvec_local).abs().max()))
    return dict(per_k=per_k, k0_label_max_error=clean_k0_error,
                max_rotation_bound_violations=max(row['rotation_bound_violations'] for row in rows),
                any_invalid_frames=any(row['invalid_residue_count'] for row in rows))


def main():
    parser = argparse.ArgumentParser()
    root = Path('.codex-tasks/pocketdiff-development')
    parser.add_argument('--baseline-dir', type=Path, default=root/'phase17-multik-clean-training/raw/run')
    parser.add_argument('--rollout-dir', type=Path, default=root/'phase19-clean-rollout/raw/run')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    baseline, batches = load_baseline_data(args.baseline_dir)
    result = dict(mode='self_state_label_audit', geometry_version='geometry-v2', roles={},
                  scope='detached Phase19 autonomous states; labels are audit-only')
    for role, clean in batches.items():
        result['roles'][role] = {}
        for variant in ('phase17', 'phase18'):
            path = args.rollout_dir/(role+'_'+variant+'_trajectory.pt')
            trace = load_trace(path)
            if trace['sample_ids'] != clean.sample_ids:
                raise ValueError('trajectory sample order differs: ' + str(path))
            result['roles'][role][variant] = audit_variant(clean, trace)
    result['passed'] = all(
        result['roles'][role][variant]['k0_label_max_error'] < 1e-6
        for role in result['roles'] for variant in result['roles'][role]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(dict(passed=result['passed'])))
    if not result['passed']:
        raise RuntimeError('self-state k0 label audit failed')


if __name__ == '__main__':
    main()
