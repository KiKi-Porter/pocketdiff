"""Real autonomous self-state forward/loss/gradient contract audit."""

import argparse
import json
from pathlib import Path

import torch

from pocketdiff.models import PocketDiffModel
from pocketdiff.scripts.phase18_late_step_diagnosis import load_baseline_data
from pocketdiff.training import build_self_state_batch, masked_self_state_motion_loss


def _load_positions(path, sample_ids):
    payload = torch.load(path, map_location='cpu')
    if payload.get('format') != 'pocketdiff-clean-rollout-v1':
        raise ValueError('unexpected rollout format: ' + str(path))
    if payload.get('sample_ids') != sample_ids:
        raise ValueError('rollout sample order differs: ' + str(path))
    positions = payload['positions']
    if positions.shape[0] != 21 or not torch.isfinite(positions).all():
        raise ValueError('rollout must contain finite positions[0..20]')
    return positions


def audit_batch(model, clean, positions, pocket_k):
    before = {name: value.clone() for name, value in clean.__dict__.items()
              if isinstance(value, torch.Tensor)}
    state = build_self_state_batch(clean, positions, pocket_k)
    model.train()
    model.zero_grad(set_to_none=True)
    prediction = model(**state.model_kwargs())
    motion_loss = masked_self_state_motion_loss(
        prediction, state.target_translation_local, state.target_rotvec_local, state.frame_valid)
    if not torch.isfinite(motion_loss.loss):
        raise FloatingPointError('self-state loss is non-finite')
    motion_loss.loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError('self-state gradients are missing or non-finite')
    if not any(float(g.abs().sum()) > 0 for g in gradients):
        raise FloatingPointError('self-state gradients are all zero')
    input_unchanged = all(torch.equal(getattr(clean, name), value) for name, value in before.items())
    return dict(
        pocket_k=pocket_k.tolist(), valid_residue_count=motion_loss.valid_residue_count,
        loss=float(motion_loss.loss.detach()),
        translation_loss=float(motion_loss.translation_loss.detach()),
        rotation_loss=float(motion_loss.rotation_loss.detach()),
        max_translation_target=float(state.target_translation_local.abs().max()),
        max_rotation_target=float(state.target_rotvec_local.abs().max()),
        max_prediction_translation=float(prediction.remaining_translation_local.abs().max()),
        max_prediction_rotation=float(prediction.remaining_rotvec_local.abs().max()),
        gradient_norm=float(torch.sqrt(sum(g.detach().square().sum() for g in gradients))),
        input_unchanged=input_unchanged,
        detached_state=(not state.protein_pos.requires_grad and
                        not state.target_translation_local.requires_grad and
                        not state.target_rotvec_local.requires_grad),
        finite=True,
    )


def main():
    parser = argparse.ArgumentParser()
    root = Path('.codex-tasks/pocketdiff-development')
    parser.add_argument('--baseline-dir', type=Path,
                        default=root/'phase17-multik-clean-training/raw/run')
    parser.add_argument('--rollout-dir', type=Path,
                        default=root/'phase19-clean-rollout/raw/run')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    baseline, batches = load_baseline_data(args.baseline_dir)
    checkpoint = args.checkpoint or args.baseline_dir/'checkpoint_0400.pt'
    payload = torch.load(checkpoint, map_location='cpu')
    model = PocketDiffModel(**payload['model_config'])
    model.load_state_dict(payload['model_state_dict'], strict=True)
    result = dict(mode='self_state_remaining_contract', checkpoint=str(checkpoint), roles={},
                  model_parameterization=model.motion_parameterization,
                  scope='detached autonomous Phase19 states; no optimization performed')
    if model.motion_parameterization != 'remaining':
        raise ValueError('contract audit requires physical remaining model mode')
    for role, clean in batches.items():
        path = args.rollout_dir/(role+'_phase18_trajectory.pt')
        positions = _load_positions(path, clean.sample_ids)
        # Mixed graph times exercise both early and late autonomous states in one call.
        pocket_k = torch.tensor([0, 3, 7, 10, 13, 16, 18, 19], dtype=torch.long)
        if len(clean.sample_ids) != pocket_k.numel():
            raise ValueError('fixed audit expects eight samples per role')
        result['roles'][role] = audit_batch(model, clean, positions, pocket_k)
    result['passed'] = (
        result['model_parameterization'] == 'remaining' and
        all(item['finite'] and item['input_unchanged'] and item['detached_state']
            and item['gradient_norm'] > 0 for item in result['roles'].values())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(dict(passed=result['passed'], model_parameterization=model.motion_parameterization)))
    if not result['passed']:
        raise RuntimeError('self-state remaining contract audit failed')


if __name__ == '__main__':
    main()
