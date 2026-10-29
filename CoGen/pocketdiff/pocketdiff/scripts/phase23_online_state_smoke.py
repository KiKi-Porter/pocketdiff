"""Real checkpoint smoke for online autonomous-state refresh."""

import argparse
import json
from pathlib import Path

import torch

from pocketdiff.models import PocketDiffModel
from pocketdiff.scripts.phase17_multik_train import _sha, _write_json
from pocketdiff.scripts.phase18_late_step_diagnosis import load_baseline_data
from pocketdiff.training import build_latest_self_state, refresh_online_trajectory
from pocketdiff.training.multik import endpoint_fingerprint
from pocketdiff.training.self_state_trainer import trajectory_fingerprint


def load_model(path):
    payload = torch.load(path, map_location='cpu')
    model = PocketDiffModel(**payload['model_config'])
    model.load_state_dict(payload['model_state_dict'], strict=True)
    return model.eval(), payload


def main():
    parser = argparse.ArgumentParser()
    root = Path('.codex-tasks/pocketdiff-development')
    parser.add_argument('--baseline-dir', type=Path, default=root/'phase17-multik-clean-training/raw/run')
    parser.add_argument('--rate-dir', type=Path, default=root/'phase18-late-step-diagnosis/raw/run')
    parser.add_argument('--self-state-dir', type=Path, default=root/'phase22-self-state-training/raw/run')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output directory must be empty')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    baseline, batches = load_baseline_data(args.baseline_dir)
    paths = {
        'phase17': args.baseline_dir/'checkpoint_0400.pt',
        'phase18': args.rate_dir/'checkpoint_0400.pt',
        'phase22': args.self_state_dir/'checkpoint_0400.pt',
    }
    models, payloads = {}, {}
    for name, path in paths.items():
        models[name], payloads[name] = load_model(path)
    fingerprints = {role: endpoint_fingerprint(clean) for role, clean in batches.items()}
    results, trajectory_hashes = {}, {}
    input_unchanged = True
    rng_unchanged = True
    for role, clean in batches.items():
        results[role] = {}
        for name, model in models.items():
            before = {field: value.clone() for field, value in clean.__dict__.items()
                      if isinstance(value, torch.Tensor)}
            rng = torch.get_rng_state().clone()
            first = refresh_online_trajectory(model, clean)
            second = refresh_online_trajectory(model, clean)
            repeated_equal = torch.equal(first.positions, second.positions)
            input_unchanged &= all(torch.equal(getattr(clean, field), value)
                                   for field, value in before.items())
            rng_unchanged &= torch.equal(rng, torch.get_rng_state())
            if not repeated_equal:
                raise RuntimeError('repeated online refresh differs: ' + role + '/' + name)
            # Construct a mixed state batch to ensure the refresh is trainable input.
            mixed_k = torch.tensor([0, 3, 7, 10, 13, 16, 18, 19], dtype=torch.long)
            latest = build_latest_self_state(clean, first, mixed_k)
            path = args.output_dir/(role+'_'+name+'_online_trajectory.pt')
            torch.save(dict(format='pocketdiff-online-trajectory-v1', sample_ids=clean.sample_ids,
                            positions=first.positions, frame_valid=first.frame_valid,
                            update_valid=first.update_valid), path)
            trajectory_hashes[role+'/'+name] = _sha(path)
            results[role][name] = dict(
                trajectory_sha256=_sha(path), position_fingerprint=trajectory_fingerprint(first.positions),
                repeated_equal=repeated_equal, step_count=int(first.positions.shape[0]),
                finite=bool(torch.isfinite(first.positions).all()),
                valid_frames=int(first.frame_valid.sum()), invalid_frames=int((~first.frame_valid).sum()),
                mixed_k=mixed_k.tolist(), mixed_valid_residues=int(latest.frame_valid.sum()),
                final_apo_displacement=float((first.positions[-1]-clean.apo_pos_ref).square().sum(-1).mean().sqrt()),
                model_parameterization=model.motion_parameterization,
            )
    source = {str(path): _sha(path) for path in paths.values()}
    report = dict(
        passed=input_unchanged and rng_unchanged and all(
            value['repeated_equal'] and value['finite'] and value['step_count'] == 21
            for variants in results.values() for value in variants.values()),
        mode='online_state_refresh_smoke', roles={role: clean.sample_ids for role, clean in batches.items()},
        source_sha256=source, endpoint_fingerprints=fingerprints,
        trajectory_sha256=trajectory_hashes, input_unchanged=input_unchanged,
        rng_unchanged=rng_unchanged, results=results,
        scope='refresh and detached self-state input only; no optimization',
    )
    _write_json(args.output_dir/'report.json', report)
    _write_json(args.output_dir/'code_fingerprints.json', {
        str(path): _sha(path) for path in sorted(Path('pocketdiff').rglob('*.py'))})
    print(json.dumps(dict(passed=report['passed'], input_unchanged=input_unchanged,
                          rng_unchanged=rng_unchanged)), flush=True)
    if not report['passed']:
        raise RuntimeError('online refresh smoke failed')


if __name__ == '__main__':
    main()
