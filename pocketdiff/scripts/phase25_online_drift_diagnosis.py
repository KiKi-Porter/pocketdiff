"""Read-only state and motion-scale diagnosis for Phase 24 checkpoints."""

import argparse
import json
from pathlib import Path
import time

import torch

from pocketdiff.evaluation import (
    summarize_online_self_state_scales,
    summarize_online_trajectory,
)
from pocketdiff.evaluation.clean_rollout import score_clean_rollout
from pocketdiff.models import PocketDiffModel
from pocketdiff.sampling.clean_rollout import INFERENCE_FIELDS, run_clean_rollout
from pocketdiff.scripts.phase17_multik_train import _sha, _write_json
from pocketdiff.scripts.phase18_late_step_diagnosis import load_baseline_data
from pocketdiff.training import OnlineTrajectory, refresh_online_trajectory
from pocketdiff.training.multik import endpoint_fingerprint
from pocketdiff.training.self_state_trainer import trajectory_fingerprint


CHECKPOINT_STEPS = (0, 50, 100, 150, 200)


def _load_checkpoint(path, clean):
    rng_before = torch.get_rng_state().clone()
    payload = torch.load(path, map_location='cpu')
    if payload.get('format') != 'pocketdiff-online-self-state-v1':
        raise ValueError('unexpected Phase 24 checkpoint format: ' + str(path))
    if payload.get('sample_ids') != clean.sample_ids:
        raise ValueError('checkpoint sample order differs: ' + str(path))
    if payload.get('endpoint_fingerprint') != endpoint_fingerprint(clean):
        raise ValueError('checkpoint endpoint fingerprint differs: ' + str(path))
    trajectory = OnlineTrajectory(
        payload['trajectory_positions'].detach().cpu().clone(),
        payload['trajectory_frame_valid'].detach().cpu().clone(),
        payload['trajectory_update_valid'].detach().cpu().clone(),
    )
    if payload.get('trajectory_fingerprint') != trajectory_fingerprint(trajectory.positions):
        raise ValueError('checkpoint trajectory fingerprint differs: ' + str(path))
    model = PocketDiffModel(**payload['model_config'])
    model.load_state_dict(payload['model_state_dict'], strict=True)
    torch.set_rng_state(rng_before)
    model.eval()
    return model, trajectory, payload


@torch.no_grad()
def _autonomous_summary(model, clean):
    inputs = {name: getattr(clean, name) for name in INFERENCE_FIELDS}
    trace = run_clean_rollout(model, inputs)
    metrics = score_clean_rollout(
        trace, inputs, clean.protein_pos_holo, clean.sample_ids,
    )
    return dict(
        apo_holo_rmsd=metrics['per_step'][0]['holo_rmsd'],
        final_holo_rmsd=metrics['per_step'][-1]['holo_rmsd'],
        final_apo_displacement_rmsd=metrics['per_step'][-1]['apo_displacement_rmsd'],
        max_apo_displacement=metrics['per_step'][-1]['max_apo_displacement'],
        per_step=[dict(step=row['step'], holo_rmsd=row['holo_rmsd'],
                       apo_displacement_rmsd=row['apo_displacement_rmsd'])
                  for row in metrics['per_step']],
    )


def _checkpoint_diagnosis(model, clean, trajectory):
    trajectory_summary = summarize_online_trajectory(clean, trajectory)
    scales = summarize_online_self_state_scales(model, clean, trajectory)
    return dict(
        trajectory=trajectory_summary,
        self_state_scales=scales,
        autonomous=_autonomous_summary(model, clean),
    )


def main():
    parser = argparse.ArgumentParser()
    root = Path('.codex-tasks/pocketdiff-development')
    parser.add_argument('--phase24-dir', type=Path,
                        default=root / 'phase24-online-self-state-training/raw/run')
    parser.add_argument('--baseline-dir', type=Path,
                        default=root / 'phase17-multik-clean-training/raw/run')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output directory must be empty; preserve previous diagnostics')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.monotonic()
    process_rng_before = torch.get_rng_state().clone()
    try:
        baseline, batches = load_baseline_data(args.baseline_dir)
        phase24_report = json.loads((args.phase24_dir / 'report.json').read_text())
        if not phase24_report.get('engineering_passed'):
            raise ValueError('Phase24 engineering report is not passed')
        endpoint_fingerprints = {
            role: endpoint_fingerprint(batch) for role, batch in batches.items()
        }
        for role, value in endpoint_fingerprints.items():
            if value != baseline['preparation'][role + '_endpoint_fingerprint']:
                raise ValueError('endpoint fingerprint differs from historical baseline: ' + role)
        checkpoint_paths = {
            step: args.phase24_dir / ('checkpoint_%04d.pt' % step)
            for step in CHECKPOINT_STEPS
        }
        source_sha = {
            str(path): _sha(path)
            for path in checkpoint_paths.values()
        }
        source_sha[str(args.phase24_dir / 'report.json')] = _sha(
            args.phase24_dir / 'report.json',
        )
        _write_json(args.output_dir / 'provenance.json', dict(
            phase24_report_sha256=_sha(args.phase24_dir / 'report.json'),
            checkpoint_sha256=source_sha,
            endpoint_fingerprints=endpoint_fingerprints,
            scope='read-only trajectory, self-state scale and autonomous diagnosis',
        ))
        _write_json(args.output_dir / 'code_fingerprints.json', {
            str(path): _sha(path)
            for path in sorted(Path('pocketdiff').rglob('*.py'))
        })

        results = {}
        guards = dict()
        for step, path in checkpoint_paths.items():
            if not path.exists():
                raise FileNotFoundError(str(path))
            results[str(step)] = {}
            for role, clean in batches.items():
                model, train_trajectory, payload = _load_checkpoint(
                    path, batches['train'],
                )
                if role == 'train':
                    trajectory = train_trajectory
                else:
                    trajectory = refresh_online_trajectory(model, clean)
                before_input = endpoint_fingerprint(clean)
                before_rng = torch.get_rng_state().clone()
                before_parameters = {
                    name: value.clone()
                    for name, value in model.state_dict().items()
                }
                value = _checkpoint_diagnosis(model, clean, trajectory)
                after_input = endpoint_fingerprint(clean)
                parameters_unchanged = all(
                    torch.equal(model.state_dict()[name], parameter)
                    for name, parameter in before_parameters.items()
                )
                rng_unchanged = torch.equal(before_rng, torch.get_rng_state())
                input_unchanged = before_input == after_input
                finite = all(
                    torch.isfinite(torch.tensor(number))
                    for section in (
                        value['trajectory']['per_step'],
                        value['self_state_scales']['per_k'],
                        value['autonomous']['per_step'],
                    )
                    for row in section
                    for number in row.values()
                    if isinstance(number, (float, int))
                )
                invalid_frames = sum(
                    row['frame_invalid_count']
                    for row in value['trajectory']['per_step']
                )
                value['guards'] = dict(
                    finite=bool(finite),
                    invalid_frames=invalid_frames,
                    input_unchanged=input_unchanged,
                    parameters_unchanged=parameters_unchanged,
                    rng_unchanged=rng_unchanged,
                )
                results[str(step)][role] = value
                guards[str(step) + '/' + role] = all((
                    finite, invalid_frames == 0, input_unchanged,
                    parameters_unchanged, rng_unchanged,
                ))

        summary = {}
        for step in CHECKPOINT_STEPS:
            row = results[str(step)]
            summary[str(step)] = {}
            for role in ('train', 'holdout'):
                value = row[role]
                trajectory_final = value['trajectory']['final']
                scales = value['self_state_scales']
                summary[str(step)][role] = dict(
                    trajectory_final_apo_rmsd=trajectory_final['apo_rmsd'],
                    trajectory_final_holo_rmsd=trajectory_final['holo_rmsd'],
                    trajectory_max_apo_displacement=trajectory_final['max_apo_displacement'],
                    max_target_translation_rms=scales['max_translation_target_rms'],
                    max_prediction_translation_rms=scales['max_translation_prediction_rms'],
                    max_target_rotation_rms=scales['max_rotation_target_rms'],
                    max_prediction_rotation_rms=scales['max_rotation_prediction_rms'],
                    mean_self_state_loss=scales['mean_loss'],
                    autonomous_final_holo_rmsd=value['autonomous']['final_holo_rmsd'],
                )
        report = dict(
            passed=all(guards.values()),
            mode='phase25_online_drift_diagnosis',
            elapsed_seconds=time.monotonic() - started,
            source_sha256=source_sha,
            endpoint_fingerprints=endpoint_fingerprints,
            checkpoint_steps=list(CHECKPOINT_STEPS),
            guards=guards,
            summary=summary,
            results=results,
            interpretation=dict(
                comparison_checkpoint_50_to_0={
                    role: dict(
                        trajectory_final_apo_rmsd_delta=(
                            summary['50'][role]['trajectory_final_apo_rmsd']
                            - summary['0'][role]['trajectory_final_apo_rmsd']
                        ),
                        max_target_translation_rms_delta=(
                            summary['50'][role]['max_target_translation_rms']
                            - summary['0'][role]['max_target_translation_rms']
                        ),
                        max_prediction_translation_rms_delta=(
                            summary['50'][role]['max_prediction_translation_rms']
                            - summary['0'][role]['max_prediction_translation_rms']
                        ),
                        autonomous_final_holo_rmsd_delta=(
                            summary['50'][role]['autonomous_final_holo_rmsd']
                            - summary['0'][role]['autonomous_final_holo_rmsd']
                        ),
                    )
                    for role in ('train', 'holdout')
                },
                next_step='Use these read-only scales to choose the smallest '
                          'objective or representation change; do not add steps '
                          'or select an intermediate holo-scored state.',
            ),
        )
        report['guards']['process_rng_unchanged'] = torch.equal(
            process_rng_before, torch.get_rng_state(),
        )
        report['passed'] = report['passed'] and report['guards']['process_rng_unchanged']
        _write_json(args.output_dir / 'report.json', report)
        print(json.dumps(dict(
            passed=report['passed'],
            checkpoint_count=len(CHECKPOINT_STEPS),
            guard_count=len(guards),
        )), flush=True)
        if not report['passed']:
            raise RuntimeError('online drift diagnosis guard failed')
    except Exception as exc:
        _write_json(args.output_dir / 'failure.json', dict(
            error_type=type(exc).__name__,
            error=str(exc),
            elapsed_seconds=time.monotonic() - started,
        ))
        raise


if __name__ == '__main__':
    main()
