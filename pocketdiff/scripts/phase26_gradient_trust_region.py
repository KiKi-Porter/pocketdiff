"""Phase 26: one fixed gradient trust-region online training comparison."""

import argparse
import json
from pathlib import Path
import time

import torch

from pocketdiff.evaluation import summarize_online_trajectory
from pocketdiff.scripts.phase17_multik_train import _sha, _write_json
from pocketdiff.scripts.phase18_late_step_diagnosis import load_baseline_data
from pocketdiff.scripts.phase24_online_self_state_train import (
    FIXED_EVAL_EVERY,
    FIXED_REFRESH_INTERVAL,
    FIXED_SEED,
    FIXED_STEPS,
    _evaluate,
    _reload_error,
)
from pocketdiff.training import OnlineSelfStateTrainer, OnlineTrajectory
from pocketdiff.training.multik import endpoint_fingerprint


MAX_GRAD_NORM = 1.0
LEARNING_RATE = 1e-3


def main():
    parser = argparse.ArgumentParser()
    root = Path('.codex-tasks/pocketdiff-development')
    parser.add_argument('--baseline-dir', type=Path,
                        default=root / 'phase17-multik-clean-training/raw/run')
    parser.add_argument('--phase24-dir', type=Path,
                        default=root / 'phase24-online-self-state-training/raw/run')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output directory must be empty; preserve previous experiments')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.monotonic()
    try:
        baseline, batches = load_baseline_data(args.baseline_dir)
        phase24_report_path = args.phase24_dir / 'report.json'
        phase24_report = json.loads(phase24_report_path.read_text())
        endpoint_fingerprints = {
            role: endpoint_fingerprint(batch) for role, batch in batches.items()
        }
        for role, value in endpoint_fingerprints.items():
            if value != baseline['preparation'][role + '_endpoint_fingerprint']:
                raise ValueError('endpoint fingerprint differs from baseline: ' + role)
        initial_checkpoint = args.baseline_dir / 'checkpoint_0400.pt'
        initial_payload = torch.load(initial_checkpoint, map_location='cpu')
        provenance = dict(
            baseline_dir=str(args.baseline_dir.resolve()),
            initial_checkpoint_sha256=_sha(initial_checkpoint),
            phase24_report_sha256=_sha(phase24_report_path),
            endpoint_fingerprints=endpoint_fingerprints,
            fixed=dict(steps=FIXED_STEPS, eval_every=FIXED_EVAL_EVERY,
                       refresh_interval=FIXED_REFRESH_INTERVAL, seed=FIXED_SEED,
                       learning_rate=LEARNING_RATE,
                       max_grad_norm=MAX_GRAD_NORM),
            only_change='max_grad_norm 10.0 -> 1.0 relative to Phase24',
        )
        _write_json(args.output_dir / 'provenance.json', provenance)
        _write_json(args.output_dir / 'code_fingerprints.json', {
            str(path): _sha(path)
            for path in sorted(Path('pocketdiff').rglob('*.py'))
        })
        trainer = OnlineSelfStateTrainer.from_initial_checkpoint(
            initial_checkpoint, batches['train'],
            refresh_interval=FIXED_REFRESH_INTERVAL,
            seed=FIXED_SEED, learning_rate=LEARNING_RATE,
            max_grad_norm=MAX_GRAD_NORM,
        )
        initial_weights_identical = all(
            torch.equal(value, initial_payload['model_state_dict'][name])
            for name, value in trainer.model.state_dict().items()
        )
        metadata = dict(
            provenance=provenance,
            initial_trajectory_fingerprint=trainer.trajectory_fingerprint,
            torch_version=torch.__version__,
            cpu_threads=torch.get_num_threads(),
        )
        trainer.save_checkpoint(args.output_dir / 'checkpoint_0000.pt',
                                metadata=metadata)
        evaluations = [_evaluate(trainer, batches, args.output_dir)]
        history_path = args.output_dir / 'history.jsonl'
        history = []
        with history_path.open('w') as handle:
            while trainer.step_count < FIXED_STEPS:
                record = trainer.step()
                history.append(record)
                handle.write(json.dumps(record, allow_nan=False) + '\n')
                handle.flush()
                if (trainer.step_count % FIXED_EVAL_EVERY == 0
                        or trainer.step_count == FIXED_STEPS):
                    checkpoint = args.output_dir / ('checkpoint_%04d.pt'
                                                    % trainer.step_count)
                    trainer.save_checkpoint(checkpoint, metadata=metadata)
                    evaluations.append(_evaluate(trainer, batches, args.output_dir))
                    print(json.dumps(dict(
                        step=trainer.step_count,
                        train_final=evaluations[-1]['autonomous']['train']['final_holo_rmsd'],
                        holdout_final=evaluations[-1]['autonomous']['holdout']['final_holo_rmsd'],
                    )), flush=True)

        final_checkpoint = args.output_dir / 'checkpoint_0200.pt'
        reload_error = _reload_error(
            final_checkpoint, batches['train'], trainer.model, trainer.trajectory,
        )
        checkpoint_50 = torch.load(args.output_dir / 'checkpoint_0050.pt',
                                   map_location='cpu')
        trajectory_50 = OnlineTrajectory(
            checkpoint_50['trajectory_positions'],
            checkpoint_50['trajectory_frame_valid'],
            checkpoint_50['trajectory_update_valid'],
        )
        trajectory_50_summary = summarize_online_trajectory(
            batches['train'], trajectory_50,
        )
        refresh_records = [
            row for row in history
            if row['refreshed'] and row['step'] % FIXED_REFRESH_INTERVAL == 0
        ]
        guards = dict(
            initial_weights_identical=initial_weights_identical,
            fixed_step_count=trainer.step_count == FIXED_STEPS,
            fixed_refresh_schedule=len(refresh_records) == FIXED_STEPS // FIXED_REFRESH_INTERVAL,
            refresh_steps=[row['step'] for row in refresh_records]
            == list(range(FIXED_REFRESH_INTERVAL, FIXED_STEPS + 1,
                          FIXED_REFRESH_INTERVAL)),
            k_coverage=bool((trainer.k_histogram > 0).all()),
            checkpoint_reload_exact=reload_error == 0,
            inputs_unchanged=all(
                endpoint_fingerprint(batches[role]) == value
                for role, value in endpoint_fingerprints.items()
            ),
            finite_parameters=all(torch.isfinite(parameter).all()
                                  for parameter in trainer.model.parameters()),
            complete_evaluations=[row['step'] for row in evaluations]
            == [0, 50, 100, 150, 200],
        )
        initial = evaluations[0]
        final = evaluations[-1]
        apo_train = initial['autonomous']['train']['apo_holo_rmsd']
        apo_holdout = initial['autonomous']['holdout']['apo_holo_rmsd']
        final_train = final['autonomous']['train']['final_holo_rmsd']
        final_holdout = final['autonomous']['holdout']['final_holo_rmsd']
        engineering_passed = all(guards.values())
        learning_goal_met = bool(final_train < apo_train and final_holdout < apo_holdout)
        report = dict(
            passed=engineering_passed,
            engineering_passed=engineering_passed,
            learning_goal_met=learning_goal_met,
            mode='phase26_gradient_trust_region',
            steps=trainer.step_count,
            max_grad_norm=MAX_GRAD_NORM,
            elapsed_seconds=time.monotonic() - started,
            provenance=provenance,
            guards=guards,
            reload_max_error=reload_error,
            k_histogram=trainer.k_histogram.tolist(),
            max_gradient_norm_before_clip=max(
                row['gradient_norm_before_clip'] for row in history
            ),
            phase24_reference=dict(
                final_train=phase24_report['final_autonomous']['train'],
                final_holdout=phase24_report['final_autonomous']['holdout'],
                checkpoint_50_train_trajectory_final_apo_rmsd=0.541701,
                checkpoint_50_holdout_final_holo_rmsd=1.180337,
            ),
            phase26_checkpoint_50_train_trajectory_final=trajectory_50_summary['final'],
            initial=initial,
            final=final,
            evaluations=evaluations,
            apo_baseline=dict(train=apo_train, holdout=apo_holdout),
            final_autonomous=dict(train=final_train, holdout=final_holdout),
            limitations=[
                'Only max_grad_norm changed from Phase24; this is a fixed one-factor comparison',
                '8/8 development samples; holdout is not a strict test set',
                'clean reference ligand only; no noised ligand, TargetDiff or chi head',
                'engineering passed does not imply the apo-to-holo learning goal was met',
            ],
        )
        _write_json(args.output_dir / 'report.json', report)
        print(json.dumps(dict(
            passed=report['passed'],
            learning_goal_met=learning_goal_met,
            holdout_final=final_holdout,
            holdout_apo=apo_holdout,
            checkpoint_50_apo_rmsd=trajectory_50_summary['final']['apo_rmsd'],
        )), flush=True)
        if not report['passed']:
            raise RuntimeError('Phase26 engineering guards failed')
    except Exception as exc:
        _write_json(args.output_dir / 'failure.json', dict(
            error_type=type(exc).__name__,
            error=str(exc),
            elapsed_seconds=time.monotonic() - started,
        ))
        raise


if __name__ == '__main__':
    main()
