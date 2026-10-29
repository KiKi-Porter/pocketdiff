"""Fixed-budget online self-state training for the apo-to-holo goal."""

import argparse
import json
from pathlib import Path
import time

import torch

from pocketdiff.evaluation.clean_rollout import score_clean_rollout
from pocketdiff.sampling.clean_rollout import INFERENCE_FIELDS, run_clean_rollout
from pocketdiff.scripts.phase17_multik_train import _sha, _write_json
from pocketdiff.scripts.phase18_late_step_diagnosis import load_baseline_data
from pocketdiff.training import (
    OnlineSelfStateTrainer,
    build_latest_self_state,
    evaluate_self_state,
    refresh_online_trajectory,
)
from pocketdiff.training.multik import endpoint_fingerprint
from pocketdiff.training.self_state_trainer import trajectory_fingerprint


FIXED_STEPS = 200
FIXED_EVAL_EVERY = 50
FIXED_REFRESH_INTERVAL = 10
FIXED_SEED = 17
FIXED_LEARNING_RATE = 1e-3
FIXED_MAX_GRAD_NORM = 10.0


def _inference_inputs(clean):
    return {name: getattr(clean, name) for name in INFERENCE_FIELDS}


@torch.no_grad()
def _autonomous(model, batches):
    result = {}
    for role, clean in batches.items():
        inputs = _inference_inputs(clean)
        trace = run_clean_rollout(model, inputs)
        result[role] = score_clean_rollout(
            trace, inputs, clean.protein_pos_holo, clean.sample_ids,
        )
    return result


def _autonomous_summary(metrics):
    summary = {}
    for role, value in metrics.items():
        initial = value['per_step'][0]
        final = value['per_step'][-1]
        final_rows = [row for row in value['per_graph'] if row['step'] == 20]
        best = min(value['per_step'], key=lambda row: row['holo_rmsd'])
        summary[role] = dict(
            apo_holo_rmsd=initial['holo_rmsd'],
            final_holo_rmsd=final['holo_rmsd'],
            final_backbone_holo_rmsd=final['backbone_holo_rmsd'],
            improvement=initial['holo_rmsd'] - final['holo_rmsd'],
            improved_samples=sum(row['holo_rmsd'] < initial['holo_rmsd']
                                 for row in final_rows),
            regressed_samples=sum(row['holo_rmsd'] > initial['holo_rmsd']
                                  for row in final_rows),
            best_observed_step=best['step'],
            best_observed_holo_rmsd=best['holo_rmsd'],
            final_is_fixed_step_20=True,
            max_apo_displacement=final['max_apo_displacement'],
            per_step=[dict(step=row['step'], holo_rmsd=row['holo_rmsd'],
                           backbone_holo_rmsd=row['backbone_holo_rmsd'],
                           apo_displacement_rmsd=row['apo_displacement_rmsd'])
                      for row in value['per_step']],
        )
    return summary


def _evaluate(trainer, batches, output_dir):
    """Evaluate latest detached states and the no-holo autonomous endpoint."""
    train_trajectory = trainer.trajectory
    holdout_trajectory = refresh_online_trajectory(trainer.model, batches['holdout'])
    self_state = dict(
        step=trainer.step_count,
        train=evaluate_self_state(trainer.model, batches['train'],
                                   train_trajectory.positions),
        holdout=evaluate_self_state(trainer.model, batches['holdout'],
                                    holdout_trajectory.positions),
        train_trajectory_fingerprint=trajectory_fingerprint(train_trajectory.positions),
        holdout_trajectory_fingerprint=trajectory_fingerprint(holdout_trajectory.positions),
    )
    autonomous = _autonomous(trainer.model, batches)
    value = dict(
        step=trainer.step_count,
        self_state=self_state,
        autonomous=_autonomous_summary(autonomous),
    )
    _write_json(output_dir / ('evaluation_%04d.json' % trainer.step_count), value)
    return value


def _reload_error(checkpoint, clean, reference, trajectory):
    restored = OnlineSelfStateTrainer.from_checkpoint(checkpoint, clean)
    restored.model.eval()
    reference.eval()
    error = 0.0
    with torch.no_grad():
        for k in range(20):
            pocket_k = torch.full((len(clean.sample_ids),), k, dtype=torch.long)
            batch = build_latest_self_state(clean, trajectory, pocket_k)
            expected = reference(**batch.model_kwargs())
            actual = restored.model(**batch.model_kwargs())
            for name in ('remaining_translation_local', 'remaining_rotvec_local'):
                error = max(error, float(
                    (getattr(expected, name) - getattr(actual, name)).abs().max(),
                ))
    return error


def _check_fixed_args(args):
    fixed = (
        ('steps', args.steps, FIXED_STEPS),
        ('eval-every', args.eval_every, FIXED_EVAL_EVERY),
        ('refresh-interval', args.refresh_interval, FIXED_REFRESH_INTERVAL),
        ('seed', args.seed, FIXED_SEED),
        ('learning-rate', args.learning_rate, FIXED_LEARNING_RATE),
        ('max-grad-norm', args.max_grad_norm, FIXED_MAX_GRAD_NORM),
    )
    for name, value, expected in fixed:
        if value != expected:
            raise ValueError('%s is fixed at %s for Phase 24, got %s'
                             % (name, expected, value))


def main():
    parser = argparse.ArgumentParser()
    root = Path('.codex-tasks/pocketdiff-development')
    parser.add_argument('--baseline-dir', type=Path,
                        default=root / 'phase17-multik-clean-training/raw/run')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=FIXED_STEPS)
    parser.add_argument('--eval-every', type=int, default=FIXED_EVAL_EVERY)
    parser.add_argument('--refresh-interval', type=int, default=FIXED_REFRESH_INTERVAL)
    parser.add_argument('--seed', type=int, default=FIXED_SEED)
    parser.add_argument('--learning-rate', type=float, default=FIXED_LEARNING_RATE)
    parser.add_argument('--max-grad-norm', type=float, default=FIXED_MAX_GRAD_NORM)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    _check_fixed_args(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
        parser.error('output directory must be empty; preserve previous experiments')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.monotonic()
    try:
        baseline, batches = load_baseline_data(args.baseline_dir)
        if any(len(batch.sample_ids) != 8 for batch in batches.values()):
            raise ValueError('Phase 24 requires the fixed eight train/eight holdout samples')
        endpoint_fingerprints = {
            role: endpoint_fingerprint(batch) for role, batch in batches.items()
        }
        for role, value in endpoint_fingerprints.items():
            expected = baseline['preparation'][role + '_endpoint_fingerprint']
            if value != expected:
                raise ValueError('endpoint fingerprint differs from Phase 17: ' + role)
        initial_checkpoint = args.baseline_dir / 'checkpoint_0400.pt'
        initial_payload = torch.load(initial_checkpoint, map_location='cpu')
        initial_sha = _sha(initial_checkpoint)
        if initial_sha != baseline['checkpoint_sha256']:
            raise ValueError('Phase 17 initial checkpoint SHA differs from report')

        provenance = dict(
            baseline_dir=str(args.baseline_dir.resolve()),
            initial_checkpoint=str(initial_checkpoint.resolve()),
            initial_checkpoint_sha256=initial_sha,
            baseline_report_sha256=_sha(args.baseline_dir / 'report.json'),
            baseline_manifest_sha256=_sha(args.baseline_dir / 'manifest.json'),
            endpoint_fingerprints=endpoint_fingerprints,
            model_config=dict(initial_payload['model_config'],
                              motion_parameterization='remaining'),
            fixed=dict(steps=FIXED_STEPS, eval_every=FIXED_EVAL_EVERY,
                       refresh_interval=FIXED_REFRESH_INTERVAL, seed=FIXED_SEED,
                       learning_rate=FIXED_LEARNING_RATE,
                       max_grad_norm=FIXED_MAX_GRAD_NORM),
            condition='clean reference ligand; physical remaining self-state',
            targetdiff_bridge=False,
        )
        _write_json(args.output_dir / 'provenance.json', provenance)
        _write_json(args.output_dir / 'code_fingerprints.json', {
            str(path): _sha(path)
            for path in sorted(Path('pocketdiff').rglob('*.py'))
        })

        history_path = args.output_dir / 'history.jsonl'
        if args.resume:
            payload = torch.load(args.resume, map_location='cpu')
            if payload.get('metadata', {}).get('provenance') != provenance:
                raise ValueError('resume provenance differs from current Phase 24 inputs')
            trainer = OnlineSelfStateTrainer.from_checkpoint(
                args.resume, batches['train'],
            )
            if trainer.step_count > args.steps:
                raise ValueError('resume checkpoint exceeds fixed Phase 24 steps')
            if not history_path.exists():
                raise ValueError('resume requires history.jsonl')
            history = [json.loads(line) for line in history_path.read_text().splitlines()]
            history = [record for record in history if record['step'] <= trainer.step_count]
            history_path.write_text(''.join(json.dumps(row) + '\n' for row in history))
            evaluations = [
                json.loads(path.read_text())
                for path in sorted(args.output_dir.glob('evaluation_*.json'))
                if int(path.stem.split('_')[-1]) <= trainer.step_count
            ]
            if not evaluations or evaluations[0]['step'] != 0:
                raise ValueError('resume requires evaluation_0000.json')
        else:
            if history_path.exists() or list(args.output_dir.glob('checkpoint_*.pt')):
                raise FileExistsError('existing Phase 24 run; use --resume or a new output directory')
            trainer = OnlineSelfStateTrainer.from_initial_checkpoint(
                initial_checkpoint, batches['train'],
                refresh_interval=FIXED_REFRESH_INTERVAL,
                seed=FIXED_SEED, learning_rate=FIXED_LEARNING_RATE,
                max_grad_norm=FIXED_MAX_GRAD_NORM,
            )
            initial_weights_identical = all(
                torch.equal(value, initial_payload['model_state_dict'][name])
                for name, value in trainer.model.state_dict().items()
            )
            if not initial_weights_identical:
                raise ValueError('online trainer weights differ from Phase 17 checkpoint_0400')
            metadata = dict(provenance=provenance,
                            initial_trajectory_fingerprint=trainer.trajectory_fingerprint,
                            torch_version=torch.__version__,
                            cpu_threads=torch.get_num_threads())
            trainer.save_checkpoint(args.output_dir / 'checkpoint_0000.pt',
                                    metadata=metadata)
            evaluations = [_evaluate(trainer, batches, args.output_dir)]
            history = []

        metadata = dict(provenance=provenance,
                        initial_trajectory_fingerprint=evaluations[0]['self_state']
                        ['train_trajectory_fingerprint'],
                        torch_version=torch.__version__,
                        cpu_threads=torch.get_num_threads())
        with history_path.open('a') as handle:
            while trainer.step_count < args.steps:
                record = trainer.step()
                history.append(record)
                handle.write(json.dumps(record, allow_nan=False) + '\n')
                handle.flush()
                if (trainer.step_count % args.eval_every == 0
                        or trainer.step_count == args.steps):
                    checkpoint = args.output_dir / ('checkpoint_%04d.pt'
                                                    % trainer.step_count)
                    trainer.save_checkpoint(checkpoint, metadata=metadata)
                    evaluation = _evaluate(trainer, batches, args.output_dir)
                    evaluations.append(evaluation)
                    print(json.dumps(dict(
                        step=trainer.step_count,
                        train_loss=evaluation['self_state']['train']['mean']['loss'],
                        holdout_loss=evaluation['self_state']['holdout']['mean']['loss'],
                        train_final=evaluation['autonomous']['train']['final_holo_rmsd'],
                        holdout_final=evaluation['autonomous']['holdout']['final_holo_rmsd'],
                    )), flush=True)

        final_checkpoint = args.output_dir / ('checkpoint_%04d.pt'
                                              % trainer.step_count)
        reload_error = _reload_error(
            final_checkpoint, batches['train'], trainer.model, trainer.trajectory,
        )
        final = evaluations[-1]
        initial = evaluations[0]
        expected_refreshes = trainer.step_count // FIXED_REFRESH_INTERVAL
        refresh_records = [
            row for row in history if row.get('refreshed') and
            row['step'] % FIXED_REFRESH_INTERVAL == 0
        ]
        guards = dict(
            initial_weights_identical=True,
            fixed_step_count=trainer.step_count == FIXED_STEPS,
            fixed_refresh_schedule=len(refresh_records) == expected_refreshes,
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
        final = evaluations[-1]
        apo_train = initial['autonomous']['train']['apo_holo_rmsd']
        apo_holdout = initial['autonomous']['holdout']['apo_holo_rmsd']
        final_train = final['autonomous']['train']['final_holo_rmsd']
        final_holdout = final['autonomous']['holdout']['final_holo_rmsd']
        learning_goal_met = bool(final_train < apo_train and final_holdout < apo_holdout)
        engineering_passed = all(
            value for value in guards.values()
            if not isinstance(value, list)
        )
        report = dict(
            passed=engineering_passed,
            engineering_passed=engineering_passed,
            learning_goal_met=learning_goal_met,
            mode='online_self_state_remaining_training',
            steps=trainer.step_count,
            refresh_interval=FIXED_REFRESH_INTERVAL,
            elapsed_seconds=time.monotonic() - started,
            guards=guards,
            provenance=provenance,
            reload_max_error=reload_error,
            k_histogram=trainer.k_histogram.tolist(),
            initial=initial,
            final=final,
            evaluations=evaluations,
            apo_baseline=dict(train=apo_train, holdout=apo_holdout),
            final_autonomous=dict(train=final_train, holdout=final_holdout),
            limitations=[
                '8/8 development samples; holdout is not a strict test set',
                'clean reference ligand only; no noised ligand, TargetDiff or chi head',
                'holo is used only for labels and read-only scoring',
                'best intermediate step is diagnostic only; final metric is fixed step 20',
                'engineering passed does not imply the apo-to-holo learning goal was met',
            ],
        )
        _write_json(args.output_dir / 'report.json', report)
        print(json.dumps(dict(
            passed=report['passed'],
            learning_goal_met=learning_goal_met,
            holdout_final=final_holdout,
            holdout_apo=apo_holdout,
            guards=guards,
        )), flush=True)
        if not report['passed']:
            raise RuntimeError('online self-state engineering checks failed; inspect report.json')
    except Exception as exc:
        _write_json(args.output_dir / 'failure.json', dict(
            error_type=type(exc).__name__,
            error=str(exc),
            elapsed_seconds=time.monotonic() - started,
        ))
        raise


if __name__ == '__main__':
    main()
