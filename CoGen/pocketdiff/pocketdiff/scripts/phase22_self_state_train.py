"""One fixed-budget self-state training comparison for the apo-to-holo goal."""

import argparse
import json
from pathlib import Path
import time

import torch

from pocketdiff.evaluation.clean_rollout import score_clean_rollout
from pocketdiff.sampling.clean_rollout import INFERENCE_FIELDS, run_clean_rollout
from pocketdiff.scripts.phase17_multik_train import _sha, _write_json
from pocketdiff.scripts.phase18_late_step_diagnosis import load_baseline_data
from pocketdiff.scripts.phase19_clean_rollout import load_model
from pocketdiff.training import (SelfStateTrainer, build_self_state_batch,
                                 evaluate_self_state)
from pocketdiff.training.multik import endpoint_fingerprint
from pocketdiff.training.self_state_trainer import CHECKPOINT_FORMAT, trajectory_fingerprint


def _load_positions(path, sample_ids):
    payload = torch.load(path, map_location='cpu')
    if payload.get('format') != 'pocketdiff-clean-rollout-v1' or payload.get('sample_ids') != sample_ids:
        raise ValueError('incompatible autonomous trajectory: ' + str(path))
    positions = payload['positions']
    if positions.shape != (21, payload['positions'].shape[1], 3) or not torch.isfinite(positions).all():
        raise ValueError('trajectory must contain finite 21-state positions')
    return positions


def _evaluate_self(trainer, batches, trajectories, output_dir):
    result = dict(step=trainer.step_count)
    for role in ('train', 'holdout'):
        result[role] = evaluate_self_state(trainer.model, batches[role], trajectories[role])
    _write_json(output_dir/('self_state_evaluation_%04d.json' % trainer.step_count), result)
    return result


def _reload_error(checkpoint, batches, trajectories, reference):
    restored = SelfStateTrainer.from_checkpoint(checkpoint, batches['train'], trajectories['train'])
    restored.model.eval()
    reference.eval()
    error = 0.0
    with torch.no_grad():
        for role in ('train', 'holdout'):
            clean, trajectory = batches[role], trajectories[role]
            for k in range(20):
                batch = build_self_state_batch(clean, trajectory,
                                               torch.full((len(clean.sample_ids),), k, dtype=torch.long))
                expected = reference(**batch.model_kwargs())
                actual = restored.model(**batch.model_kwargs())
                for name in ('remaining_translation_local', 'remaining_rotvec_local'):
                    error = max(error, float((getattr(expected, name)-getattr(actual, name)).abs().max()))
    return error


def _autonomous(model, batches):
    result = {}
    for role, clean in batches.items():
        inputs = {name: getattr(clean, name) for name in INFERENCE_FIELDS}
        trace = run_clean_rollout(model, inputs)
        result[role] = score_clean_rollout(trace, inputs, clean.protein_pos_holo, clean.sample_ids)
    return result


def _autonomous_summary(metrics):
    result = {}
    for role, value in metrics.items():
        final = value['per_step'][-1]
        initial = value['per_step'][0]
        final_rows = [row for row in value['per_graph'] if row['step'] == 20]
        result[role] = dict(
            apo_holo_rmsd=initial['holo_rmsd'], final_holo_rmsd=final['holo_rmsd'],
            improvement=initial['holo_rmsd']-final['holo_rmsd'],
            final_backbone_holo_rmsd=final['backbone_holo_rmsd'],
            improved_samples=sum(row['holo_rmsd'] < initial['holo_rmsd'] for row in final_rows),
            regressed_samples=sum(row['holo_rmsd'] > initial['holo_rmsd'] for row in final_rows),
            max_apo_displacement=final['max_apo_displacement'],
            per_step=[dict(step=row['step'], holo_rmsd=row['holo_rmsd'],
                           ideal_bridge_distance_rmsd=row.get('ideal_bridge_distance_rmsd'))
                      for row in value['per_step']],
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    root = Path('.codex-tasks/pocketdiff-development')
    parser.add_argument('--baseline-dir', type=Path, default=root/'phase17-multik-clean-training/raw/run')
    parser.add_argument('--rate-dir', type=Path, default=root/'phase18-late-step-diagnosis/raw/run')
    parser.add_argument('--rollout-dir', type=Path, default=root/'phase19-clean-rollout/raw/run')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--eval-every', type=int, default=100)
    args = parser.parse_args()
    if args.steps <= 0 or args.eval_every <= 0:
        parser.error('steps and eval-every must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output directory must be empty; preserve previous experiments')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.monotonic()
    try:
        baseline, batches = load_baseline_data(args.baseline_dir)
        rate_report = json.loads((args.rate_dir/'report.json').read_text())
        prior_rollout = json.loads((args.rollout_dir/'report.json').read_text())
        trajectories = {
            role: _load_positions(args.rollout_dir/(role+'_phase18_trajectory.pt'), batches[role].sample_ids)
            for role in ('train', 'holdout')
        }
        endpoint_fingerprints = {role: endpoint_fingerprint(batch) for role, batch in batches.items()}
        for role, value in endpoint_fingerprints.items():
            if value != baseline['preparation'][role+'_endpoint_fingerprint']:
                raise ValueError('endpoint fingerprint differs from baseline: ' + role)
        trajectory_fingerprints = {role: trajectory_fingerprint(value) for role, value in trajectories.items()}
        initial_payload = torch.load(args.baseline_dir/'checkpoint_0000.pt', map_location='cpu')
        phase17_payload = torch.load(args.baseline_dir/'checkpoint_0400.pt', map_location='cpu')
        model_config = dict(initial_payload['model_config'], motion_parameterization='remaining')
        trainer = SelfStateTrainer(batches['train'], trajectories['train'], model_config=model_config,
                                   **initial_payload['trainer_config'])
        initial_weights_identical = all(torch.equal(value, initial_payload['model_state_dict'][name])
                                        for name, value in trainer.model.state_dict().items())
        if not initial_weights_identical:
            raise ValueError('self-state initial weights differ from Phase17 checkpoint_0000')
        provenance = dict(
            baseline_dir=str(args.baseline_dir.resolve()), rate_dir=str(args.rate_dir.resolve()),
            rollout_dir=str(args.rollout_dir.resolve()), baseline_report_sha256=_sha(args.baseline_dir/'report.json'),
            phase17_checkpoint_sha256=_sha(args.baseline_dir/'checkpoint_0400.pt'),
            phase18_checkpoint_sha256=_sha(args.rate_dir/'checkpoint_0400.pt'),
            rollout_report_sha256=_sha(args.rollout_dir/'report.json'),
            endpoint_fingerprints=endpoint_fingerprints, trajectory_fingerprints=trajectory_fingerprints,
            model_config=trainer.model_config, trainer_config=trainer.config,
        )
        _write_json(args.output_dir/'provenance.json', provenance)
        _write_json(args.output_dir/'code_fingerprints.json', {
            str(path): _sha(path) for path in sorted(Path('pocketdiff').rglob('*.py'))})
        trainer.save_checkpoint(args.output_dir/'checkpoint_0000.pt', metadata=provenance)
        evaluations = [_evaluate_self(trainer, batches, trajectories, args.output_dir)]
        history_path = args.output_dir/'history.jsonl'
        expected_history = [json.loads(line) for line in (args.baseline_dir/'history.jsonl').read_text().splitlines()]
        if args.steps != len(expected_history):
            raise ValueError('steps must equal fixed Phase17 history length for controlled comparison')
        rng_checks = []
        with history_path.open('w') as handle:
            for expected in expected_history:
                record = trainer.step()
                handle.write(json.dumps(record, allow_nan=False)+'\n')
                handle.flush()
                if record['step'] != expected['step'] or record['pocket_k'] != expected['pocket_k']:
                    raise ValueError('self-state k sequence differs from Phase17')
                if trainer.step_count % args.eval_every == 0:
                    checkpoint = args.output_dir/('checkpoint_%04d.pt' % trainer.step_count)
                    trainer.save_checkpoint(checkpoint, metadata=provenance)
                    baseline_checkpoint = torch.load(args.baseline_dir/checkpoint.name, map_location='cpu')
                    rng_equal = (torch.equal(torch.get_rng_state(), baseline_checkpoint['torch_rng_state']) and
                                 torch.equal(trainer.generator.get_state(), baseline_checkpoint['k_generator_state']))
                    rng_checks.append(dict(step=trainer.step_count, equal=rng_equal))
                    if not rng_equal:
                        raise ValueError('dropout/k RNG state differs from Phase17 at step %d' % trainer.step_count)
                    evaluations.append(_evaluate_self(trainer, batches, trajectories, args.output_dir))
                    print(json.dumps(dict(step=trainer.step_count,
                                          train=evaluations[-1]['train']['mean']['loss'],
                                          holdout=evaluations[-1]['holdout']['mean']['loss'])), flush=True)
        final_checkpoint = args.output_dir/('checkpoint_%04d.pt' % trainer.step_count)
        reload_error = _reload_error(final_checkpoint, batches, trajectories, trainer.model)
        autonomous = _autonomous(trainer.model, batches)
        self_state_final = evaluations[-1]
        teacher_forced = {}
        from pocketdiff.training import evaluate_multik_clean
        for role, batch in batches.items():
            teacher_forced[role] = evaluate_multik_clean(trainer.model, batch)
        guards = dict(
            initial_weights_identical=initial_weights_identical,
            sampled_k_identical=True,
            rng_states_identical=all(item['equal'] for item in rng_checks),
            reload_exact=reload_error == 0,
            k_coverage=bool((trainer.k_histogram > 0).all()),
            inputs_unchanged=all(endpoint_fingerprint(batches[role]) == value
                                 for role, value in endpoint_fingerprints.items()),
            trajectories_unchanged=all(trajectory_fingerprint(trajectories[role]) == value
                                       for role, value in trajectory_fingerprints.items()),
            finite_parameters=all(torch.isfinite(parameter).all() for parameter in trainer.model.parameters()),
        )
        report = dict(
            passed=all(guards.values()), mode='self_state_remaining_training', steps=trainer.step_count,
            elapsed_seconds=time.monotonic()-started, guards=guards, provenance=provenance,
            rng_checks=rng_checks, reload_max_error=reload_error,
            k_histogram=trainer.k_histogram.tolist(), initial=evaluations[0], final=self_state_final,
            teacher_forced_clean=teacher_forced,
            autonomous=_autonomous_summary(autonomous),
            prior_autonomous={role: prior_rollout['comparison'][role]['summary'] for role in ('train', 'holdout')},
            limitations=['8/8 development samples; holdout is not a strict test set',
                         'fixed Phase18 autonomous states; no online state refresh during training',
                         'clean reference ligand only; no noised ligand, TargetDiff or chi head',
                         'passed means engineering/replay checks, not apo-to-holo success'],
        )
        _write_json(args.output_dir/'report.json', report)
        print(json.dumps(dict(passed=report['passed'], guards=guards,
                              autonomous_holdout_final=report['autonomous']['holdout']['final_holo_rmsd'])), flush=True)
        if not report['passed']:
            raise RuntimeError('self-state training engineering checks failed')
    except Exception as exc:
        _write_json(args.output_dir/'failure.json', dict(error_type=type(exc).__name__, error=str(exc),
                                                        elapsed_seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
