"""One preregistered 400-step bridge-rate comparison on Phase17's fixed data."""

import argparse
import json
from pathlib import Path
import time

import torch

from pocketdiff.scripts.phase17_multik_train import _evaluate, _reload_error, _sha, _write_json
from pocketdiff.scripts.phase18_late_step_diagnosis import load_baseline_data
from pocketdiff.training import MultiKCleanTrainer
from pocketdiff.training.multik import endpoint_fingerprint


def comparison(baseline, final):
    result = {}
    for role in ('train', 'holdout'):
        candidates = dict(zero_update=baseline['initial'][role],
                          phase17=baseline['final'][role], bridge_rate=final[role])
        result[role] = {name: dict(mean_remaining_loss=metrics['mean']['loss'],
                                   mean_endpoint_holo_rmsd=metrics['mean']['endpoint_holo_rmsd'],
                                   k0_endpoint_holo_rmsd=metrics['per_k'][0]['endpoint_holo_rmsd'],
                                   k18_next_bridge_rmsd=metrics['per_k'][18]['next_bridge_rmsd'],
                                   k19_next_bridge_rmsd=metrics['per_k'][19]['next_bridge_rmsd'],
                                   k19_remaining_loss=metrics['per_k'][19]['loss'])
                        for name, metrics in candidates.items()}
        # Preserve each sample, including regressions, rather than only means.
        rows = []
        for row in final[role]['per_graph']:
            if row['k'] not in (0, 18, 19):
                continue
            entry = dict(sample_id=row['sample_id'], k=row['k'])
            for name, metrics in candidates.items():
                selected = next(v for v in metrics['per_graph']
                                if v['sample_id'] == row['sample_id'] and v['k'] == row['k'])
                entry[name] = {key: selected[key] for key in
                               ('loss', 'next_bridge_rmsd', 'endpoint_holo_rmsd')}
            rows.append(entry)
        result[role]['per_sample'] = rows
    return result


def plot_comparison(output_dir, baseline, evaluations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for column, role in enumerate(('train', 'holdout')):
        ax = axes[0, column]
        ax.plot([v['step'] for v in evaluations], [v[role]['mean']['loss'] for v in evaluations],
                marker='o', label='Bridge rate')
        ax.axhline(baseline['final'][role]['mean']['loss'], linestyle='--', label='Phase17 final')
        ax.set(title=role + ': teacher-forced remaining loss', xlabel='Training steps', ylabel='Remaining loss')
        ax = axes[1, column]
        for label, metrics in (('Zero update', baseline['initial'][role]),
                               ('Phase17 final', baseline['final'][role]),
                               ('Bridge rate', evaluations[-1][role])):
            ax.plot(range(20), [v['next_bridge_rmsd'] for v in metrics['per_k']], label=label)
        ax.set(title=role + ': one-step bridge error', xlabel='Pocket step k', ylabel='RMSD (angstrom)')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir/'comparison.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-dir', type=Path, default=Path(
        '.codex-tasks/pocketdiff-development/phase17-multik-clean-training/raw/run'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output directory must be empty; preserve existing experiments')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.monotonic()
    try:
        baseline, batches = load_baseline_data(args.baseline_dir)
        initial_payload = torch.load(args.baseline_dir/'checkpoint_0000.pt', map_location='cpu')
        history = [json.loads(line) for line in (args.baseline_dir/'history.jsonl').read_text().splitlines()]
        if baseline['steps'] != 400 or len(history) != 400:
            raise ValueError('expected fixed 400-step baseline')
        model_config = dict(initial_payload['model_config'], motion_parameterization='bridge_rate')
        trainer = MultiKCleanTrainer(batches['train'], model_config=model_config,
                                     **initial_payload['trainer_config'])
        identical_weights = all(torch.equal(value, initial_payload['model_state_dict'][name])
                                for name, value in trainer.model.state_dict().items())
        if not identical_weights or not torch.equal(torch.get_rng_state(), initial_payload['torch_rng_state']):
            raise ValueError('initial weights or dropout RNG differ from Phase17')
        data_fingerprints = {role: endpoint_fingerprint(batch) for role, batch in batches.items()}
        for role, fingerprint in data_fingerprints.items():
            if fingerprint != baseline['preparation'][role + '_endpoint_fingerprint']:
                raise ValueError('endpoint fingerprint differs from Phase17: ' + role)
        provenance = dict(baseline_dir=str(args.baseline_dir.resolve()),
                          baseline_report_sha256=_sha(args.baseline_dir/'report.json'),
                          baseline_checkpoint_sha256=_sha(args.baseline_dir/'checkpoint_0400.pt'),
                          manifest_sha256=_sha(args.baseline_dir/'manifest.json'),
                          endpoint_fingerprints=data_fingerprints, model_config=trainer.model_config,
                          trainer_config=trainer.config)
        _write_json(args.output_dir/'provenance.json', provenance)
        _write_json(args.output_dir/'code_fingerprints.json', {
            str(path): _sha(path) for path in sorted(Path('pocketdiff').rglob('*.py'))})
        trainer.save_checkpoint(args.output_dir/'checkpoint_0000.pt', metadata=provenance)
        evaluations = [_evaluate(trainer, batches, args.output_dir)]
        if evaluations[0] != baseline['initial']:
            raise ValueError('initial physical-unit evaluation differs from zero-update baseline')
        rng_checks = []
        with (args.output_dir/'history.jsonl').open('w') as handle:
            for expected in history:
                record = trainer.step()
                handle.write(json.dumps(record, allow_nan=False) + '\n')
                handle.flush()
                if record['step'] != expected['step'] or record['pocket_k'] != expected['pocket_k']:
                    raise ValueError('sampled k sequence differs from baseline')
                if trainer.step_count % 100 == 0:
                    checkpoint = args.output_dir/('checkpoint_%04d.pt' % trainer.step_count)
                    trainer.save_checkpoint(checkpoint, metadata=provenance)
                    old = torch.load(args.baseline_dir/checkpoint.name, map_location='cpu')
                    rng_equal = (torch.equal(torch.get_rng_state(), old['torch_rng_state']) and
                                 torch.equal(trainer.generator.get_state(), old['k_generator_state']))
                    rng_checks.append(dict(step=trainer.step_count, equal=rng_equal))
                    if not rng_equal:
                        raise ValueError('dropout/k RNG consumption differs from baseline')
                    evaluations.append(_evaluate(trainer, batches, args.output_dir))
                    print(json.dumps(dict(step=trainer.step_count,
                                          train=evaluations[-1]['train']['mean']['loss'],
                                          holdout=evaluations[-1]['holdout']['mean']['loss'])), flush=True)
        reload_error = _reload_error(checkpoint, batches, trainer.model)
        guards = dict(initial_weights_identical=identical_weights, sampled_k_identical=True,
                      rng_states_identical=all(v['equal'] for v in rng_checks),
                      reload_exact=reload_error == 0,
                      inputs_unchanged=all(endpoint_fingerprint(batches[r]) == f
                                           for r, f in data_fingerprints.items()),
                      k_coverage=bool((trainer.k_histogram > 0).all()))
        report = dict(passed=all(guards.values()), guards=guards, steps=400,
                      elapsed_seconds=time.monotonic()-started, provenance=provenance,
                      reload_max_error=reload_error, rng_checks=rng_checks,
                      checkpoint=str(checkpoint), checkpoint_sha256=_sha(checkpoint),
                      k_histogram=trainer.k_histogram.tolist(), initial=evaluations[0], final=evaluations[-1],
                      comparison=comparison(baseline, evaluations[-1]),
                      limitations=['8/8 development examples; not a strict test',
                                   'Teacher-forced clean only; no autonomous rollout',
                                   'Optimization loss is rate-normalized; evaluation uses actual remaining units',
                                   'Rotation bound is pi*r; autonomous correction ability is unverified',
                                   'passed means engineering/replay checks, not biological or generalization success'])
        _write_json(args.output_dir/'report.json', report)
        plot_comparison(args.output_dir, baseline, evaluations)
        print(json.dumps(dict(passed=report['passed'], guards=guards)), flush=True)
        if not report['passed']:
            raise RuntimeError('experiment engineering checks failed')
    except Exception as exc:
        _write_json(args.output_dir/'failure.json', dict(error_type=type(exc).__name__, error=str(exc),
                                                        elapsed_seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
