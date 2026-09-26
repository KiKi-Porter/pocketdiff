"""Fixed-subset clean multi-k training; preserves historical data roles/results."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.preprocessing import (
    GEOMETRY_VERSION, fingerprint_file, load_cached_clean_examples, load_manifest,
    save_sample_cache, write_manifest,
)
from pocketdiff.training import (
    MultiKCleanTrainer, build_bridge_batch, collate_clean_examples, evaluate_multik_clean,
)
from pocketdiff.training.multik import endpoint_fingerprint


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_data(args):
    """Read old identity/source metadata only, never load old geometry labels."""
    prior = json.loads(args.prior_report.read_text())
    roles = {role: prior['split'][role + '_ids'][:8] for role in ('train', 'holdout')}
    if any(len(ids) != 8 or len(set(ids)) != 8 for ids in roles.values()):
        raise ValueError('historical split must provide eight distinct samples per role')
    if set(roles['train']) & set(roles['holdout']):
        raise ValueError('train/holdout identity overlap')
    manifest_path = args.output_dir / 'manifest.json'
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        if manifest['config']['roles'] != roles:
            raise ValueError('existing cache roles differ from the fixed experiment')
    else:
        old_manifest = json.loads(args.prior_manifest.read_text())
        by_id = {entry['sample_id']: entry for entry in old_manifest['entries']}
        adapter = Apo2MolAdapter('.')
        entries = []
        for role, ids in roles.items():
            for sample_id in ids:
                source = by_id[sample_id]['source']
                paths = {name: Path(info['path']).resolve() for name, info in source.items()}
                for name, path in paths.items():
                    if fingerprint_file(path)['sha256'] != source[name]['sha256']:
                        raise ValueError('source differs from historical fingerprint: ' + str(path))
                value, _ = adapter.convert_paths(paths['holo_pocket'], paths['apo_pocket'], paths['ligand'],
                                                 sample_id=sample_id)
                cache_path = args.output_dir / 'cache_samples' / (sample_id + '.pt')
                entry = save_sample_cache(cache_path, value, source_paths=paths, split=role)
                entry['cache_path'] = str(cache_path.relative_to(args.output_dir))
                entries.append(entry)
        write_manifest(manifest_path, entries, config=dict(
            roles=roles, selection='first 8 IDs per role from Phase 6b',
            prior_report_sha256=_sha(args.prior_report), prior_manifest_sha256=_sha(args.prior_manifest),
            original_dataset_split='Apo2Mol train; development holdout, not strict test',
            cache_edge_policy='dynamic edges not cached',
        ))
    examples = load_cached_clean_examples(manifest_path, verify_sources=True)
    by_id = {example.complex_value.sample_id: example for example in examples}
    batches = {role: collate_clean_examples([by_id[sample_id] for sample_id in ids]) for role, ids in roles.items()}
    preparation = dict(geometry_version=GEOMETRY_VERSION, roles=roles, overlap=[],
                       manifest_sha256=_sha(manifest_path), source_fingerprints_verified=True,
                       train_endpoint_fingerprint=endpoint_fingerprint(batches['train']),
                       holdout_endpoint_fingerprint=endpoint_fingerprint(batches['holdout']))
    _write_json(args.output_dir / 'preparation.json', preparation)
    return batches, preparation


def _evaluate(trainer, batches, output_dir):
    metrics = dict(step=trainer.step_count, **{
        role: evaluate_multik_clean(trainer.model, batch) for role, batch in batches.items()
    })
    _write_json(output_dir / ('evaluation_%04d.json' % trainer.step_count), metrics)
    return metrics


def _reload_error(checkpoint, batches, reference):
    restored = MultiKCleanTrainer.from_checkpoint(checkpoint, batches['train'])
    # Compare the actual trained instance with the restored one at every time.
    reference.eval()
    restored.model.eval()
    error = 0.0
    with torch.no_grad():
        for clean in batches.values():
            for k in range(20):
                batch = build_bridge_batch(clean, torch.full((len(clean.sample_ids),), k, dtype=torch.long))
                expected = reference(**batch.model_kwargs())
                actual = restored.model(**batch.model_kwargs())
                for name in ('remaining_translation_local', 'remaining_rotvec_local'):
                    error = max(error, float((getattr(expected, name) - getattr(actual, name)).abs().max()))
    return error


def _plot_curve(output_dir, evaluations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for role in ('train', 'holdout'):
        axes[0].plot([r['step'] for r in evaluations], [r[role]['mean']['loss'] for r in evaluations],
                     marker='o', label=role)
        axes[1].plot(range(20), [r['loss'] for r in evaluations[-1][role]['per_k']],
                     marker='.', label=role + ' final')
        axes[1].plot(range(20), [r['loss'] for r in evaluations[0][role]['per_k']],
                     linestyle='--', label=role + ' initial')
    axes[0].set(xlabel='Optimizer steps', ylabel='Mean remaining-motion loss',
                title='Fixed 20-time teacher-forced evaluation')
    axes[1].set(xlabel='Pocket step k', ylabel='Remaining-motion loss', title='Each time, graph-equal mean')
    for ax in axes:
        ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output_dir / 'training_curve.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    prior_root = Path('.codex-tasks/pocketdiff-development/phase6b-clean-generalization/raw')
    parser.add_argument('--prior-report', type=Path, default=prior_root / 'generalization_report.json')
    parser.add_argument('--prior-manifest', type=Path, default=prior_root / 'manifest.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.steps <= 0 or args.eval_every <= 0:
        parser.error('steps and eval-every must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.monotonic()
    try:
        batches, preparation = prepare_data(args)
        if args.prepare_only:
            print(json.dumps(dict(prepared=True, samples=16, geometry_version=GEOMETRY_VERSION)), flush=True)
            return
        metadata = dict(preparation=preparation, condition='clean', training_state='teacher_forced_bridge',
                        torch_version=torch.__version__, cpu_threads=torch.get_num_threads())
        history_path = args.output_dir / 'history.jsonl'
        if args.resume:
            payload = torch.load(args.resume, map_location='cpu')
            if payload['metadata']['preparation'] != preparation:
                raise ValueError('resume cache/holdout provenance differs')
            trainer = MultiKCleanTrainer.from_checkpoint(args.resume, batches['train'])
            if trainer.step_count > args.steps:
                raise ValueError('resume step exceeds requested total steps')
            history = [json.loads(line) for line in history_path.read_text().splitlines()]
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
            history_path.with_name('history_before_resume_' + stamp + '.jsonl').write_bytes(history_path.read_bytes())
            history = [record for record in history if record['step'] <= trainer.step_count]
            history_path.write_text(''.join(json.dumps(row) + '\n' for row in history))
            evaluations = [json.loads(p.read_text()) for p in sorted(args.output_dir.glob('evaluation_*.json'))]
            evaluations = [r for r in evaluations if r['step'] <= trainer.step_count]
            if not evaluations or evaluations[0]['step'] != 0:
                raise ValueError('resume requires saved initial evaluation')
        else:
            if history_path.exists() or list(args.output_dir.glob('checkpoint_*.pt')):
                raise FileExistsError('existing training run; use --resume or a new output directory')
            trainer = MultiKCleanTrainer(batches['train'], seed=args.seed, learning_rate=args.learning_rate)
            history = []
            evaluations = [_evaluate(trainer, batches, args.output_dir)]
            trainer.save_checkpoint(args.output_dir / 'checkpoint_0000.pt', metadata=metadata)
        print(json.dumps(dict(stage='training', step=trainer.step_count,
                              initial_train_loss=evaluations[0]['train']['mean']['loss'],
                              initial_holdout_loss=evaluations[0]['holdout']['mean']['loss'])), flush=True)
        with history_path.open('a') as log:
            while trainer.step_count < args.steps:
                record = trainer.step()
                history.append(record)
                log.write(json.dumps(record, allow_nan=False) + '\n'); log.flush()
                if trainer.step_count % args.eval_every == 0 or trainer.step_count == args.steps:
                    # Save before evaluation so a diagnostic interruption cannot lose completed optimization.
                    trainer.save_checkpoint(args.output_dir / ('checkpoint_%04d.pt' % trainer.step_count), metadata=metadata)
                    metrics = _evaluate(trainer, batches, args.output_dir)
                    evaluations.append(metrics)
                    print(json.dumps(dict(step=trainer.step_count, train=metrics['train']['mean'],
                                          holdout=metrics['holdout']['mean'])), flush=True)
        if evaluations[-1]['step'] != trainer.step_count:
            evaluations.append(_evaluate(trainer, batches, args.output_dir))
        initial, final = evaluations[0], evaluations[-1]
        checkpoint = args.output_dir / ('checkpoint_%04d.pt' % trainer.step_count)
        reload_error = _reload_error(checkpoint, batches, trainer.model)
        guards = dict(all_times_sampled=bool((trainer.k_histogram > 0).all()),
                      train_fixed_grid_loss_decreased=final['train']['mean']['loss'] < initial['train']['mean']['loss'],
                      checkpoint_reload_exact=reload_error == 0,
                      train_input_unchanged=endpoint_fingerprint(batches['train']) == preparation['train_endpoint_fingerprint'],
                      holdout_input_unchanged=endpoint_fingerprint(batches['holdout']) == preparation['holdout_endpoint_fingerprint'])
        report = dict(
            passed=all(guards.values()), guards=guards, steps=trainer.step_count,
            model_config=trainer.model_config, trainer_config=trainer.config,
            geometry_version=GEOMETRY_VERSION, preparation=preparation,
            checkpoint=str(checkpoint), checkpoint_sha256=_sha(checkpoint), reload_max_error=reload_error,
            k_histogram=trainer.k_histogram.tolist(), initial=initial, final=final,
            evaluation_steps=[r['step'] for r in evaluations], elapsed_seconds=time.monotonic()-started,
            holdout_loss_improved=final['holdout']['mean']['loss'] < initial['holdout']['mean']['loss'],
            limitations='small development subset; teacher-forced clean only; no self-rollout, no noised training; no strict generalization claim',
        )
        _write_json(args.output_dir / 'report.json', report)
        _plot_curve(args.output_dir, evaluations)
        print(json.dumps(dict(passed=report['passed'], guards=guards,
                              holdout_loss_improved=report['holdout_loss_improved'], checkpoint=str(checkpoint))), flush=True)
        if not report['passed']:
            raise RuntimeError('multi-k training gate failed; see report.json')
    except Exception as exc:
        _write_json(args.output_dir / 'failure.json', dict(error_type=type(exc).__name__, error=str(exc),
                                                         elapsed_seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
