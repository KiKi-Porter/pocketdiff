"""Fixed clean autonomous trajectories: Phase17, Phase18 and zero update."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from pocketdiff.evaluation.clean_rollout import score_clean_rollout
from pocketdiff.models import PocketDiffModel
from pocketdiff.preprocessing import GEOMETRY_VERSION, load_cached_clean_examples
from pocketdiff.sampling.clean_rollout import INFERENCE_FIELDS, run_clean_rollout
from pocketdiff.scripts.phase17_multik_train import _sha, _write_json
from pocketdiff.scripts.phase18_late_step_diagnosis import load_baseline_data
from pocketdiff.training import build_bridge_batch, collate_clean_examples
from pocketdiff.training.multik import CHECKPOINT_FORMAT, endpoint_fingerprint


def inference_inputs(clean):
    # Intentionally excludes holo, labels, clean.frame_valid and current state.
    return {name: getattr(clean, name) for name in INFERENCE_FIELDS}


def load_model(path):
    payload = torch.load(path, map_location='cpu')
    if payload['format'] != CHECKPOINT_FORMAT or payload['geometry_version'] != GEOMETRY_VERSION:
        raise ValueError('incompatible checkpoint format/geometry: ' + str(path))
    model = PocketDiffModel(**payload['model_config'])
    model.load_state_dict(payload['model_state_dict'], strict=True)
    return model.eval(), payload['model_config']


@torch.no_grad()
def add_bridge_distance(metrics, trace, clean):
    """Post-hoc scoring only: ideal bridge coordinates never enter inference."""
    atoms_valid = trace.frame_valid[0][clean.atom_to_residue_global]
    by_step = {0: clean.apo_pos_ref}
    for step in range(1, 20):
        bridge = build_bridge_batch(clean, torch.full_like(clean.pocket_k, step))
        by_step[step] = bridge.protein_pos
        if step == 19:
            by_step[20] = bridge.protein_pos_next_target
    for row in metrics['per_graph']:
        graph = clean.sample_ids.index(row['sample_id'])
        mask = atoms_valid & (clean.batch_protein == graph)
        step = row['step']
        row['ideal_bridge_distance_rmsd'] = float(
            (trace.positions[step, mask]-by_step[step][mask]).square().sum(-1).mean().sqrt())
    for row in metrics['per_step']:
        subset = [v['ideal_bridge_distance_rmsd'] for v in metrics['per_graph'] if v['step'] == row['step']]
        row['ideal_bridge_distance_rmsd'] = sum(subset)/len(subset)
    metrics['ideal_bridge_scoring_only'] = True
    metrics['apo_mask_matches_training_mask'] = torch.equal(trace.frame_valid[0], clean.frame_valid)


def summarize(results, prior):
    comparison = {}
    for role, variants in results.items():
        initial = variants['zero_update']['per_step'][0]
        rows = []
        summaries = {}
        for variant, metrics in variants.items():
            final = metrics['per_step'][-1]
            final_rows = [row for row in metrics['per_graph'] if row['step'] == 20]
            initial_rows = {row['sample_id']: row for row in metrics['per_graph'] if row['step'] == 0}
            summaries[variant] = dict(
                apo_holo_rmsd=initial['holo_rmsd'], final_holo_rmsd=final['holo_rmsd'],
                improvement=initial['holo_rmsd']-final['holo_rmsd'],
                final_backbone_holo_rmsd=final['backbone_holo_rmsd'],
                max_apo_displacement=final['max_apo_displacement'],
                final_ideal_bridge_distance_rmsd=final['ideal_bridge_distance_rmsd'],
                improved_samples=sum(row['holo_rmsd'] < initial_rows[row['sample_id']]['holo_rmsd']-1e-7
                                     for row in final_rows),
                regressed_samples=sum(row['holo_rmsd'] > initial_rows[row['sample_id']]['holo_rmsd']+1e-7
                                      for row in final_rows),
                sample_count=len(final_rows))
            if variant in prior:
                teacher = prior[variant]['final'][role]
                summaries[variant]['teacher_forced_k0_full_remaining_holo_rmsd'] = teacher['per_k'][0]['endpoint_holo_rmsd']
                summaries[variant]['teacher_forced_k19_full_remaining_holo_rmsd'] = teacher['per_k'][19]['endpoint_holo_rmsd']
        sample_ids = [row['sample_id'] for row in variants['zero_update']['per_graph'] if row['step'] == 0]
        for sample_id in sample_ids:
            row = dict(sample_id=sample_id)
            for variant, metrics in variants.items():
                final = next(v for v in metrics['per_graph'] if v['sample_id'] == sample_id and v['step'] == 20)
                row[variant] = {key: final[key] for key in
                                ('holo_rmsd', 'backbone_holo_rmsd', 'max_apo_displacement', 'ideal_bridge_distance_rmsd')}
            rows.append(row)
        comparison[role] = dict(summary=summaries, per_sample=rows)
    return comparison


def plot_results(output_dir, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for column, role in enumerate(('train', 'holdout')):
        for name, metrics in results[role].items():
            steps = metrics['per_step']
            axes[0, column].plot(range(21), [v['holo_rmsd'] for v in steps], label=name)
            axes[1, column].plot(range(21), [v['ideal_bridge_distance_rmsd'] for v in steps], label=name)
        axes[0, column].set(title=role + ': autonomous apo to holo', ylabel='Holo RMSD (angstrom)')
        axes[1, column].set(title=role + ': deviation from ideal training bridge', ylabel='Bridge distance (angstrom)')
    for ax in axes.flat:
        ax.set_xlabel('Completed pocket updates')
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir/'rollout_comparison.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    root = Path('.codex-tasks/pocketdiff-development')
    parser.add_argument('--baseline-dir', type=Path, default=root/'phase17-multik-clean-training/raw/run')
    parser.add_argument('--rate-dir', type=Path, default=root/'phase18-late-step-diagnosis/raw/run')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output directory must be empty; do not overwrite previous evidence')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.monotonic()
    try:
        baseline, batches = load_baseline_data(args.baseline_dir)
        rate_report = json.loads((args.rate_dir/'report.json').read_text())
        if any(len(batch.sample_ids) != 8 for batch in batches.values()):
            raise ValueError('expected the fixed eight samples per role')
        fingerprint_before = {role: endpoint_fingerprint(batch) for role, batch in batches.items()}
        for role, value in fingerprint_before.items():
            if (value != rate_report['provenance']['endpoint_fingerprints'][role]
                    or value != baseline['preparation'][role+'_endpoint_fingerprint']):
                raise ValueError('data fingerprint differs from previous experiments')
        paths = dict(zero_update=args.baseline_dir/'checkpoint_0000.pt',
                     phase17=args.baseline_dir/'checkpoint_0400.pt',
                     phase18=args.rate_dir/'checkpoint_0400.pt')
        tracked = list(paths.values()) + [args.baseline_dir/'report.json', args.rate_dir/'report.json',
                                         args.baseline_dir/'manifest.json']
        hashes = {str(path): _sha(path) for path in tracked}
        if hashes[str(paths['phase17'])] != baseline['checkpoint_sha256'] or hashes[str(paths['phase18'])] != rate_report['checkpoint_sha256']:
            raise ValueError('checkpoint differs from original training report')
        models, configs = {}, {}
        for name, path in paths.items():
            models[name], configs[name] = load_model(path)
        _write_json(args.output_dir/'provenance.json', dict(source_sha256=hashes, model_configs=configs,
                                                         endpoint_fingerprints=fingerprint_before))
        _write_json(args.output_dir/'code_fingerprints.json', {
            str(path): _sha(path) for path in sorted(Path('pocketdiff').rglob('*.py'))})

        # Real one-example gate before expanding to the fixed 8/8 comparison.
        examples = load_cached_clean_examples(args.baseline_dir/'manifest.json', verify_sources=True)
        first = next(e for e in examples if e.complex_value.sample_id == batches['train'].sample_ids[0])
        single = collate_clean_examples([first])
        smoke = {}
        for name, model in models.items():
            inputs = inference_inputs(single)
            trace = run_clean_rollout(model, inputs)
            metrics = score_clean_rollout(trace, inputs, single.protein_pos_holo, single.sample_ids)
            smoke[name] = metrics['per_step'][-1]
            if any(row['lost_apo_frame_count'] or row['skipped_valid_update_count'] for row in metrics['per_step']):
                _write_json(args.output_dir/'smoke.json', smoke)
                raise RuntimeError('real single-example frame/update gate failed')
        _write_json(args.output_dir/'smoke.json', dict(sample_id=single.sample_ids[0], passed=True, results=smoke))
        print(json.dumps(dict(smoke_passed=True, sample_id=single.sample_ids[0])), flush=True)

        results, trace_artifacts = {}, {}
        zero_exact = True
        rng_unchanged = True
        for role, clean in batches.items():
            results[role] = {}
            inputs = inference_inputs(clean)
            for name, model in models.items():
                rng = torch.get_rng_state().clone()
                trace = run_clean_rollout(model, inputs)
                rng_unchanged &= torch.equal(rng, torch.get_rng_state())
                if name == 'zero_update':
                    zero_exact &= torch.equal(trace.positions, clean.apo_pos_ref.expand(21, -1, -1))
                # Save completed inference before any target-dependent scoring.
                path = args.output_dir/(role+'_'+name+'_trajectory.pt')
                torch.save(dict(format='pocketdiff-clean-rollout-v1', sample_ids=clean.sample_ids,
                                batch_protein=clean.batch_protein, batch_residue=clean.batch_residue,
                                **asdict(trace)), path)
                trace_artifacts[role+'_'+name] = dict(path=str(path), sha256=_sha(path))
                metrics = score_clean_rollout(trace, inputs, clean.protein_pos_holo, clean.sample_ids)
                add_bridge_distance(metrics, trace, clean)
                results[role][name] = metrics
                _write_json(args.output_dir/(role+'_'+name+'_metrics.json'), metrics)
                print(json.dumps(dict(role=role, model=name, final_holo_rmsd=metrics['per_step'][-1]['holo_rmsd'])), flush=True)
        guards = dict(
            zero_update_exact=zero_exact, inference_rng_unchanged=rng_unchanged,
            source_files_unchanged=all(_sha(path) == digest for path, digest in hashes.items()),
            inputs_unchanged=all(endpoint_fingerprint(batches[role]) == value for role, value in fingerprint_before.items()),
            all_trajectories_complete=all(len(m['per_step']) == 21 and len(m['per_graph']) == 21*8
                                          for variants in results.values() for m in variants.values()),
            no_lost_frames_or_skipped_valid_updates=all(
                row['lost_apo_frame_count'] == 0 and row['skipped_valid_update_count'] == 0
                for variants in results.values() for m in variants.values() for row in m['per_step']),
            apo_mask_matches_previous_evaluations=all(m['apo_mask_matches_training_mask']
                                                     for variants in results.values() for m in variants.values()),
        )
        report = dict(passed=all(guards.values()), guards=guards, mode='autonomous_clean',
                      steps=20, training_performed=False, elapsed_seconds=time.monotonic()-started,
                      roles={role: b.sample_ids for role, b in batches.items()},
                      comparison=summarize(results, dict(phase17=baseline, phase18=rate_report)),
                      results=results, trajectories=trace_artifacts,
                      limitations=['8/8 development samples; not a strict test set',
                                   'Clean reference ligand only; no generated/noised ligand',
                                   'No holo input or ideal bridge reset during inference',
                                   'passed describes engineering checks, not apo-to-holo learning success'])
        _write_json(args.output_dir/'report.json', report)
        plot_results(args.output_dir, results)
        print(json.dumps(dict(passed=report['passed'], guards=guards)), flush=True)
        if not report['passed']:
            raise RuntimeError('autonomous rollout engineering checks failed; inspect saved report')
    except Exception as exc:
        _write_json(args.output_dir/'failure.json', dict(error_type=type(exc).__name__, error=str(exc),
                                                       elapsed_seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
