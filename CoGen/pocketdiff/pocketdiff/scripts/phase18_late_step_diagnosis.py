"""Read-only amplitude/error/time audit of the Phase 17 clean checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from pocketdiff.models import PocketDiffModel
from pocketdiff.preprocessing import load_cached_clean_examples
from pocketdiff.training import build_bridge_batch, collate_clean_examples, evaluate_multik_clean
from pocketdiff.scripts.phase17_multik_train import _write_json, _sha


def load_baseline_data(baseline_dir):
    report = json.loads((baseline_dir / 'report.json').read_text())
    examples = load_cached_clean_examples(baseline_dir / 'manifest.json', verify_sources=True)
    by_id = {e.complex_value.sample_id: e for e in examples}
    batches = {role: collate_clean_examples([by_id[s] for s in ids])
               for role, ids in report['preparation']['roles'].items()}
    return report, batches


@torch.no_grad()
def diagnose(model, clean):
    model.eval()
    rows = []
    for k in range(20):
        batch = build_bridge_batch(clean, torch.full((len(clean.sample_ids),), k, dtype=torch.long))
        pred = model(**batch.model_kwargs())
        for graph, sample_id in enumerate(clean.sample_ids):
            mask = batch.frame_valid & pred.frame_valid & (batch.batch_residue == graph)
            row = dict(sample_id=sample_id, k=k, remaining_fraction=(20-k)/20)
            for name, predicted, target in (
                ('translation', pred.remaining_translation_local, batch.target_translation_local),
                ('rotation', pred.remaining_rotvec_local, batch.target_rotvec_local),
            ):
                p, y = predicted[mask], target[mask]
                error = p-y
                bias = error.mean(0)
                row[name] = dict(target_rms=float(y.square().mean().sqrt()),
                                 prediction_rms=float(p.square().mean().sqrt()),
                                 target_rate_rms=float((y/row['remaining_fraction']).square().mean().sqrt()),
                                 error_mse=float(error.square().mean()),
                                 bias_mse=float(bias.square().mean()),
                                 centered_error_mse=float((error-bias).square().mean()),
                                 signed_local_error_bias=bias.tolist())
            rows.append(row)
    per_k = []
    for k in range(20):
        subset = [r for r in rows if r['k'] == k]
        per_k.append(dict(k=k, **{name: {key: sum(r[name][key] for r in subset)/len(subset)
                                        for key in ('target_rms', 'prediction_rms', 'target_rate_rms',
                                                    'error_mse', 'bias_mse', 'centered_error_mse')}
                                 for name in ('translation', 'rotation')}))
    # Intentionally inconsistent times on fixed P_19: sensitivity only, not a
    # validation distribution or alternative supervised example.
    frozen = build_bridge_batch(clean, torch.full((len(clean.sample_ids),), 19, dtype=torch.long))
    sensitivity = []
    reference = model(**frozen.model_kwargs())
    for k in (0, 9, 18, 19):
        kwargs = frozen.model_kwargs()
        kwargs['pocket_k'] = torch.full_like(frozen.pocket_k, k)
        kwargs['targetdiff_t'] = 199-10*kwargs['pocket_k']
        pred = model(**kwargs)
        mask = frozen.frame_valid & pred.frame_valid
        sensitivity.append(dict(input_k=k, frozen_structure_k=19,
            translation_delta_rms=float((pred.remaining_translation_local[mask]-reference.remaining_translation_local[mask]).square().mean().sqrt()),
            rotation_delta_rms=float((pred.remaining_rotvec_local[mask]-reference.remaining_rotvec_local[mask]).square().mean().sqrt())))
    return dict(per_graph=rows, per_k=per_k, fixed_structure_time_sensitivity=sensitivity)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-dir', type=Path, default=Path('.codex-tasks/pocketdiff-development/phase17-multik-clean-training/raw/run'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    baseline, batches = load_baseline_data(args.baseline_dir)
    checkpoint = args.baseline_dir / 'checkpoint_0400.pt'
    payload = torch.load(checkpoint, map_location='cpu')
    model = PocketDiffModel(**payload['model_config'])
    model.load_state_dict(payload['model_state_dict'], strict=True)
    model.eval()
    result = dict(checkpoint_sha256=_sha(checkpoint), diagnosis={}, baseline_reproduction_max_error=0.0,
                  scope='teacher-forced clean; frozen-structure wrong times are sensitivity probes only')
    for role, clean in batches.items():
        metrics = evaluate_multik_clean(model, clean)
        error = max(abs(metrics['per_k'][k][key]-baseline['final'][role]['per_k'][k][key])
                    for k in range(20) for key in ('loss', 'next_bridge_rmsd', 'endpoint_holo_rmsd'))
        result['baseline_reproduction_max_error'] = max(result['baseline_reproduction_max_error'], error)
        result['diagnosis'][role] = diagnose(model, clean)
    result['passed'] = result['baseline_reproduction_max_error'] < 1e-7
    _write_json(args.output, result)
    print(json.dumps(dict(passed=result['passed'], baseline_reproduction_max_error=result['baseline_reproduction_max_error'])))
    if not result['passed']:
        raise RuntimeError('baseline no longer reproduces; inspect diagnosis report')


if __name__ == '__main__':
    main()
