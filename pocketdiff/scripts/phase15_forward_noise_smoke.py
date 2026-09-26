"""Check reference ligand forward noise against official TargetDiff at 20 times."""

import argparse
from dataclasses import fields
import hashlib
import json
import pickle
from pathlib import Path

import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.targetdiff import (
    TargetDiffAdapter, forward_noise_reference, initialize_targetdiff_state,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('Apo2Mol-main/Apo2MOl-dataset/data_folder'))
    parser.add_argument('--split-pickle', type=Path, default=Path('Apo2Mol-main/Apo2MOl-dataset/split_druglike_dict.pkl'))
    parser.add_argument('--checkpoint', type=Path, default=Path('targetdiff-main/targetdiff-main/pretrained_models/pretrained_diffusion.pt'))
    parser.add_argument('--seed', type=int, default=20260919)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'passed': False, 'seed': args.seed, 'condition': 'reference_forward_noised',
              'scope': 'condition contract only; no training or generated-ligand quality claims'}
    try:
        with args.split_pickle.open('rb') as handle:
            records = pickle.load(handle)['train']
        record = next(r for r in records if str(r[0]).startswith('3txj') or str(r[1]).startswith('3txj'))
        value = Apo2MolAdapter(args.data_root).convert_record(record)
        state = initialize_targetdiff_state(
            protein_pos=value.protein_pos_apo, protein_v=value.protein_feature,
            batch_protein=torch.zeros(value.num_protein_atoms, dtype=torch.long),
            ligand_pos=value.ligand_pos_ref, ligand_v=value.ligand_type_ref,
            batch_ligand=torch.zeros(value.num_ligand_atoms, dtype=torch.long),
        )
        snapshots = {f.name: getattr(state, f.name).clone() for f in fields(state)}
        adapter = TargetDiffAdapter.from_checkpoint(args.checkpoint, device='cpu')
        # Imported only after the adapter selects the read-only upstream source.
        from models.molopt_score_model import index_to_log_onehot
        log_v0 = index_to_log_onehot(state.ligand_v, 13)
        rows = []
        for k in range(20):
            t = 199 - 10 * k
            times = torch.tensor([t])
            seed = args.seed + k
            out = forward_noise_reference(adapter, state, times,
                                          generator=torch.Generator().manual_seed(seed))
            # Independent official training formula and q_v_sample call.
            with torch.no_grad(), torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                a = adapter.model.alphas_cumprod.index_select(0, times)[state.batch_ligand].unsqueeze(-1)
                noise = torch.randn_like(state.ligand_pos)
                expected_pos = a.sqrt() * state.ligand_pos + (1 - a).sqrt() * noise
                expected_v, _ = adapter.model.q_v_sample(log_v0, times, state.batch_ligand)
            delta = torch.linalg.vector_norm(out.ligand_pos - state.ligand_pos, dim=-1)
            row = {
                'k': k, 't': t, 'alpha_bar_position': float(a[0, 0]),
                'alpha_bar_type': float(adapter.model.log_alphas_cumprod_v[t].exp()),
                'official_position_max_error': float((out.ligand_pos - expected_pos).abs().max()),
                'official_type_mismatches': int((out.ligand_v != expected_v).sum()),
                'reference_ligand_rmsd': float(delta.square().mean().sqrt()),
                'reference_type_changed_fraction': float((out.ligand_v != state.ligand_v).float().mean()),
                'finite': bool(torch.isfinite(out.ligand_pos).all()),
                'type_valid': bool(((out.ligand_v >= 0) & (out.ligand_v < 13)).all()),
                'input_unchanged': all(torch.equal(getattr(state, n), v) for n, v in snapshots.items()),
                'protein_and_offset_unchanged': all(torch.equal(getattr(out, n), v) for n, v in snapshots.items()
                                                  if n not in ('ligand_pos', 'ligand_v')),
            }
            row['passed'] = (row['official_position_max_error'] < 1e-6
                             and row['official_type_mismatches'] == 0
                             and all(row[n] for n in ('finite', 'type_valid', 'input_unchanged', 'protein_and_offset_unchanged')))
            rows.append(row)
        digest = hashlib.sha256()
        with args.checkpoint.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        report.update(sample_id=value.sample_id, num_ligand_atoms=value.num_ligand_atoms,
                      checkpoint=str(args.checkpoint), checkpoint_sha256=digest.hexdigest(),
                      torch_version=torch.__version__, rows=rows,
                      passed=all(row['passed'] for row in rows))
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'passed': report['passed'], 'sample_id': report['sample_id'],
                      'timesteps_checked': len(report['rows']), 'output': str(args.output)}))
    if not report['passed']:
        raise RuntimeError('forward-noise smoke failed; see saved report')


if __name__ == '__main__':
    main()
