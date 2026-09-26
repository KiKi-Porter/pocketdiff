"""Real Apo2Mol adapter smoke for chi-angle extraction."""

import argparse
import json
from pathlib import Path
import pickle

import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.geometry import periodic_chi_delta
from pocketdiff.scripts.phase17_multik_train import _sha, _write_json


def main():
    parser = argparse.ArgumentParser()
    root = Path('.codex-tasks/pocketdiff-development')
    parser.add_argument('--manifest', type=Path,
                        default=root / 'phase6b-clean-generalization/raw/manifest.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=16)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error('limit must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output directory must be empty')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    entries = manifest['entries'][:args.limit]
    adapter = Apo2MolAdapter('.')
    results = []
    for entry in entries:
        source = entry['source']
        value, diagnostics = adapter.convert_paths(
            source['holo_pocket']['path'],
            source['apo_pocket']['path'],
            source['ligand']['path'],
            sample_id=entry['sample_id'],
        )
        delta = periodic_chi_delta(value.chi_apo, value.chi_holo, value.chi_mask)
        valid = value.chi_mask
        finite = bool(torch.isfinite(value.chi_apo).all()
                      and torch.isfinite(value.chi_holo).all()
                      and torch.isfinite(delta).all())
        numbers = delta[valid]
        results.append(dict(
            sample_id=value.sample_id,
            num_residues=value.num_residues,
            chi_shape=list(value.chi_apo.shape),
            valid_chi_count=int(valid.sum()),
            valid_residue_count=int(valid.any(dim=1).sum()),
            finite=finite,
            chi_min=float(numbers.min()) if numbers.numel() else 0.0,
            chi_max=float(numbers.max()) if numbers.numel() else 0.0,
            aligned_calpha_rmsd=diagnostics.aligned_calpha_rmsd,
            source_sha256={
                name: info['sha256'] for name, info in source.items()
            },
        ))
        if value.chi_apo.shape != (value.num_residues, 5):
            raise RuntimeError('chi shape mismatch for ' + value.sample_id)
        if not finite or (numbers.numel() and bool((numbers.abs() > 3.141593).any())):
            raise RuntimeError('invalid chi values for ' + value.sample_id)
    report = dict(
        passed=bool(results) and all(value['finite'] for value in results),
        mode='phase27_real_chi_geometry_smoke',
        sample_count=len(results),
        source_manifest_sha256=_sha(args.manifest),
        results=results,
        total_valid_chi_count=sum(value['valid_chi_count'] for value in results),
        total_valid_residue_count=sum(value['valid_residue_count'] for value in results),
        scope='raw adapter labels only; no cache rewrite and no optimization',
    )
    _write_json(args.output_dir / 'report.json', report)
    _write_json(args.output_dir / 'code_fingerprints.json', {
        str(path): _sha(path)
        for path in sorted(Path('pocketdiff').rglob('*.py'))
    })
    print(json.dumps(dict(
        passed=report['passed'],
        sample_count=report['sample_count'],
        total_valid_chi_count=report['total_valid_chi_count'],
    )), flush=True)
    if not report['passed']:
        raise RuntimeError('chi geometry smoke failed')


if __name__ == '__main__':
    main()
