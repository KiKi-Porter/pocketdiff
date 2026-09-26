"""Audit real heavy-atom χ rotations without training or running a sampler."""
import argparse
import json
from pathlib import Path

import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter, AA_NAMES
from pocketdiff.geometry.chi import (
    SIDECHAIN_BONDS, build_chi_update_metadata, apply_chi_updates,
    extract_chi_angles, periodic_chi_delta,
)
from pocketdiff.scripts.phase28_chi_head_smoke import _sha


def audit(value):
    names = [AA_NAMES[i] for i in value.residue_type.tolist()]
    ids = value.atom_to_residue
    atom_names = value.protein_atom_name
    pos = value.protein_pos_apo
    before = pos.clone()
    meta = build_chi_update_metadata(atom_names, ids, names)
    active = meta.valid & value.chi_mask
    apo, apo_valid = extract_chi_angles(pos, atom_names, ids, names)
    target, target_valid = extract_chi_angles(value.protein_pos_holo, atom_names, ids, names)
    delta = periodic_chi_delta(apo, target, active & apo_valid & target_valid)
    updated = apply_chi_updates(pos, meta.axis_start, meta.axis_end,
                               meta.downstream_atom_mask, delta, valid=active)
    actual, actual_valid = extract_chi_angles(updated.positions, atom_names, ids, names)
    error = periodic_chi_delta(apo, actual, active) - delta
    error = torch.atan2(error.sin(), error.cos())
    max_angle_error = float(error[active].abs().max()) if active.any() else 0.
    edges = []
    for r, name in enumerate(names):
        by_name = {atom_names[i]: i for i in torch.where(ids == r)[0].tolist()}
        for pair in ('N-CA CA-C C-O C-OXT ' + SIDECHAIN_BONDS[name]).split():
            a, b = pair.split('-')
            if a in by_name and b in by_name:
                edges.append((by_name[a], by_name[b]))
    edges = torch.tensor(edges, dtype=torch.long)
    lengths = lambda p: (p[edges[:, 0]] - p[edges[:, 1]]).norm(dim=-1)
    max_bond_error = float((lengths(updated.positions) - lengths(pos)).abs().max())
    backbone = torch.tensor([a in ('N', 'CA', 'C', 'O', 'OXT') for a in atom_names])
    pro = torch.tensor([name == 'PRO' for name in names])
    excluded = value.chi_mask & ~meta.valid
    zero = apply_chi_updates(pos, meta.axis_start, meta.axis_end,
                            meta.downstream_atom_mask, torch.zeros_like(delta), valid=active)
    guards = dict(
        finite=bool(torch.isfinite(updated.positions).all()),
        input_unchanged=torch.equal(before, pos),
        backbone_unchanged=torch.equal(pos[backbone], updated.positions[backbone]),
        proline_unchanged=torch.equal(pos[pro[ids]], updated.positions[pro[ids]]),
        zero_exact=torch.equal(zero.positions, pos),
        active_preserved=bool((actual_valid[active] & updated.valid[active]).all()),
        angle_readback=max_angle_error < 2e-4,
        bond_lengths=max_bond_error < 2e-5,
    )
    return dict(sample_id=value.sample_id, residues=len(names), atoms=pos.shape[0],
                raw_valid_chi=int(value.chi_mask.sum()), rotatable_chi=int(active.sum()),
                excluded_proline=int(excluded[pro].sum()),
                excluded_missing_canonical=int(excluded[~pro].sum()),
                max_angle_error_rad=max_angle_error, max_bond_error_angstrom=max_bond_error,
                guards=guards, passed=all(guards.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=Path(
        '.codex-tasks/pocketdiff-development/phase6b-clean-generalization/raw/manifest.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=16)
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())['entries']
    if not 1 <= args.limit <= len(entries):
        parser.error('limit must be within available manifest entries')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output directory must be empty')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter = Apo2MolAdapter('.')
    rows = []
    for entry in entries[:args.limit]:
        source = entry['source']
        hashes = {key: _sha(Path(info['path'])) for key, info in source.items()}
        if any(hashes[key] != info['sha256'] for key, info in source.items()):
            raise ValueError('source changed: ' + entry['sample_id'])
        value, _ = adapter.convert_paths(source['holo_pocket']['path'],
                                        source['apo_pocket']['path'], source['ligand']['path'],
                                        sample_id=entry['sample_id'])
        row = audit(value)
        row['source_sha256'] = hashes
        rows.append(row)
    report = dict(passed=all(row['passed'] for row in rows), mode='chi_topology_only',
                  selection='manifest first N entries; no train/holdout roles assigned',
                  manifest_sha256=_sha(args.manifest), sample_count=len(rows), results=rows,
                  totals={key: sum(row[key] for row in rows) for key in (
                      'raw_valid_chi', 'rotatable_chi', 'excluded_proline', 'excluded_missing_canonical')},
                  max_angle_error_rad=max(row['max_angle_error_rad'] for row in rows),
                  max_bond_error_angstrom=max(row['max_bond_error_angstrom'] for row in rows),
                  scope='residue-local heavy atoms; no cross-residue links, training, cache writes or sampling')
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output_dir / 'code_fingerprints.json').write_text(json.dumps(
        {str(p): _sha(p) for p in sorted(Path('pocketdiff').rglob('*.py'))}, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'results'}), flush=True)
    if not report['passed']:
        raise RuntimeError('real χ topology audit failed; see report.json')


if __name__ == '__main__':
    main()
