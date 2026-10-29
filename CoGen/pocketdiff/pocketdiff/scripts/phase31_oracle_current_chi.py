"""Measure rigid + chi reconstruction on raw structures; no learned model."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from pocketdiff.data.apo2mol_adapter import AA_NAMES, Apo2MolAdapter
from pocketdiff.geometry import (
    build_current_chi_state, oracle_rigid_chi_reconstruction, periodic_chi_delta,
)
from pocketdiff.geometry.chi import SIDECHAIN_BONDS
from pocketdiff.geometry.current_state import CURRENT_CHI_VERSION


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rmsd(pos, target, mask):
    if not mask.any():
        return None
    return float((pos[mask] - target[mask]).square().sum(-1).mean().sqrt())


def audit(value):
    apo, holo = value.protein_pos_apo, value.protein_pos_holo
    names, ids = value.protein_atom_name, value.atom_to_residue
    residues = [AA_NAMES[i] for i in value.residue_type.tolist()]
    before = {k: v.clone() for k, v in vars(value).items() if isinstance(v, torch.Tensor)}
    out = oracle_rigid_chi_reconstruction(apo, holo, ids, names, residues, value.frame_valid)
    atom_valid = out.frame_valid[ids]
    backbone = torch.tensor([name in ('N', 'CA', 'C', 'O', 'OXT') for name in names])
    target = build_current_chi_state(holo, names, ids, residues)
    readback = build_current_chi_state(out.rigid_chi_positions, names, ids, residues)
    active = out.supervision_mask
    error = periodic_chi_delta(readback.angles, target.angles, active)
    max_error = float(error[active].abs().max()) if active.any() else 0.
    edges = []
    per_residue = []
    for r, residue in enumerate(residues):
        by_name = {names[i]: i for i in torch.where(ids == r)[0].tolist()}
        for pair in ('N-CA CA-C C-O C-OXT ' + SIDECHAIN_BONDS[residue]).split():
            a, b = pair.split('-')
            if a in by_name and b in by_name:
                edges.append((by_name[a], by_name[b]))
        mask = (ids == r) & atom_valid
        per_residue.append(dict(
            residue_index=r, residue=residue, chain=value.residue_chain_id[r],
            sequence=value.residue_sequence_id[r],
            apo_rmsd=rmsd(apo, holo, mask),
            rigid_rmsd=rmsd(out.rigid_only_positions, holo, mask),
            rigid_chi_rmsd=rmsd(out.rigid_chi_positions, holo, mask),
            rotatable_chi=int(out.current_chi.geometry_rotatable_mask[r].sum()),
        ))
    edges = torch.tensor(edges, dtype=torch.long)
    def lengths(pos):
        return (pos[edges[:, 0]] - pos[edges[:, 1]]).norm(dim=-1)
    bond_error = float((lengths(apo) - lengths(out.rigid_chi_positions)).abs().max())
    # Recompute the inference state independently using current inputs only.
    # Adversarial holo degeneracy is covered by the separate oracle unit test.
    current_again = build_current_chi_state(apo, names, ids, residues)
    pro = torch.tensor([residue == 'PRO' for residue in residues])
    apo_chi_observed = int(value.chi_mask.sum())  # Historical label count only.
    guards = dict(
        finite=bool(torch.isfinite(out.rigid_chi_positions).all()),
        input_unchanged=all(torch.equal(getattr(value, k), v) for k, v in before.items()),
        current_state_repeatable=torch.equal(current_again.angles, out.current_chi.angles),
        masks_separated=not bool((out.training_safe_mask & ~out.supervision_mask).any()),
        ambiguous_training_excluded=not bool((out.training_safe_mask & current_again.ambiguous_chi_mask).any()),
        backbone_unchanged_by_chi=torch.equal(out.rigid_only_positions[backbone], out.rigid_chi_positions[backbone]),
        proline_internal_unchanged=torch.equal(out.rigid_only_positions[pro[ids]], out.rigid_chi_positions[pro[ids]]),
        active_angles_preserved=bool(readback.geometry_rotatable_mask[active].all()),
        chi_readback=max_error < 2e-4,
        residue_bond_lengths=bond_error < 2e-5,
    )
    row = dict(
        sample_id=value.sample_id, atoms=len(names), residues=len(residues),
        apo_atom_rmsd=rmsd(apo, holo, atom_valid),
        rigid=asdict(out.rigid_metrics), rigid_chi=asdict(out.rigid_chi_metrics),
        apo_backbone_rmsd=rmsd(apo, holo, atom_valid & backbone),
        apo_sidechain_rmsd=rmsd(apo, holo, atom_valid & ~backbone),
        rigid_sidechain_rmsd=rmsd(out.rigid_only_positions, holo, atom_valid & ~backbone),
        rigid_chi_sidechain_rmsd=rmsd(out.rigid_chi_positions, holo, atom_valid & ~backbone),
        label_chi_count=apo_chi_observed,
        geometry_rotatable_chi=int(current_again.geometry_rotatable_mask.sum()),
        oracle_supervised_chi=int(active.sum()),
        training_safe_chi=int(out.training_safe_mask.sum()),
        ambiguous_excluded_from_training=int((active & current_again.ambiguous_chi_mask).sum()),
        proline_label_chi_excluded=int(value.chi_mask[pro].sum()),
        max_angle_error_rad=max_error, max_bond_error_angstrom=bond_error,
        guards=guards, passed=all(guards.values()), per_residue=per_residue,
    )
    coordinates = dict(
        sample_id=value.sample_id, apo=apo, holo=holo,
        rigid_only=out.rigid_only_positions, rigid_chi=out.rigid_chi_positions,
        current_chi=out.current_chi.angles, applied_chi=out.applied_chi,
        geometry_rotatable_mask=out.current_chi.geometry_rotatable_mask,
        oracle_supervision_mask=active, training_safe_mask=out.training_safe_mask,
    )
    return row, coordinates


def main():
    parser = argparse.ArgumentParser()
    root = Path('.codex-tasks/pocketdiff-development')
    parser.add_argument('--manifest', type=Path, default=root / 'phase6b-clean-generalization/raw/manifest.json')
    parser.add_argument('--output-dir', type=Path, default=root / 'phase31-oracle-current-chi/raw/run')
    parser.add_argument('--limit', type=int, default=16)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if not 1 <= args.limit <= len(manifest['entries']):
        parser.error('limit must lie within manifest entries')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output directory must be empty; preserve prior results')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    adapter = Apo2MolAdapter('.')
    rows, coordinates = [], []
    for entry in manifest['entries'][:args.limit]:
        source = entry['source']
        hashes = {k: sha(info['path']) for k, info in source.items()}
        if any(hashes[k] != info['sha256'] for k, info in source.items()):
            raise ValueError('source changed: ' + entry['sample_id'])
        value, _ = adapter.convert_paths(source['holo_pocket']['path'], source['apo_pocket']['path'],
                                        source['ligand']['path'], sample_id=entry['sample_id'])
        row, positions = audit(value)
        row['source_sha256'] = hashes
        rows.append(row)
        coordinates.append(positions)
    summary = dict(
        mean_apo_atom_rmsd=sum(r['apo_atom_rmsd'] for r in rows) / len(rows),
        mean_rigid_atom_rmsd=sum(r['rigid']['atom_rmsd'] for r in rows) / len(rows),
        mean_rigid_chi_atom_rmsd=sum(r['rigid_chi']['atom_rmsd'] for r in rows) / len(rows),
        mean_rigid_backbone_rmsd=sum(r['rigid']['backbone_rmsd'] for r in rows) / len(rows),
        mean_rigid_chi_backbone_rmsd=sum(r['rigid_chi']['backbone_rmsd'] for r in rows) / len(rows),
        improved_over_rigid=sum(r['rigid_chi']['atom_rmsd'] < r['rigid']['atom_rmsd'] for r in rows),
        improved_over_apo=sum(r['rigid_chi']['atom_rmsd'] < r['apo_atom_rmsd'] for r in rows),
        max_angle_error_rad=max(r['max_angle_error_rad'] for r in rows),
        max_bond_error_angstrom=max(r['max_bond_error_angstrom'] for r in rows),
    )
    summary.update({key: sum(r[key] for r in rows) for key in (
        'label_chi_count', 'geometry_rotatable_chi', 'oracle_supervised_chi', 'training_safe_chi',
        'ambiguous_excluded_from_training', 'proline_label_chi_excluded',
    )})
    report = dict(
        passed=all(r['passed'] for r in rows), learning_goal_met=False,
        current_state_version=CURRENT_CHI_VERSION, sample_count=len(rows), summary=summary,
        selection='manifest first N entries; diagnostic only; not Phase17 train/holdout roles',
        metric='per-complex RMSD then equal-weight mean; frame-valid named heavy atoms; no realignment',
        limitation='reference-frame/chi reconstruction residual; not an optimized lower bound or learned prediction',
        symmetry_policy='named oracle retains ambiguous slots; training_safe_mask excludes them until equivalent-target loss',
        manifest_sha256=sha(args.manifest), source_manifest_geometry_version=manifest['geometry_version'],
        cache_policy='source paths only; original cache tensors not used; no cache migration',
        results=rows,
    )
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    torch.save(dict(format='pocketdiff-oracle-diagnostic-v1', results=coordinates), args.output_dir / 'coordinates.pt')
    (args.output_dir / 'code_fingerprints.json').write_text(json.dumps(
        {str(p): sha(p) for p in sorted(Path('pocketdiff').rglob('*.py'))}, indent=2) + '\n')
    print(json.dumps(dict(passed=report['passed'], sample_count=len(rows), **summary)), flush=True)
    if not report['passed']:
        raise RuntimeError('oracle contract failed; see report.json')


if __name__ == '__main__':
    main()
