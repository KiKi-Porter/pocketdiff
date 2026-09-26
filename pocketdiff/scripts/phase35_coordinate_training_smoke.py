"""Three optimizer steps on raw 3txj; a training-path check, not generalization."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from pocketdiff.data.apo2mol_adapter import AA_NAMES, Apo2MolAdapter
from pocketdiff.geometry.bridge import apply_fractional_update, remaining_transform_current_to_holo
from pocketdiff.geometry.chi import apply_chi_updates, periodic_chi_delta
from pocketdiff.geometry.current_state import build_current_chi_state
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.inference import PocketInputs
from pocketdiff.models import PocketDiffModel
from pocketdiff.training import NextXYZBatch, next_xyz_objective, train_next_xyz_step


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def make_batch(value):
    """Use independent fractional rigid/chi primitives for a k=0 target.

    This is the 1/20 step towards the named rigid+chi reconstruction, not a
    direct holo endpoint target. Ambiguous residues are excluded from this
    narrow smoke's supervision until equivalent-target handling is available.
    """
    apo, holo = value.protein_pos_apo, value.protein_pos_holo
    ids, names = value.atom_to_residue, value.protein_atom_name
    residues = [AA_NAMES[i] for i in value.residue_type.tolist()]
    frames = build_residue_frames(apo, ids, names, num_residues=value.num_residues)
    target_frames = build_residue_frames(holo, ids, names, num_residues=value.num_residues)
    valid = frames.valid & target_frames.valid
    remaining = remaining_transform_current_to_holo(
        frames.origins, frames.frames, target_frames.origins, target_frames.frames, frame_valid=valid)
    current = build_current_chi_state(apo, names, ids, residues)
    target = build_current_chi_state(holo, names, ids, residues)
    chi_valid = current.geometry_rotatable_mask & target.geometry_rotatable_mask & valid[:, None]
    delta = periodic_chi_delta(current.angles, target.angles, chi_valid)
    rigid = apply_fractional_update(apo, ids, frames.origins, frames.frames,
                                    remaining.translation_local, remaining.rotvec_local,
                                    remaining_steps=20, frame_valid=valid)
    next_target = apply_chi_updates(rigid, current.axis_start, current.axis_end,
                                    current.downstream_atom_mask, delta / 20, valid=chi_valid).positions
    # Also exclude residues with an unusable target torsion that is movable at
    # inference; a missing target angle must not become a zero-angle label.
    unsupported = (current.geometry_rotatable_mask & ~target.geometry_rotatable_mask).any(-1)
    ambiguous = current.ambiguous_chi_mask.any(-1)
    supervision = valid & ~ambiguous & ~unsupported
    state = PocketInputs(
        protein_pos=apo.clone(), apo_pos_ref=apo.clone(), protein_feature=value.protein_feature,
        atom_to_residue_global=ids, residue_type=value.residue_type,
        batch_protein=torch.zeros(value.num_protein_atoms, dtype=torch.long),
        batch_residue=torch.zeros(value.num_residues, dtype=torch.long),
        ligand_pos=value.ligand_pos_ref, ligand_v=value.ligand_type_ref,
        batch_ligand=torch.zeros(value.num_ligand_atoms, dtype=torch.long),
        targetdiff_t=torch.tensor([199]), pocket_k=torch.tensor([0]), protein_atom_name=names,
    )
    details = dict(supervised_residues=int(supervision.sum()),
                   ambiguous_residues_excluded=int((valid & ambiguous).sum()),
                   unsupported_chi_residues_excluded=int((valid & ~ambiguous & unsupported).sum()),
                   supervised_chi=int((chi_valid & supervision[:, None]).sum()),
                   target_max_displacement=float((next_target - apo).norm(dim=-1).max()))
    return NextXYZBatch(state, next_target, supervision), details


@torch.no_grad()
def evaluate(model, batch):
    was_training = model.training
    model.eval()
    try:
        return float(next_xyz_objective(model, batch).coordinate.loss)
    finally:
        model.train(was_training)


def same_tree(first, second):
    if isinstance(first, torch.Tensor):
        return torch.equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(same_tree(first[k], second[k]) for k in first)
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(same_tree(a, b) for a, b in zip(first, second))
    return first == second


def audit_backend(backend, batch, output_dir):
    torch.manual_seed(35)
    config = dict(encoder_backend=backend, predict_chi=True, chi_input_mode='current', dropout=0.)
    if backend == 'scalar':
        config.update(encoder_layers=1, knn=8)
    model = PocketDiffModel(**config).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    encoder_before = {key: v.clone() for key, v in model.encoder.state_dict().items()}
    before = {key: v.clone() for key, v in vars(batch.inputs).items() if isinstance(v, torch.Tensor)}
    names_before = list(batch.inputs.protein_atom_name)
    target_before = batch.target_pos_next.clone()
    mask_before = batch.supervision_frame_valid.clone()
    initial = evaluate(model, batch)
    records, losses = [], [initial]
    for _ in range(2):
        records.append(asdict(train_next_xyz_step(model, batch, optimizer)))
        losses.append(evaluate(model, batch))
    checkpoint = output_dir / (backend + '_step2.pt')
    torch.save(dict(format='pocketdiff-coordinate-smoke-v1', model_config=config,
                    input_contract=model.input_contract, model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(), steps=2, seed=35,
                    sample_id='3txj__1__1.A__1.K', loss='coordinate-mse-meanxyz-v1',
                    torch_rng_state=torch.get_rng_state()), checkpoint)
    payload = torch.load(checkpoint, map_location='cpu')
    restored = PocketDiffModel(**payload['model_config']).eval()
    restored.load_state_dict(payload['model_state_dict'], strict=True)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4)
    restored_optimizer.load_state_dict(payload['optimizer_state_dict'])
    with torch.no_grad():
        out = next_xyz_objective(model, batch).step_output.protein_pos_next
        reloaded = next_xyz_objective(restored, batch).step_output.protein_pos_next
    reload_error = float((out - reloaded).abs().max())
    # Use the same RNG even though this smoke disables dropout.
    torch.set_rng_state(payload['torch_rng_state'])
    records.append(asdict(train_next_xyz_step(model, batch, optimizer)))
    losses.append(evaluate(model, batch))
    torch.set_rng_state(payload['torch_rng_state'])
    resumed_record = asdict(train_next_xyz_step(restored, batch, restored_optimizer))
    resume_exact = (same_tree(model.state_dict(), restored.state_dict()) and
                    same_tree(optimizer.state_dict(), restored_optimizer.state_dict()) and
                    records[-1] == resumed_record)
    encoder_change = max(float((v - encoder_before[key]).abs().max())
                         for key, v in model.encoder.state_dict().items())
    guards = dict(
        loss_decreased=losses[-1] < losses[0],
        all_losses_finite=all(torch.isfinite(torch.tensor(losses))),
        head_gradients_nonzero=all(records[0]['gradient_norms'][key] > 0
                                   for key in ('translation', 'rotation', 'chi')),
        initial_encoder_gradient_zero=records[0]['gradient_norms']['encoder'] == 0.,
        later_encoder_gradient_nonzero=all(r['gradient_norms']['encoder'] > 0 for r in records[1:]),
        encoder_parameters_changed=encoder_change > 0.,
        inputs_unchanged=all(torch.equal(getattr(batch.inputs, key), v) for key, v in before.items())
                         and list(batch.inputs.protein_atom_name) == names_before,
        labels_unchanged=torch.equal(batch.target_pos_next, target_before)
                         and torch.equal(batch.supervision_frame_valid, mask_before),
        reload_exact=reload_error == 0., resume_step_exact=resume_exact,
        mode_preserved=not model.training and not restored.training,
    )
    torch.save(dict(model_config=config, model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(), steps=3,
                    scope='single-example training smoke, not validated model'),
               output_dir / (backend + '_step3.pt'))
    return dict(backend=backend, config=config, losses=losses, records=records,
                coordinate_rmsd_to_next_target=[(3 * loss) ** .5 for loss in losses],
                reload_max_error=reload_error, encoder_max_parameter_change=encoder_change,
                guards=guards, passed=all(guards.values()))


def main():
    root = Path('.codex-tasks/pocketdiff-development')
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=root / 'phase6b-clean-generalization/raw/manifest.json')
    parser.add_argument('--output-dir', type=Path, default=root / 'phase35-coordinate-training/raw/run')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output directory must be empty; preserve old experiments')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    entry = next(e for e in json.loads(args.manifest.read_text())['entries']
                 if e['sample_id'].startswith('3txj__'))
    source = entry['source']
    hashes = {key: sha(info['path']) for key, info in source.items()}
    if any(hashes[key] != info['sha256'] for key, info in source.items()):
        raise ValueError('source changed')
    value, _ = Apo2MolAdapter('.').convert_paths(source['holo_pocket']['path'],
                                                source['apo_pocket']['path'], source['ligand']['path'],
                                                sample_id=entry['sample_id'])
    batch, target_details = make_batch(value)
    results = []
    for backend in ('scalar', 'targetdiff'):
        row = audit_backend(backend, batch, args.output_dir)
        results.append(row)
        (args.output_dir / (backend + '_report.json')).write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
        print(json.dumps(dict(backend=backend, passed=row['passed'], losses=row['losses'])), flush=True)
    report = dict(passed=all(row['passed'] for row in results), learning_goal_met=False,
                   sample_id=value.sample_id, source_sha256=hashes, manifest_sha256=sha(args.manifest),
                   seed=35, steps=3, learning_rate=1e-4, max_grad_norm=1.,
                   target='fractional rigid+chi oracle at 1/20 from apo; ambiguous residues loss-masked',
                   target_details=target_details, results=results,
                   scope='training-path smoke only; not autonomous apo-to-holo or held-out generalization')
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (args.output_dir / 'code_fingerprints.json').write_text(json.dumps(
        {str(p): sha(p) for p in sorted(Path('pocketdiff').rglob('*.py'))}, indent=2) + '\n')
    if not report['passed']:
        raise RuntimeError('coordinate training smoke failed; inspect preserved reports')


if __name__ == '__main__':
    main()
