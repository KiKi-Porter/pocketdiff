"""Real current-state -> joint heads -> coordinates audit; no optimizer."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from pocketdiff.data.apo2mol_adapter import AA_NAMES, Apo2MolAdapter
from pocketdiff.data.schema import PocketDiffPrediction
from pocketdiff.geometry import build_residue_frames, periodic_chi_delta, remaining_transform_current_to_holo
from pocketdiff.geometry.current_state import build_current_chi_state
from pocketdiff.geometry.joint_update import apply_joint_update
from pocketdiff.geometry.oracle import oracle_rigid_chi_reconstruction
from pocketdiff.inference import PocketInputs, predict_joint_step
from pocketdiff.models import PocketDiffModel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_inputs(value, current, k):
    return PocketInputs(
        protein_pos=current, apo_pos_ref=value.protein_pos_apo,
        protein_feature=value.protein_feature, atom_to_residue_global=value.atom_to_residue,
        residue_type=value.residue_type,
        batch_protein=torch.zeros(value.num_protein_atoms, dtype=torch.long),
        batch_residue=torch.zeros(value.num_residues, dtype=torch.long),
        ligand_pos=value.ligand_pos_ref, ligand_v=value.ligand_type_ref,
        batch_ligand=torch.zeros(value.num_ligand_atoms, dtype=torch.long),
        pocket_k=torch.tensor([k]), targetdiff_t=torch.tensor([199 - 10 * k]),
        protein_atom_name=value.protein_atom_name,
    )


def geometry_args(value):
    return (value.atom_to_residue, value.protein_atom_name,
            [AA_NAMES[i] for i in value.residue_type.tolist()])


def audit(value, zero_model, probe_model, restored, index):
    apo, holo = value.protein_pos_apo, value.protein_pos_holo
    ids, names, residues = geometry_args(value)
    before = {key: v.clone() for key, v in vars(value).items() if isinstance(v, torch.Tensor)}
    frames = build_residue_frames(apo, ids, names, num_residues=value.num_residues)
    # Construct a non-apo current state without inspecting holo or its masks.
    imposed = PocketDiffPrediction(
        torch.full((value.num_residues, 3), .03),
        torch.full((value.num_residues, 3), .01),
        torch.full((value.num_residues, 5), .1), frames.valid, {},
    )
    with torch.no_grad():
        current = apply_joint_update(apo, apo, ids, names, residues, imposed, remaining_steps=1).protein_pos_next
    k = (0, 7, 19)[index % 3]
    state = make_inputs(value, current, k)
    zero_model.zero_grad(set_to_none=True)
    out = predict_joint_step(zero_model, state)
    loss = (out.protein_pos_next - holo).square().mean()
    loss.backward()
    rigid_grad = zero_model.motion_head.network[-1].weight.grad
    chi_grad = zero_model.current_chi_head.network[-1].weight.grad
    gradients = dict(translation=float(rigid_grad[:3].norm()), rotation=float(rigid_grad[3:].norm()),
                     chi=float(chi_grad.norm()))
    gradient_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in zero_model.parameters())
    with torch.no_grad():
        nonzero = predict_joint_step(probe_model, state)
        reloaded = predict_joint_step(restored, state)
        reload_error = float((nonzero.protein_pos_next - reloaded.protein_pos_next).abs().max())
        # Holo and all legacy label-derived fields are deliberately changed.
        # The public convenience current-state path must ignore them as well.
        poisoned = replace(value, protein_pos_holo=holo + 100., chi_apo=value.chi_apo + 1.,
                           chi_holo=value.chi_holo - 1., chi_mask=~value.chi_mask,
                           frame_valid=~value.frame_valid)
        a = probe_model.forward_complex(value, protein_pos=current, pocket_k=k, targetdiff_t=199 - 10*k)
        b = probe_model.forward_complex(poisoned, protein_pos=current, pocket_k=k, targetdiff_t=199 - 10*k)
        independent_match = all(torch.equal(getattr(a, key), getattr(nonzero.prediction, key))
                                for key in ('remaining_translation_local', 'remaining_rotvec_local', 'remaining_chi'))
        no_label_leak = all(torch.equal(getattr(a, key), getattr(b, key))
                            for key in ('remaining_translation_local', 'remaining_rotvec_local', 'remaining_chi', 'frame_valid'))
        mask = nonzero.diagnostics['chi_rotatable_mask']
        observed = periodic_chi_delta(nonzero.diagnostics['current_chi'], nonzero.diagnostics['chi_next'], mask)
        angle_error = float((observed - nonzero.applied_chi)[mask].abs().max()) if mask.any() else 0.
        # Independent endpoint geometry check against Phase31 oracle semantics.
        target_frames = build_residue_frames(holo, ids, names, num_residues=value.num_residues)
        valid = frames.valid & target_frames.valid
        rem = remaining_transform_current_to_holo(frames.origins, frames.frames,
                                                  target_frames.origins, target_frames.frames, frame_valid=valid)
        c = build_current_chi_state(apo, names, ids, residues)
        h = build_current_chi_state(holo, names, ids, residues)
        delta = periodic_chi_delta(c.angles, h.angles, c.geometry_rotatable_mask & h.geometry_rotatable_mask)
        oracle_pred = PocketDiffPrediction(rem.translation_local, rem.rotvec_local, delta, valid, {})
        joint = apply_joint_update(apo, apo, ids, names, residues, oracle_pred, remaining_steps=1)
        oracle = oracle_rigid_chi_reconstruction(apo, holo, ids, names, residues, valid)
        oracle_error = float((joint.protein_pos_next - oracle.rigid_chi_positions).abs().max())
        # Coordinate evaluation of probe output is intentionally not a learning metric.
        displacement = float((nonzero.protein_pos_next - current).norm(dim=-1).max())
    guards = dict(
        zero_exact=torch.equal(out.protein_pos_next, current),
        finite_coordinates=bool(torch.isfinite(nonzero.protein_pos_next).all()),
        finite_parameter_gradients=gradient_finite,
        three_head_gradients=all(v > 0 for v in gradients.values()),
        input_unchanged=all(torch.equal(getattr(value, key), v) for key, v in before.items()),
        holo_and_legacy_mask_independent=no_label_leak,
        convenience_matches_independent=independent_match,
        reload_exact=reload_error == 0., chi_readback=angle_error < 2e-4,
        oracle_matches_phase31=oracle_error < 2e-5,
        next_frames_preserved=bool(nonzero.diagnostics['frame_valid_next'][nonzero.diagnostics['frame_valid']].all()),
        next_chi_preserved=bool(nonzero.diagnostics['chi_rotatable_mask_next'][mask].all()),
        nonzero_probe_moves=displacement > 0.,
    )
    row = dict(sample_id=value.sample_id, pocket_k=k, atoms=len(apo), residues=value.num_residues,
               active_chi=int(mask.sum()), initial_coordinate_loss=float(loss), head_gradient_norms=gradients,
               max_probe_displacement=displacement, max_chi_error_rad=angle_error,
               oracle_max_coordinate_error=oracle_error, reload_error=reload_error,
               guards=guards, passed=all(guards.values()))
    trace = dict(sample_id=value.sample_id, current=current, apo=apo,
                 probe_next=nonzero.protein_pos_next, applied_chi=nonzero.applied_chi,
                 next_chi=nonzero.diagnostics['chi_next'])
    return row, trace


def main():
    parser = argparse.ArgumentParser()
    root = Path('.codex-tasks/pocketdiff-development')
    parser.add_argument('--manifest', type=Path, default=root / 'phase6b-clean-generalization/raw/manifest.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=16)
    args = parser.parse_args()
    entries = json.loads(args.manifest.read_text())['entries']
    if not 1 <= args.limit <= len(entries):
        parser.error('limit must be within manifest entries')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output directory must be empty')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(32)
    config = dict(encoder_layers=1, knn=8, predict_chi=True, chi_input_mode='current', dropout=0.)
    zero_model = PocketDiffModel(**config).eval()
    probe = PocketDiffModel(**config).eval()
    with torch.no_grad():
        for head in (probe.motion_head, probe.current_chi_head):
            head.network[-1].weight.normal_(0., .003)
            head.network[-1].bias.fill_(.01)
    before = {name: value.clone() for name, value in zero_model.state_dict().items()}
    checkpoint = args.output_dir / 'untrained_probe.pt'
    torch.save(dict(format='pocketdiff-current-joint-probe-v1', input_contract=probe.input_contract,
                    model_config=config, model_state_dict=probe.state_dict(), trained=False), checkpoint)
    payload = torch.load(checkpoint, map_location='cpu')
    restored = PocketDiffModel(**payload['model_config']).eval()
    restored.load_state_dict(payload['model_state_dict'], strict=True)
    rows, traces = [], []
    adapter = Apo2MolAdapter('.')
    for i, entry in enumerate(entries[:args.limit]):
        source = entry['source']
        hashes = {k: sha(info['path']) for k, info in source.items()}
        if any(hashes[k] != info['sha256'] for k, info in source.items()):
            raise ValueError('source changed: ' + entry['sample_id'])
        value, _ = adapter.convert_paths(source['holo_pocket']['path'], source['apo_pocket']['path'],
                                        source['ligand']['path'], sample_id=entry['sample_id'])
        row, trace = audit(value, zero_model, probe, restored, i)
        row['source_sha256'] = hashes
        rows.append(row)
        traces.append(trace)
    no_optimizer = all(torch.equal(zero_model.state_dict()[name], value) for name, value in before.items())
    summary = dict(sample_count=len(rows), passed=all(row['passed'] for row in rows) and no_optimizer,
                   parameter_values_unchanged=no_optimizer, active_chi=sum(r['active_chi'] for r in rows),
                   max_chi_error_rad=max(r['max_chi_error_rad'] for r in rows),
                   oracle_max_coordinate_error=max(r['oracle_max_coordinate_error'] for r in rows),
                   max_reload_error=max(r['reload_error'] for r in rows))
    report = dict(**summary, mode='independent-current-joint-contract', learning_goal_met=False,
                  model_config=config, input_contract=probe.input_contract, results=rows,
                  manifest_sha256=sha(args.manifest), selection='first N manifest entries; diagnostic only',
                  scope='no optimizer, bridge, fine-tuning, cache migration or learned quality claim')
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    torch.save(dict(format='pocketdiff-current-joint-diagnostic-v1', results=traces), args.output_dir / 'coordinates.pt')
    (args.output_dir / 'code_fingerprints.json').write_text(json.dumps(
        {str(p): sha(p) for p in sorted(Path('pocketdiff').rglob('*.py'))}, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    if not summary['passed']:
        raise RuntimeError('current joint contract failed; see report.json')


if __name__ == '__main__':
    main()
