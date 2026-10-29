"""Fixed-slice raw-data coordinate experiment with autonomous evaluation."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
import traceback

import torch

from pocketdiff.data.apo2mol_adapter import AA_NAMES, Apo2MolAdapter
from pocketdiff.data.schema import PocketDiffPrediction
from pocketdiff.geometry import (build_residue_frames, remaining_transform_current_to_holo,
                                periodic_chi_delta, oracle_rigid_chi_reconstruction)
from pocketdiff.geometry.current_state import build_current_chi_state
from pocketdiff.geometry.joint_update import apply_joint_update
from pocketdiff.inference import PocketInputs
from pocketdiff.models import PocketDiffModel
from pocketdiff.rollout import rollout_joint_from_apo
from pocketdiff.training import build_joint_reference, next_xyz_objective, train_next_xyz_step


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def tensor_fingerprint(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def apo_inputs(value):
    """Only read apo/ligand inventory; do not use holo or shared label masks."""
    return PocketInputs(
        protein_pos=value.protein_pos_apo.clone(), apo_pos_ref=value.protein_pos_apo.clone(),
        protein_feature=value.protein_feature, atom_to_residue_global=value.atom_to_residue,
        residue_type=value.residue_type, batch_protein=torch.zeros(value.num_protein_atoms, dtype=torch.long),
        batch_residue=torch.zeros(value.num_residues, dtype=torch.long),
        ligand_pos=value.ligand_pos_ref, ligand_v=value.ligand_type_ref,
        batch_ligand=torch.zeros(value.num_ligand_atoms, dtype=torch.long),
        pocket_k=torch.tensor([0]), targetdiff_t=torch.tensor([199]),
        protein_atom_name=value.protein_atom_name,
    )


def data_fingerprint(value, reference):
    values = dict(vars(value))
    values.update({'input_' + k: v for k, v in vars(reference.apo_inputs).items()})
    values.update(positions=reference.positions, reference_valid=reference.reference_frame_valid,
                  supervision=reference.supervision_frame_valid, ambiguous=reference.ambiguous_residues)
    return tensor_fingerprint(values)


@torch.no_grad()
def audit_reference(value, reference):
    ids, names = value.atom_to_residue, value.protein_atom_name
    residues = [AA_NAMES[i] for i in value.residue_type.tolist()]
    holo = value.protein_pos_holo
    valid = reference.reference_frame_valid
    target_frames = build_residue_frames(holo, ids, names, num_residues=value.num_residues)
    target_chi = build_current_chi_state(holo, names, ids, residues)
    max_error = 0.
    for k in range(20):
        batch = reference.batch_at(torch.tensor([k]))
        current_pos = batch.inputs.protein_pos
        frames = build_residue_frames(current_pos, ids, names, num_residues=value.num_residues)
        current_chi = build_current_chi_state(current_pos, names, ids, residues)
        rem = remaining_transform_current_to_holo(frames.origins, frames.frames,
                                                  target_frames.origins, target_frames.frames, frame_valid=valid)
        chi = periodic_chi_delta(current_chi.angles, target_chi.angles,
                                  current_chi.geometry_rotatable_mask & target_chi.geometry_rotatable_mask)
        pred = PocketDiffPrediction(rem.translation_local, rem.rotvec_local, chi, valid, {})
        out = apply_joint_update(current_pos, value.protein_pos_apo, ids, names, residues,
                                  pred, remaining_steps=20 - k)
        max_error = max(max_error, float((out.protein_pos_next - batch.target_pos_next).abs().max()))
    oracle = oracle_rigid_chi_reconstruction(value.protein_pos_apo, holo, ids, names, residues, valid)
    endpoint_error = float((reference.positions[-1] - oracle.rigid_chi_positions).abs().max())
    safe = reference.supervision_frame_valid
    inference_independent = tensor_fingerprint(vars(apo_inputs(value))) == tensor_fingerprint(vars(apo_inputs(
        replace(value, protein_pos_holo=holo + 100., chi_apo=value.chi_apo + 1.,
                chi_holo=value.chi_holo - 1., chi_mask=~value.chi_mask, frame_valid=~value.frame_valid))))
    return dict(sample_id=value.sample_id, max_oracle_step_error=max_error,
                 max_oracle_endpoint_error=endpoint_error, supervised_residues=int(safe.sum()),
                 supervised_atoms=int(safe[ids].sum()), residues=value.num_residues,
                 ambiguous_residues_excluded=int((valid & reference.ambiguous_residues).sum()),
                 apo_inputs_label_independent=inference_independent,
                 passed=bool(max_error < 3e-5 and endpoint_error < 3e-5 and inference_independent
                             and torch.equal(reference.positions[0], value.protein_pos_apo)))


def rmsd(positions, target, mask):
    return float((positions[mask] - target[mask]).square().sum(-1).mean().sqrt())


@torch.no_grad()
def evaluate(model, complexes, references, roles, step, output_dir):
    was_training = model.training
    parameter_before = tensor_fingerprint(model.state_dict())
    rng_before = torch.get_rng_state().clone()
    model.eval()
    per_sample, traces = [], {}
    try:
        for role in ('train', 'holdout'):
            for sample_id in roles[role]:
                value, reference = complexes[sample_id], references[sample_id]
                losses = [float(next_xyz_objective(model, reference.batch_at(torch.tensor([k]))).coordinate.loss)
                          for k in range(20)]
                # This call gets a fresh apo-only input object, not the label trajectory.
                trajectory = rollout_joint_from_apo(model, apo_inputs(value))
                mask = reference.reference_frame_valid[value.atom_to_residue]
                safe = reference.supervision_frame_valid[value.atom_to_residue]
                all_rmsd = [rmsd(pos, value.protein_pos_holo, mask) for pos in trajectory.positions]
                safe_rmsd = [rmsd(pos, value.protein_pos_holo, safe) for pos in trajectory.positions]
                oracle_rmsd = rmsd(reference.positions[-1], value.protein_pos_holo, mask)
                rows = dict(sample_id=sample_id, role=role, teacher_forced_loss_by_k=losses,
                            mean_teacher_forced_loss=sum(losses) / 20,
                            apo_rmsd=all_rmsd[0], final_rmsd=all_rmsd[-1],
                            improvement=all_rmsd[0] - all_rmsd[-1],
                            oracle_rmsd=oracle_rmsd, autonomous_rmsd_by_step=all_rmsd,
                            supervised_subset_rmsd_by_step=safe_rmsd,
                            max_displacement_from_apo=float((trajectory.positions - value.protein_pos_apo).norm(dim=-1).max()),
                            valid_atoms=int(mask.sum()), supervised_atoms=int(safe.sum()))
                per_sample.append(rows)
                traces[sample_id] = dict(apo=value.protein_pos_apo, holo=value.protein_pos_holo,
                                         predicted_positions=trajectory.positions,
                                         reference_endpoint=reference.positions[-1],
                                         evaluation_atom_mask=mask, supervised_atom_mask=safe)
                print(json.dumps(dict(evaluation_step=step, sample=sample_id, role=role,
                                      apo_rmsd=rows['apo_rmsd'], final_rmsd=rows['final_rmsd'])), flush=True)
    finally:
        model.train(was_training)
    if tensor_fingerprint(model.state_dict()) != parameter_before or not torch.equal(torch.get_rng_state(), rng_before):
        raise RuntimeError('evaluation mutated parameters or RNG')
    summary = {}
    for role in roles:
        rows = [r for r in per_sample if r['role'] == role]
        summary[role] = {key: sum(row[key] for row in rows) / len(rows)
                         for key in ('apo_rmsd', 'final_rmsd', 'improvement', 'oracle_rmsd', 'mean_teacher_forced_loss')}
        summary[role]['improved_samples'] = sum(row['improvement'] > 0 for row in rows)
        summary[role]['sample_count'] = len(rows)
    result = dict(step=step, summary=summary, per_sample=per_sample,
                   primary_metric='mean per-complex all-reference-valid named-atom RMSD; fixed apo frame; no Kabsch',
                   heldout_scope='development holdout from historical Phase17, not independent test')
    write_json(output_dir / ('evaluation_%04d.json' % step), result)
    torch.save(traces, output_dir / ('trajectories_%04d.pt' % step))
    return result


def save_checkpoint(output_dir, step, model, optimizer, config, roles, schedule, preparation, seed):
    torch.save(dict(format='pocketdiff-joint-multik-coordinate-v1', step=step,
                    model_config=config, input_contract=model.input_contract,
                    model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                    roles=roles, schedule=schedule, torch_rng_state=torch.get_rng_state(),
                    seed=seed, loss='coordinate-mse-meanxyz-v1', preparation=preparation),
                   output_dir / ('checkpoint_%04d.pt' % step))


def run(args):
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    output_dir = args.output_dir
    old = json.loads(args.prior_preparation.read_text())
    counts_by_role = {'train': args.train_count, 'holdout': args.holdout_count}
    roles = {key: old['roles'][key][:counts_by_role[key]] for key in ('train', 'holdout')}
    if any(counts_by_role[key] < 1 or len(old['roles'][key]) < counts_by_role[key]
           for key in ('train', 'holdout')):
        raise ValueError('requested role count exceeds the historical preparation')
    if any(len(ids) != counts_by_role[key] or len(set(ids)) != counts_by_role[key]
           for key, ids in roles.items()) or set(roles['train']) & set(roles['holdout']):
        raise ValueError('expected disjoint fixed role slices')
    total_steps = len(roles['train']) * 20
    entries = {e['sample_id']: e for e in json.loads(args.manifest.read_text())['entries']}
    adapter = Apo2MolAdapter('.')
    complexes, references, audits, fingerprints, sources = {}, {}, [], {}, {}
    for role, sample_ids in roles.items():
        for sample_id in sample_ids:
            source = entries[sample_id]['source']
            hashes = {key: sha(info['path']) for key, info in source.items()}
            if any(hashes[key] != info['sha256'] for key, info in source.items()):
                raise ValueError('raw source changed: ' + sample_id)
            value, _ = adapter.convert_paths(source['holo_pocket']['path'], source['apo_pocket']['path'],
                                             source['ligand']['path'], sample_id=sample_id)
            reference = build_joint_reference(apo_inputs(value), value.protein_pos_holo)
            row = audit_reference(value, reference)
            audits.append(row)
            write_json(output_dir / 'geometry_audit.json', dict(passed=all(r['passed'] for r in audits), results=audits))
            if not row['passed']:
                raise RuntimeError('real geometry audit failed: ' + sample_id)
            complexes[sample_id], references[sample_id] = value, reference
            sources[sample_id] = source
            fingerprints[sample_id] = data_fingerprint(value, reference)
            print(json.dumps(dict(audit=sample_id, passed=True, max_step_error=row['max_oracle_step_error'])), flush=True)
    official_path = Path('targetdiff-main/targetdiff-main/models/uni_transformer.py')
    preparation = dict(roles=roles, source=sources, input_fingerprints=fingerprints,
                        manifest_sha256=sha(args.manifest), prior_preparation_sha256=sha(args.prior_preparation),
                        official_encoder_sha256=sha(official_path), seed=args.seed, device='cpu',
                        steps=total_steps, train_count=args.train_count, holdout_count=args.holdout_count,
                        selection='first N per historical Phase17 role; no role reassignment',
                        geometry_audit_passed=True)
    write_json(output_dir / 'preparation.json', preparation)
    write_json(output_dir / 'code_fingerprints.json',
               {str(p): sha(p) for p in sorted(Path('pocketdiff').rglob('*.py'))})
    if args.audit_only:
        return
    config = dict(encoder_backend='targetdiff', predict_chi=True, chi_input_mode='current', dropout=0.,
                  motion_parameterization=args.motion_parameterization)
    model = PocketDiffModel(**config).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    pairs = [(sample_id, k) for sample_id in roles['train'] for k in range(20)]
    if len(pairs) != total_steps:
        raise RuntimeError('training schedule size does not match selected train slice')
    order = torch.randperm(total_steps, generator=torch.Generator().manual_seed(args.seed)).tolist()
    schedule = [pairs[i] for i in order]
    write_json(output_dir / 'schedule.json', schedule)
    save_checkpoint(output_dir, 0, model, optimizer, config, roles, schedule, preparation, args.seed)
    initial = evaluate(model, complexes, references, roles, 0, output_dir)
    if any(abs(row['improvement']) > 1e-8 for row in initial['per_sample']):
        raise RuntimeError('zero initialized baseline was not stationary')
    coverage = {sid: [0] * 20 for sid in roles['train']}
    with (output_dir / 'training.jsonl').open('w') as log:
        for step, (sid, k) in enumerate(schedule, start=1):
            record = train_next_xyz_step(model, references[sid].batch_at(torch.tensor([k])), optimizer)
            row = dict(step=step, sample_id=sid, k=k, **asdict(record))
            log.write(json.dumps(row, allow_nan=False) + '\n')
            log.flush()
            coverage[sid][k] += 1
            if step % 10 == 0:
                print(json.dumps(dict(step=step, k=k, sample_id=sid, loss=row['loss_before_update'])), flush=True)
            if step in {total_steps // 2, total_steps}:
                save_checkpoint(output_dir, step, model, optimizer, config, roles, schedule, preparation, args.seed)
    write_json(output_dir / 'coverage.json', coverage)
    final = evaluate(model, complexes, references, roles, total_steps, output_dir)
    final_checkpoint = output_dir / ('checkpoint_%04d.pt' % total_steps)
    payload = torch.load(final_checkpoint, map_location='cpu')
    restored = PocketDiffModel(**payload['model_config']).eval()
    restored.load_state_dict(payload['model_state_dict'], strict=True)
    reload_max_error = 0.
    with torch.no_grad():
        for sid in roles['holdout']:
            batch = references[sid].batch_at(torch.tensor([19]))
            expected = next_xyz_objective(model, batch).step_output.protein_pos_next
            actual = next_xyz_objective(restored, batch).step_output.protein_pos_next
            reload_max_error = max(reload_max_error, float((actual - expected).abs().max()))
    guards = dict(complete_k_coverage=all(counts == [1] * 20 for counts in coverage.values()),
                   inputs_unchanged=all(data_fingerprint(complexes[sid], references[sid]) == fingerprints[sid]
                                        for sid in complexes),
                   strict_reload_exact=reload_max_error == 0.,
                   official_source_unchanged=sha(official_path) == preparation['official_encoder_sha256'],
                   zero_initialization_stationary=all(row['improvement'] == 0 for row in initial['per_sample']))
    report = dict(passed=all(guards.values()), learning_goal_met=False, guards=guards,
                   learning_signal_both_roles_improve=all(v['improvement'] > 0 for v in final['summary'].values()),
                   roles=roles, model_config=config, steps=total_steps, seed=args.seed, learning_rate=1e-4,
                   train_count=args.train_count, holdout_count=args.holdout_count,
                   max_grad_norm=1., initial=initial['summary'], final=final['summary'],
                   reload_max_error=reload_max_error, elapsed_seconds=time.monotonic() - started,
                   scope='finite coordinate-only fixed role-slice experiment; no bridge or fine-tuning',
                   limitations=['teacher-forced joint reference targets preserve apo internal geometry',
                                'ambiguous residues excluded only from training loss',
                                'fixed clean reference ligand; not ligand-free apo prediction',
                                'tiny development holdout; no strict generalization claim'])
    write_json(output_dir / 'report.json', report)
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise RuntimeError('experiment engineering guards failed')


def main():
    root = Path('.codex-tasks/pocketdiff-development')
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=root / 'phase6b-clean-generalization/raw/manifest.json')
    parser.add_argument('--prior-preparation', type=Path, default=root / 'phase17-multik-clean-training/raw/run/preparation.json')
    parser.add_argument('--output-dir', type=Path, default=root / 'phase36-multik-joint-learning/raw/run')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--motion-parameterization', choices=('remaining', 'bridge_rate'), default='remaining')
    parser.add_argument('--seed', type=int, default=36)
    parser.add_argument('--train-count', type=int, default=4)
    parser.add_argument('--holdout-count', type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output directory must be empty; preserve prior experiments')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        run(args)
    except Exception as exc:
        write_json(args.output_dir / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
