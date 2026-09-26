"""Read-only, paired-k diagnosis of the Phase36 joint-coordinate model."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import traceback

import torch

from pocketdiff.data.apo2mol_adapter import AA_NAMES, Apo2MolAdapter
from pocketdiff.geometry import build_residue_frames, remaining_transform_current_to_holo, periodic_chi_delta
from pocketdiff.geometry.current_state import build_current_chi_state
from pocketdiff.geometry.joint_update import apply_joint_update
from pocketdiff.models import PocketDiffModel
from pocketdiff.training import build_joint_reference, next_xyz_objective
from pocketdiff.training.coordinate import masked_next_xyz_loss
from pocketdiff.scripts.phase36_multik_joint_train import (
    apo_inputs, data_fingerprint, sha, tensor_fingerprint, write_json,
)


def rms(value, mask):
    selected = value.detach()[mask]
    return float(selected.square().mean().sqrt()) if selected.numel() else None


def cosine(a, b):
    a, b = a.detach().flatten(), b.detach().flatten()
    denominator = a.norm() * b.norm()
    return float((a @ b) / denominator) if float(denominator) > 1e-20 else None


def diagnose_batch(model, batch, endpoint):
    """No optimizer/backward accumulation; preserve mode, RNG and existing grads.

    endpoint is label-side oracle geometry. The counterfactual scales already
    predicted remaining outputs by n/20, holding input and weights fixed. It is
    a sensitivity probe, NOT an evaluated replacement inference policy.
    """
    was_training = model.training
    model.eval()
    try:
        objective = next_xyz_objective(model, batch)
        out, state = objective.step_output, batch.inputs
        pred = out.prediction
        ids, names = state.atom_to_residue_global, state.protein_atom_name
        residues = [AA_NAMES[i] for i in state.residue_type.tolist()]
        valid = batch.supervision_frame_valid & out.diagnostics['frame_valid']
        atom_mask = valid[ids]
        with torch.no_grad():
            frames = build_residue_frames(state.protein_pos, ids, names, num_residues=len(residues))
            target_frames = build_residue_frames(endpoint, ids, names, num_residues=len(residues))
            remaining = remaining_transform_current_to_holo(
                frames.origins, frames.frames, target_frames.origins, target_frames.frames, frame_valid=valid)
            current_chi = build_current_chi_state(state.protein_pos, names, ids, residues)
            target_chi = build_current_chi_state(endpoint, names, ids, residues)
            chi_mask = valid[:, None] & current_chi.geometry_rotatable_mask & target_chi.geometry_rotatable_mask
            target_angles = periodic_chi_delta(current_chi.angles, target_chi.angles, chi_mask)
        outputs = [pred.remaining_translation_local, pred.remaining_rotvec_local, pred.remaining_chi]
        targets = [remaining.translation_local, remaining.rotvec_local, target_angles]
        masks = [valid, valid, chi_mask]
        named_parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        gradients = torch.autograd.grad(objective.coordinate.loss,
                                       outputs + [p for _, p in named_parameters], allow_unused=True)
        if any(g is not None and not torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError('nonfinite diagnostic gradient')
        steps = (20 - state.pocket_k)[state.batch_residue, None].float()
        motion = {}
        for key, prediction, target, mask, grad in zip(('translation', 'rotation', 'chi'), outputs, targets, masks, gradients[:3]):
            motion[key] = dict(target_remaining_rms=rms(target, mask), predicted_remaining_rms=rms(prediction, mask),
                               target_step_rms=rms(target / steps, mask), predicted_step_rms=rms(prediction / steps, mask),
                               error_remaining_rms=rms(prediction - target, mask),
                               direction_cosine=cosine(prediction[mask], target[mask]),
                               output_gradient_norm=float(grad.norm()))
        groups = dict(encoder=0., translation_final=0., rotation_final=0., chi=0., other=0.)
        final_grads = []
        for (name, _), grad in zip(named_parameters, gradients[3:]):
            if grad is None:
                continue
            if name.startswith('motion_head.network.7.'):
                groups['translation_final'] += float(grad[:3].square().sum())
                groups['rotation_final'] += float(grad[3:].square().sum())
                final_grads.append(grad.flatten())
            else:
                group = 'encoder' if name.startswith('encoder.') else 'chi' if name.startswith('current_chi_head.') else 'other'
                groups[group] += float(grad.square().sum())
                if name.startswith('current_chi_head.network.7.'):
                    final_grads.append(grad.flatten())
        with torch.no_grad():
            scaled = replace(pred, remaining_translation_local=outputs[0] * steps / 20.,
                             remaining_rotvec_local=outputs[1] * steps / 20., remaining_chi=outputs[2] * steps / 20.)
            alternative = apply_joint_update(state.protein_pos, state.apo_pos_ref, ids, names, residues,
                                             scaled, remaining_steps=steps[:, 0].long())
            alternative_loss = masked_next_xyz_loss(alternative.protein_pos_next, batch.target_pos_next,
                                                    ids, valid, state.batch_protein).loss
            displacement = out.protein_pos_next - state.protein_pos
            target_displacement = batch.target_pos_next - state.protein_pos
            zero_loss = masked_next_xyz_loss(state.protein_pos, batch.target_pos_next, ids, valid, state.batch_protein).loss
        row = dict(motion=motion, loss=float(objective.coordinate.loss.detach()), zero_loss=float(zero_loss),
                   scaled_output_loss=float(alternative_loss),
                   target_displacement_rms=float(target_displacement[atom_mask].square().sum(-1).mean().sqrt()),
                   predicted_displacement_rms=float(displacement.detach()[atom_mask].square().sum(-1).mean().sqrt()),
                   displacement_cosine=cosine(displacement[atom_mask], target_displacement[atom_mask]),
                   gradient_norm=sum(groups.values()) ** .5,
                   parameter_gradient_norms={k: v ** .5 for k, v in groups.items()})
        return row, torch.cat(final_grads).detach()
    finally:
        model.train(was_training)


def run(args):
    torch.set_num_threads(1)
    history = {str(p): sha(p) for p in args.prior_run.iterdir() if p.is_file()}
    code = {str(p): sha(p) for p in Path('pocketdiff').rglob('*.py')}
    payload = torch.load(args.prior_run / 'checkpoint_0080.pt', map_location='cpu')
    model = PocketDiffModel(**payload['model_config']).eval()
    model.load_state_dict(payload['model_state_dict'], strict=True)
    before = tensor_fingerprint(model.state_dict())
    rng = torch.get_rng_state().clone()
    prep = payload['preparation']
    previous = json.loads((args.prior_run / 'evaluation_0080.json').read_text())
    expected = {r['sample_id']: r['teacher_forced_loss_by_k'] for r in previous['per_sample']}
    adapter = Apo2MolAdapter('.')
    records, unchanged, errors = [], [], []
    with (args.output_dir / 'per_sample_k.jsonl').open('w') as log:
        for role, sample_ids in payload['roles'].items():
            for sid in sample_ids:
                source = prep['source'][sid]
                if any(sha(info['path']) != info['sha256'] for info in source.values()):
                    raise ValueError('raw source changed: ' + sid)
                value, _ = adapter.convert_paths(source['holo_pocket']['path'], source['apo_pocket']['path'],
                                                  source['ligand']['path'], sample_id=sid)
                ref = build_joint_reference(apo_inputs(value), value.protein_pos_holo)
                fingerprint = data_fingerprint(value, ref)
                if fingerprint != prep['input_fingerprints'][sid]:
                    raise ValueError('input fingerprint differs from Phase36: ' + sid)
                first_gradient = None
                for k in range(20):
                    row, gradient = diagnose_batch(model, ref.batch_at(torch.tensor([k])), ref.positions[-1])
                    if first_gradient is None:
                        first_gradient = gradient
                    row.update(sample_id=sid, role=role, k=k, final_head_gradient_cosine_to_k0=cosine(gradient, first_gradient))
                    errors.append(abs(row['loss'] - expected[sid][k]))
                    records.append(row)
                    log.write(json.dumps(row, allow_nan=False) + '\n')
                    log.flush()
                unchanged.append(data_fingerprint(value, ref) == fingerprint)
                print(json.dumps(dict(sample_id=sid, completed_rows=len(records))), flush=True)
    summary = {}
    for role in payload['roles']:
        summary[role] = {}
        for k in range(20):
            rows = [r for r in records if r['role'] == role and r['k'] == k]
            summary[role][str(k)] = {key: sum(r[key] for r in rows) / len(rows) for key in (
                'loss', 'zero_loss', 'scaled_output_loss', 'target_displacement_rms', 'predicted_displacement_rms',
                'gradient_norm', 'displacement_cosine', 'final_head_gradient_cosine_to_k0')}
            summary[role][str(k)]['motion'] = {
                head: {metric: sum(r['motion'][head][metric] for r in rows) / len(rows)
                       for metric in rows[0]['motion'][head]} for head in ('translation', 'rotation', 'chi')}
    training = [json.loads(line) for line in (args.prior_run / 'training.jsonl').read_text().splitlines()]
    total = sum(r['gradient_norm_before_clip'] for r in training)
    training_summary = dict(
        late_k15_19_raw_norm_sum_fraction=sum(r['gradient_norm_before_clip'] for r in training if r['k'] >= 15) / total,
        k19_raw_norm_sum_fraction=sum(r['gradient_norm_before_clip'] for r in training if r['k'] == 19) / total,
        clipped_steps=[r for r in training if r['gradient_norm_before_clip'] > 1.],
        limitation='Norm concentration is NOT the causal contribution to Adam parameter updates.')
    guards = dict(rows_complete=len(records) == 160, parameters_unchanged=tensor_fingerprint(model.state_dict()) == before,
                  rng_unchanged=torch.equal(rng, torch.get_rng_state()), no_parameter_grad_accumulation=all(p.grad is None for p in model.parameters()),
                  inputs_unchanged=all(unchanged), historical_files_unchanged=all(sha(p) == h for p, h in history.items()),
                  source_unchanged=all(sha(p) == h for p, h in code.items()),
                  official_source_unchanged=sha('targetdiff-main/targetdiff-main/models/uni_transformer.py') == prep['official_encoder_sha256'],
                  phase36_loss_reproduced=max(errors) < 1e-9)
    write_json(args.output_dir / 'report.json', dict(passed=all(guards.values()), guards=guards,
               max_loss_reproduction_error=max(errors), summary=summary, training=training_summary,
               historical_sha256=history, source_sha256=code, model_config=payload['model_config'],
               learning_goal_met=False, scope='fixed checkpoint read-only diagnosis; no training'))
    if not all(guards.values()):
        raise RuntimeError('diagnosis guards failed')


def main():
    root = Path('.codex-tasks/pocketdiff-development')
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior-run', type=Path, default=root / 'phase36-multik-joint-learning/raw/run')
    parser.add_argument('--output-dir', type=Path, default=root / 'phase37-fixed-checkpoint-diagnosis/raw/run')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output directory must be empty; preserve evidence')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        run(args)
    except Exception as exc:
        write_json(args.output_dir / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
