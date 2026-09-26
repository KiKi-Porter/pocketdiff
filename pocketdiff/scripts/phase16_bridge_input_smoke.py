"""Real teacher-forced multi-time input and backward audit, without optimization."""

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import pickle

import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.geometry.bridge import apply_fractional_update, oracle_reconstruction_metrics
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.models import PocketDiffModel
from pocketdiff.preprocessing import GEOMETRY_VERSION
from pocketdiff.training import (
    build_bridge_batch, collate_clean_examples, make_clean_example,
    masked_remaining_motion_loss,
)


def _snapshot(batch):
    return {f.name: getattr(batch, f.name).clone() for f in fields(batch)
            if isinstance(getattr(batch, f.name), torch.Tensor)}


def _audit(model, clean, times):
    before = _snapshot(clean)
    batch = build_bridge_batch(clean, times)
    batch_before = _snapshot(batch)
    frames = build_residue_frames(batch.protein_pos, batch.atom_to_residue_global,
                                  batch.protein_atom_name, num_residues=batch.residue_type.numel())
    oracle_next = apply_fractional_update(
        batch.protein_pos, batch.atom_to_residue_global, frames.origins, frames.frames,
        batch.target_translation_local, batch.target_rotvec_local,
        remaining_steps=batch.remaining_steps[batch.batch_residue], frame_valid=batch.frame_valid,
    )
    step_error = float(torch.linalg.vector_norm(oracle_next - batch.protein_pos_next_target, dim=-1).max())
    model.zero_grad(set_to_none=True)
    prediction = model(**batch.model_kwargs())
    loss = masked_remaining_motion_loss(prediction, batch.target_translation_local,
                                        batch.target_rotvec_local, batch.frame_valid)
    loss.loss.backward()
    gradients = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    finite = bool(torch.isfinite(loss.loss) and gradients
                  and all(torch.isfinite(g).all() for g in gradients.values()))
    encoder_grad = sum(float(g.abs().sum()) for n, g in gradients.items() if n.startswith('encoder.'))
    head_grad = sum(float(g.abs().sum()) for n, g in gradients.items() if n.startswith('motion_head.'))
    preserved_names = ('apo_pos_ref', 'protein_pos_holo', 'protein_feature', 'ligand_pos',
                       'ligand_v', 'atom_to_residue_global', 'batch_protein', 'batch_residue', 'batch_ligand')
    invariant = (all(torch.equal(getattr(clean, n), v) for n, v in before.items())
                 and all(torch.equal(getattr(batch, n), v) for n, v in batch_before.items())
                 and all(torch.equal(getattr(batch, n), before[n]) for n in preserved_names))
    detached = all(not getattr(batch, n).requires_grad for n in (
        'protein_pos', 'protein_pos_next_target', 'target_translation_local', 'target_rotvec_local'))
    row = {
        'sample_ids': batch.sample_ids, 'k': batch.pocket_k.tolist(),
        't': batch.targetdiff_t.tolist(), 'remaining_steps': batch.remaining_steps.tolist(),
        'protein_shape': list(batch.protein_pos.shape), 'ligand_shape': list(batch.ligand_pos.shape),
        'translation_shape': list(prediction.remaining_translation_local.shape),
        'rotvec_shape': list(prediction.remaining_rotvec_local.shape),
        'valid_residues': int(batch.frame_valid.sum()), 'invalid_residues': int((~batch.frame_valid).sum()),
        'oracle_next_max_atom_error_angstrom': step_error,
        'remaining_loss_observation': float(loss.loss.detach()),
        'finite_loss_and_gradients': finite, 'encoder_gradient_abs_sum': encoder_grad,
        'head_gradient_abs_sum': head_grad, 'input_and_condition_invariants': invariant,
        'supervision_detached': detached,
    }
    row['passed'] = finite and encoder_grad > 0 and head_grad > 0 and invariant and detached and step_error < 1e-4
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('Apo2Mol-main/Apo2MOl-dataset/data_folder'))
    parser.add_argument('--split-pickle', type=Path, default=Path('Apo2Mol-main/Apo2MOl-dataset/split_druglike_dict.pkl'))
    parser.add_argument('--pocketdiff-checkpoint', type=Path,
                        default=Path('.codex-tasks/pocketdiff-development/phase6b-clean-generalization/raw/generalization_report.pt'))
    parser.add_argument('--seed', type=int, default=20260919)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'passed': False, 'scope': 'teacher-forced input and gradient contract only',
              'geometry_version': GEOMETRY_VERSION,
              'optimizer_steps': 0, 'seed': args.seed, 'rows': [],
              'checkpoint_training_scope': 'clean k=0/t=199; losses at other k are observations, not trained validation'}
    try:
        torch.manual_seed(args.seed)
        with args.split_pickle.open('rb') as handle:
            records = pickle.load(handle)['train']
        is_3txj = lambda r: str(r[0]).startswith('3txj') or str(r[1]).startswith('3txj')
        chosen = [next(r for r in records if is_3txj(r)), next(r for r in records if not is_3txj(r))]
        adapter = Apo2MolAdapter(args.data_root)
        examples = [make_clean_example(adapter.convert_record(r)) for r in chosen]
        payload = torch.load(args.pocketdiff_checkpoint, map_location='cpu')
        model = PocketDiffModel(**payload['model_config'])
        model.load_state_dict(payload['model_state_dict'], strict=True)
        model.eval()  # Deterministic dropout-free audit; gradients remain enabled.
        model_before = {n: v.clone() for n, v in model.state_dict().items()}
        clean = collate_clean_examples(examples[:1])
        for k in range(20):
            report['rows'].append(_audit(model, clean, torch.tensor([k])))
        mixed = collate_clean_examples(examples)
        for times in (torch.tensor([0, 19]), torch.tensor([7, 13])):
            report['rows'].append(_audit(model, mixed, times))
        floors = []
        for example in examples:
            endpoint = build_bridge_batch(collate_clean_examples([example]), torch.tensor([19]))
            floor = oracle_reconstruction_metrics(
                endpoint.protein_pos_next_target, endpoint.protein_pos_holo,
                endpoint.atom_to_residue_global, endpoint.protein_atom_name, endpoint.frame_valid,
            )
            floors.append(dict(sample_id=example.complex_value.sample_id, **floor.__dict__))
        unchanged = all(torch.equal(v, model_before[n]) for n, v in model.state_dict().items())
        report.update(
            checkpoint=str(args.pocketdiff_checkpoint),
            checkpoint_sha256=hashlib.sha256(args.pocketdiff_checkpoint.read_bytes()).hexdigest(),
            model_config=payload['model_config'], torch_version=torch.__version__,
            sample_ids=mixed.sample_ids,
            source_records=[dict(zip(('holo_pocket', 'apo_pocket', 'ligand'),
                                     (str(path) for path in r[:3]))) for r in chosen],
            model_parameters_unchanged=unchanged, oracle_rigid_endpoint_floors=floors,
            passed=unchanged and all(row['passed'] for row in report['rows']),
        )
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: report[key] for key in ('passed', 'sample_ids', 'optimizer_steps')}))
    if not report['passed']:
        raise RuntimeError('bridge input smoke failed; see saved report')


if __name__ == '__main__':
    main()
