"""Real 3txj smoke for both PocketDiff encoder backends; no training."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.models import PocketDiffModel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('.codex-tasks/pocketdiff-development')
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path,
                        default=root / 'phase6b-clean-generalization/raw/manifest.json')
    parser.add_argument('--output-dir', type=Path,
                        default=root / 'phase34-encoder-backend/raw/3txj')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output directory must be empty; preserve prior results')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(34)
    manifest = json.loads(args.manifest.read_text())
    entry = next(e for e in manifest['entries'] if e['sample_id'].startswith('3txj__'))
    source = entry['source']
    source_sha = {key: sha(info['path']) for key, info in source.items()}
    if any(source_sha[key] != info['sha256'] for key, info in source.items()):
        raise RuntimeError('source changed: ' + entry['sample_id'])
    value, _ = Apo2MolAdapter('.').convert_paths(
        source['holo_pocket']['path'], source['apo_pocket']['path'], source['ligand']['path'],
        sample_id=entry['sample_id'])
    kwargs = dict(
        protein_pos=value.protein_pos_apo.clone(), apo_pos_ref=value.protein_pos_apo.clone(),
        protein_feature=value.protein_feature, atom_to_residue_global=value.atom_to_residue,
        residue_type=value.residue_type, frame_valid=value.frame_valid,
        batch_protein=torch.zeros(value.num_protein_atoms, dtype=torch.long),
        batch_residue=torch.zeros(value.num_residues, dtype=torch.long),
        ligand_pos=value.ligand_pos_ref, ligand_v=value.ligand_type_ref,
        batch_ligand=torch.zeros(value.num_ligand_atoms, dtype=torch.long),
        targetdiff_t=torch.tensor([199], dtype=torch.long),
        pocket_k=torch.tensor([0], dtype=torch.long),
        protein_atom_name=value.protein_atom_name,
    )
    input_before = {key: val.clone() for key, val in kwargs.items()
                    if isinstance(val, torch.Tensor)}
    names_before = list(kwargs['protein_atom_name'])
    rows = []
    for backend in ('scalar', 'targetdiff'):
        config = dict(encoder_backend=backend, dropout=0., predict_chi=True,
                      chi_input_mode='current')
        if backend == 'scalar':
            config.update(encoder_layers=1, knn=8)
        model = PocketDiffModel(**config).eval()
        # A nonzero diagnostic probe makes reload and feature-dependence
        # checks informative; these parameters are not trained weights.
        with torch.no_grad():
            for head in (model.motion_head, model.current_chi_head):
                head.network[-1].weight.normal_(0., .003)
                head.network[-1].bias.fill_(.01)
        with torch.no_grad():
            out = model(**kwargs)
        se3_hidden_error = None
        if backend == 'targetdiff':
            # The read-only TargetDiff backend must preserve hidden features
            # under a shared rigid transform, independently of PocketDiff's
            # residue-frame descriptor.
            angle = torch.tensor(0.37)
            rotation = torch.tensor([
                [torch.cos(angle), -torch.sin(angle), 0.],
                [torch.sin(angle), torch.cos(angle), 0.],
                [0., 0., 1.],
            ])
            shift = torch.tensor([2.5, -1.25, .75])
            with torch.no_grad():
                base = model.encoder(
                    kwargs['protein_pos'], kwargs['protein_feature'], kwargs['batch_protein'],
                    kwargs['ligand_pos'], kwargs['ligand_v'], kwargs['batch_ligand'])
                transformed = model.encoder(
                    kwargs['protein_pos'] @ rotation.T + shift,
                    kwargs['protein_feature'], kwargs['batch_protein'],
                    kwargs['ligand_pos'] @ rotation.T + shift,
                    kwargs['ligand_v'], kwargs['batch_ligand'])
            se3_hidden_error = float((base['protein_hidden'] - transformed['protein_hidden']).abs().max())
        checkpoint = args.output_dir / (backend + '_untrained_probe.pt')
        torch.save(dict(model_config=config, input_contract=model.input_contract,
                        model_state_dict=model.state_dict(), trained=False), checkpoint)
        saved = torch.load(checkpoint, map_location='cpu')
        restored = PocketDiffModel(**saved['model_config']).eval()
        restored.load_state_dict(saved['model_state_dict'], strict=True)
        with torch.no_grad():
            reloaded = restored(**kwargs)
        fields = ('remaining_translation_local', 'remaining_rotvec_local', 'remaining_chi')
        with torch.no_grad():
            clean = model.forward_complex(value, pocket_k=0, targetdiff_t=199)
            poisoned_value = replace(value, protein_pos_holo=value.protein_pos_holo + 100.,
                                     chi_apo=value.chi_apo + 1., chi_holo=value.chi_holo - 1.,
                                     chi_mask=~value.chi_mask, frame_valid=~value.frame_valid)
            poisoned = model.forward_complex(poisoned_value, pocket_k=0, targetdiff_t=199)
        no_label_leak = all(torch.equal(getattr(clean, key), getattr(poisoned, key))
                            for key in fields + ('frame_valid',))
        input_unchanged = (all(torch.equal(kwargs[key], old) for key, old in input_before.items())
                           and kwargs['protein_atom_name'] == names_before)
        expected_shapes = [(value.num_residues, 3), (value.num_residues, 3), (value.num_residues, 5)]
        shape_ok = all(tuple(getattr(out, key).shape) == shape
                       for key, shape in zip(fields, expected_shapes))
        nonzero = all(bool(getattr(out, key).abs().max() > 0) for key in fields)
        reload_error = max(float((getattr(out, key) - getattr(reloaded, key)).abs().max())
                           for key in fields)
        finite = all(bool(torch.isfinite(getattr(out, key)).all()) for key in fields)
        row = dict(
            backend=backend,
            input_contract=model.input_contract,
            prediction_shapes={key: list(getattr(out, key).shape) for key in fields},
            finite_predictions=finite,
            shapes_valid=shape_ok,
            nonzero_probe_outputs=nonzero,
            holo_and_legacy_labels_independent=no_label_leak,
            strict_reload=True,
            reload_max_error=reload_error,
            se3_hidden_max_error=se3_hidden_error,
            coordinate_input_unchanged=torch.equal(kwargs['protein_pos'], input_before['protein_pos']),
            input_unchanged=input_unchanged,
            current_chi=True,
            passed=bool(finite and shape_ok and nonzero and no_label_leak and reload_error == 0. and
                        (se3_hidden_error is None or se3_hidden_error < 2e-5) and
                        torch.equal(kwargs['protein_pos'], input_before['protein_pos']) and
                        input_unchanged),
        )
        rows.append(row)
    report = dict(
        passed=all(row['passed'] for row in rows), learning_goal_met=False,
        sample_id=value.sample_id, protein_atoms=value.num_protein_atoms,
        residues=value.num_residues, ligand_atoms=value.num_ligand_atoms,
        source_sha256=source_sha, manifest_sha256=sha(args.manifest),
        seed=34, trained=False,
        backends=rows,
        scope='backend forward/reload only; no optimizer, bridge, fine-tuning or cache migration',
    )
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output_dir / 'code_fingerprints.json').write_text(json.dumps(
        {str(path): sha(path) for path in sorted(Path('pocketdiff').rglob('*.py'))},
        indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise RuntimeError('Phase 34 backend smoke failed')


if __name__ == '__main__':
    main()
