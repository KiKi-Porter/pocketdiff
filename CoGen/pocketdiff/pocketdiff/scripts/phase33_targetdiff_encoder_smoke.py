"""Real raw 3txj TargetDiff-compatible encoder smoke; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.models import TargetDiffEncoderAdapter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=Path('.codex-tasks/pocketdiff-development/phase6b-clean-generalization/raw/manifest.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output directory must be empty')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    entry = next(e for e in manifest['entries'] if e['sample_id'].startswith('3txj__'))
    source = entry['source']
    adapter = Apo2MolAdapter('.')
    value, _ = adapter.convert_paths(source['holo_pocket']['path'], source['apo_pocket']['path'], source['ligand']['path'], sample_id=entry['sample_id'])
    model = TargetDiffEncoderAdapter().eval()
    protein_pos = value.protein_pos_apo.float()
    ligand_pos = value.ligand_pos_ref.float()
    bp = torch.zeros(protein_pos.shape[0], dtype=torch.long)
    bl = torch.zeros(ligand_pos.shape[0], dtype=torch.long)
    before = torch.cat((protein_pos, ligand_pos)).clone()
    with torch.no_grad():
        out = model(protein_pos, value.protein_feature.float(), bp, ligand_pos, value.ligand_type_ref, bl)
    after = out['coordinates']
    report = dict(
        passed=bool(torch.isfinite(out['protein_hidden']).all() and torch.isfinite(out['ligand_hidden']).all() and torch.equal(before, after)),
        sample_id=value.sample_id, protein_atoms=int(protein_pos.shape[0]), ligand_atoms=int(ligand_pos.shape[0]),
        protein_hidden_shape=list(out['protein_hidden'].shape), ligand_hidden_shape=list(out['ligand_hidden'].shape),
        coordinate_max_error=float((before-after).abs().max()), model_contract=model.contract,
        config=model.config, source_sha256={k: source[k]['sha256'] for k in source},
        targetdiff_source_sha256={str(p): sha(p) for p in sorted(Path('targetdiff-main/targetdiff-main/models').glob('uni_transformer.py'))},
        scope='encoder forward only; no PocketDiffModel replacement, optimizer, bridge, cache or fine-tuning',
    )
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output_dir/'code_fingerprints.json').write_text(json.dumps({str(p):sha(p) for p in sorted(Path('pocketdiff').rglob('*.py'))},indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise RuntimeError('TargetDiff encoder smoke failed')

if __name__ == '__main__':
    main()
