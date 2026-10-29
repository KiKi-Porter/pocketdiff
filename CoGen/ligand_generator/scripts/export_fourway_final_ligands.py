"""Export four matched final ligand conformations per bridge test complex.

Three strategies are reused from a completed matched bridge run. The missing
original-TargetDiff + noisy-ligand PocketDiff condition is sampled with the
same test complexes, seed, initial ligand state, and TargetDiff RNG trace.
Outputs contain ligand atoms only, with one PDB MODEL per strategy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
POCKET_ROOT = ROOT.parent / "pocketdiff"
sys.path.insert(0, str(ROOT / "scripts"))

import bridge_targetdiff_pocketdiff as bridge
from pocketdiff.preprocessing.diffusion_cache import load_diffusion_cache
from pocketdiff.training.diffusion import load_diffusion_checkpoint


def _final_ligand_lines(path: Path, expected_models: int = 22) -> list[str]:
    models: list[list[str]] = []
    current: list[str] | None = None
    for line in path.read_text().splitlines():
        if line.startswith("MODEL"):
            current = []
        elif line.startswith("ENDMDL"):
            if current is not None:
                models.append(current)
                current = None
        elif current is not None and line.startswith("HETATM") and line[17:20].strip() == "LIG":
            current.append(line)
    if len(models) != expected_models:
        raise ValueError(f"{path}: expected {expected_models} PDB models, found {len(models)}")
    if not models[-1]:
        raise ValueError(f"{path}: final model contains no ligand atoms")
    return models[-1]


def _sample_original_with_pocketdiff(args, values, report_rows, device):
    original_model, _ = bridge._load_model(args.original_checkpoint, device)
    pocket_loaded = load_diffusion_checkpoint(
        args.pocket_adapter_checkpoint, map_location=str(device)
    )
    pocket_model = pocket_loaded.model.eval()
    if pocket_model.prediction_type != "bridge_velocity":
        raise ValueError(
            "Expected bridge_velocity PocketDiff checkpoint, got "
            f"{pocket_model.prediction_type}"
        )

    final_ligands = {}
    for index, (raw_value, row) in enumerate(zip(values, report_rows)):
        value, center = bridge._center_value(raw_value)
        value = bridge._move_value(value, device)
        if value.sample_id != row["sample_id"]:
            raise ValueError(f"sample order mismatch at {index}: {value.sample_id} != {row['sample_id']}")
        bridge._assert_holo_invariant_state(
            value,
            value.protein_pos_apo.float(),
            value.ligand_pos_ref.float(),
            value.ligand_type_ref.long(),
        )

        ligand_count = value.ligand_pos_ref.shape[0]
        generator = torch.Generator(device=device.type).manual_seed(args.seed + index)
        protein_pos = value.protein_pos_apo.float()
        init_ligand_pos = protein_pos.mean(dim=0, keepdim=True) + torch.randn(
            (ligand_count, 3), generator=generator, device=device
        )
        init_ligand_v = bridge._sample_uniform_classes(ligand_count, 13, generator, device)
        init = {
            "protein_pos": protein_pos,
            "protein_v": value.protein_feature.float(),
            "ligand_pos": init_ligand_pos,
            "ligand_v": init_ligand_v,
            "batch_protein": torch.zeros(protein_pos.shape[0], dtype=torch.long, device=device),
            "batch_ligand": torch.zeros(ligand_count, dtype=torch.long, device=device),
        }
        trace = bridge._make_trace(ligand_count, args.seed + index + 100000, device)
        final, protein_states, ligand_states, diagnostics = bridge._run_condition(
            original_model, pocket_model, value, init, trace, coupled=True
        )
        if len(diagnostics) != 20 or not bridge._trajectory_is_finite(protein_states, ligand_states):
            raise RuntimeError(f"invalid original-TargetDiff/PocketDiff rollout: {value.sample_id}")

        positions = final["ligand_pos"].detach().cpu() + center.cpu()
        atom_types = final["ligand_v"].detach().cpu().long()
        final_ligands[value.sample_id] = [
            bridge._pdb_atom(
                "HETATM", atom_index, f"L{atom_index}", "LIG", "Z", "1",
                position, bridge.TARGETDIFF_LIGAND_ELEMENTS[int(atom_type)],
            ).rstrip("\n")
            for atom_index, (atom_type, position) in enumerate(zip(atom_types, positions), 1)
        ]
        print(json.dumps({
            "sample_id": value.sample_id,
            "pocket_calls": len(diagnostics),
            "ligand_atoms": len(final_ligands[value.sample_id]),
            "finite": True,
        }), flush=True)
    return final_ligands


def _write_four_models(path: Path, strategy_lines: list[tuple[str, list[str]]]):
    atom_counts = {len(lines) for _, lines in strategy_lines}
    if len(atom_counts) != 1:
        raise ValueError(f"atom count differs across strategies for {path.stem}: {sorted(atom_counts)}")
    with path.open("w") as handle:
        for model_index, (strategy, lines) in enumerate(strategy_lines, 1):
            handle.write(f"MODEL     {model_index:4d}\n")
            handle.write(f"REMARK 900 STRATEGY {model_index}: {strategy}\n")
            for line in lines:
                handle.write(line + "\n")
            handle.write("ENDMDL\n")
        handle.write("END\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--pocket-cache", type=Path, default=POCKET_ROOT / "data/prepared_diffusion_full_v47.pt")
    parser.add_argument("--original-checkpoint", type=Path, default=ROOT / "pretrained_models/pretrained_diffusion.pt")
    parser.add_argument("--pocket-adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    report = json.loads(args.comparison.read_text())
    rows = report["records"][: args.samples]
    values = load_diffusion_cache(args.pocket_cache)["splits"]["test"][: args.samples]
    if len(rows) != args.samples or len(values) != args.samples:
        raise ValueError(f"expected {args.samples} paired samples, got report={len(rows)}, cache={len(values)}")
    if report.get("original_checkpoint") != str(args.original_checkpoint.resolve()):
        raise ValueError("original TargetDiff checkpoint differs from the source comparison")
    if report.get("pocket_adapter_checkpoint") != str(args.pocket_adapter_checkpoint.resolve()):
        raise ValueError("PocketDiff checkpoint differs from the source comparison")
    if not report.get("all_finite") or not report.get("holo_invariance_passed"):
        raise ValueError("source comparison is incomplete or failed its apo-only invariance gate")

    device = torch.device(args.device)
    new_condition = _sample_original_with_pocketdiff(args, values, rows, device)
    args.output.mkdir(parents=True, exist_ok=True)
    strategy_keys = (
        ("Original TargetDiff", "original_targetdiff"),
        ("Fine-tuned TargetDiff", "internal_cross"),
        ("Original TargetDiff + PocketDiff", None),
        ("Fine-tuned TargetDiff + PocketDiff", "noisy_ligand_pocket_bridge"),
    )
    for row in rows:
        sample_id = row["sample_id"]
        selected = []
        for label, trajectory_key in strategy_keys:
            if trajectory_key is None:
                lines = new_condition[sample_id]
            else:
                lines = _final_ligand_lines(Path(row["trajectories"][trajectory_key]))
            selected.append((label, lines))
        filename = sample_id.replace(".", "-").replace("/", "-") + "_fourway_final_ligands.pdb"
        _write_four_models(args.output / filename, selected)
    print(json.dumps({"samples": len(rows), "models_per_file": 4, "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
