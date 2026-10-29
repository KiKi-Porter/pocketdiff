"""Summarize paired TargetDiff/PocketDiff bridge runs and trajectory geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CONDITIONS = {
    "targetdiff_original_fixed_apo": "original_backbone_rmsd",
    "targetdiff_adapter_fixed_apo": "internal_backbone_rmsd",
    "pocketdiff_v48": "bridged_backbone_rmsd",
    "pocketdiff_noisy_ligand": "noisy_ligand_pocket_bridge_backbone_rmsd",
}


def _mean(values):
    return float(np.mean(values)) if values else None


def _bootstrap_mean_ci(values, seed=20261025, iterations=10000):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    rng = np.random.RandomState(seed)
    draws = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def _read_trajectory(path):
    models = []
    current = None
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith("MODEL"):
                current = {"protein": {}, "ligand_pos": [], "ligand_element": []}
            elif line.startswith(("ATOM  ", "HETATM")):
                if current is None:
                    current = {"protein": {}, "ligand_pos": [], "ligand_element": []}
                atom_name = line[12:16].strip()
                chain = line[21].strip() or "A"
                try:
                    residue_number = int(line[22:26])
                    xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                except ValueError:
                    continue
                insertion_code = line[26].strip()
                element = line[76:78].strip().upper() or atom_name[:1].upper()
                if line.startswith("ATOM  "):
                    key = (chain, residue_number, insertion_code)
                    current["protein"].setdefault(key, {})[atom_name] = (xyz, element)
                elif line[17:20].strip() == "LIG":
                    current["ligand_pos"].append(xyz)
                    current["ligand_element"].append(element)
            elif line.startswith("ENDMDL") and current is not None:
                models.append(current)
                current = None
    if current is not None:
        models.append(current)
    return models


def _distance_summary(values, nominal_range):
    if not values:
        return {"count": 0, "mean": None, "median": None, "max_abs_error_from_range": None,
                "within_nominal_range_fraction": None}
    values = np.asarray(values, dtype=np.float64)
    low, high = nominal_range
    error = np.maximum(low - values, 0.0) + np.maximum(values - high, 0.0)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "max_abs_error_from_range": float(error.max()),
        "within_nominal_range_fraction": float(np.mean((values >= low) & (values <= high))),
    }


def _structure_metrics(model):
    residues = model["protein"]
    keys = sorted(residues)
    ca_ca = []
    peptide_cn = []
    paired_ca = []
    paired_cn = []
    for first, second in zip(keys, keys[1:]):
        if first[0] != second[0] or second[1] != first[1] + 1:
            continue
        left, right = residues[first], residues[second]
        if "CA" in left and "CA" in right:
            d = float(np.linalg.norm(np.asarray(left["CA"][0]) - np.asarray(right["CA"][0])))
            ca_ca.append(d)
            paired_ca.append((first, second, d))
        if "C" in left and "N" in right:
            d = float(np.linalg.norm(np.asarray(left["C"][0]) - np.asarray(right["N"][0])))
            peptide_cn.append(d)
            paired_cn.append((first, second, d))

    protein_pos = [atom[0] for residue in residues.values() for atom in residue.values()
                   if atom[1] != "H"]
    ligand_pos = [pos for pos, element in zip(model["ligand_pos"], model["ligand_element"])
                  if element != "H"]
    if protein_pos and ligand_pos:
        delta = np.asarray(protein_pos)[:, None, :] - np.asarray(ligand_pos)[None, :, :]
        distances = np.sqrt(np.maximum(np.sum(delta * delta, axis=-1), 0.0))
        clashes = {
            "heavy_atom_min_distance": float(distances.min()),
            "heavy_atom_pairs_below_2A": int(np.sum(distances < 2.0)),
            "heavy_atom_pairs_below_2_5A": int(np.sum(distances < 2.5)),
        }
    else:
        clashes = {"heavy_atom_min_distance": None, "heavy_atom_pairs_below_2A": None,
                   "heavy_atom_pairs_below_2_5A": None}
    return {
        "residue_count": len(residues),
        "ca_ca_A": _distance_summary(ca_ca, (3.5, 4.1)),
        "peptide_C_N_A": _distance_summary(peptide_cn, (1.2, 1.5)),
        "ligand_protein_heavy_atom": clashes,
        "_paired_ca": paired_ca,
        "_paired_cn": paired_cn,
    }


def _trajectory_metrics(path):
    models = _read_trajectory(path)
    if not models:
        return {"path": str(path), "model_count": 0, "error": "no PDB models found"}
    start = _structure_metrics(models[0])
    final = _structure_metrics(models[-1])

    def paired_delta(key):
        left = {(a, b): d for a, b, d in start[key]}
        right = {(a, b): d for a, b, d in final[key]}
        common = sorted(left.keys() & right.keys())
        if not common:
            return None
        changes = np.asarray([right[k] - left[k] for k in common], dtype=np.float64)
        return {"pair_count": len(common), "mean_signed_change_A": float(changes.mean()),
                "mean_absolute_change_A": float(np.abs(changes).mean()),
                "max_absolute_change_A": float(np.abs(changes).max())}

    geometry_change = {
        "ca_ca_A": paired_delta("_paired_ca"),
        "peptide_C_N_A": paired_delta("_paired_cn"),
    }
    start.pop("_paired_ca")
    start.pop("_paired_cn")
    final.pop("_paired_ca")
    final.pop("_paired_cn")
    return {
        "path": str(path),
        "model_count": len(models),
        "initial": start,
        "final": final,
        "geometry_change": geometry_change,
    }


def _condition_summary(rows, rmsd_key):
    rmsds = [float(row[rmsd_key]) for row in rows if rmsd_key in row]
    improvements = [float(row["apo_backbone_rmsd"]) - float(row[rmsd_key])
                    for row in rows if rmsd_key in row]
    if not rmsds:
        return None
    prefix = ("noisy_ligand_pocket_bridge" if rmsd_key.startswith("noisy_")
              else "bridged" if rmsd_key == "bridged_backbone_rmsd" else None)
    displacement_key = prefix + "_protein_displacement_mean" if prefix else None
    max_displacement_key = prefix + "_protein_displacement_max" if prefix else None
    displacement_values = [float(row[displacement_key]) for row in rows
                           if displacement_key and rmsd_key in row and displacement_key in row]
    max_displacement_values = [float(row[max_displacement_key]) for row in rows
                               if max_displacement_key and rmsd_key in row and max_displacement_key in row]
    return {
        "sample_count": len(rmsds),
        "mean_backbone_rmsd_A": _mean(rmsds),
        "mean_improvement_vs_apo_A": _mean(improvements),
        "median_improvement_vs_apo_A": float(np.median(improvements)),
        "improved_fraction_vs_apo": float(np.mean(np.asarray(improvements) > 0)),
        "improvement_95pct_paired_bootstrap_ci_A": _bootstrap_mean_ci(improvements),
        "mean_protein_displacement_A": _mean(displacement_values),
        "max_protein_displacement_A": max(max_displacement_values, default=None),
        "pocket_calls_per_sample": sorted(set(
            int(row.get("noisy_ligand_pocket_bridge_calls", 0) if rmsd_key.startswith("noisy_")
                else row.get("pocket_calls", 0) if rmsd_key == "bridged_backbone_rmsd" else 0)
            for row in rows if rmsd_key in row
        )),
    }


def summarize(report_path: Path):
    report = json.loads(report_path.read_text())
    rows = report["records"]
    conditions = {
        name: _condition_summary(rows, key)
        for name, key in CONDITIONS.items()
    }
    paired = []
    for row in rows:
        if "noisy_ligand_pocket_bridge_backbone_rmsd" in row:
            paired.append(float(row["noisy_ligand_pocket_bridge_backbone_rmsd"])
                          - float(row["bridged_backbone_rmsd"]))

    trajectory_geometry = {}
    for condition in ("original_targetdiff", "internal_cross", "bridged",
                      "noisy_ligand_pocket_bridge"):
        paths = [row.get("trajectories", {}).get(condition) for row in rows]
        paths = [Path(path) for path in paths if path]
        metrics = [_trajectory_metrics(path) for path in paths if path.exists()]
        if metrics:
            final_ca = [m["final"]["ca_ca_A"]["mean"] for m in metrics
                        if m["final"]["ca_ca_A"]["mean"] is not None]
            final_cn = [m["final"]["peptide_C_N_A"]["mean"] for m in metrics
                        if m["final"]["peptide_C_N_A"]["mean"] is not None]
            final_clash = [m["final"]["ligand_protein_heavy_atom"]["heavy_atom_pairs_below_2A"]
                           for m in metrics
                           if m["final"]["ligand_protein_heavy_atom"]["heavy_atom_pairs_below_2A"] is not None]
            trajectory_geometry[condition] = {
                "sample_count": len(metrics),
                "mean_final_ca_ca_A_per_sample": _mean(final_ca),
                "mean_final_peptide_C_N_A_per_sample": _mean(final_cn),
                "mean_final_ligand_protein_pairs_below_2A": _mean(final_clash),
                "per_sample": metrics,
            }

    return {
        "samples": len(rows),
        "elapsed_seconds": report.get("elapsed_seconds"),
        "all_finite": report.get("all_finite"),
        "all_coupled_calls_changed_protein": report.get("all_coupled_calls_changed_protein"),
        "holo_is_model_input": report.get("holo_is_model_input"),
        "late_protocol": report.get("late_protocol"),
        "conditions": conditions,
        "paired_noisy_minus_v48_backbone_rmsd_A": {
            "mean": _mean(paired),
            "mean_95pct_paired_bootstrap_ci": _bootstrap_mean_ci(paired),
            "negative_favors_noisy_ligand_adapter": True,
        },
        "trajectory_geometry": trajectory_geometry,
        "ligand_chemistry_note": (
            "PDB trajectories do not contain ligand bond topology; connectedness and chemical validity "
            "must be read from the paired TargetDiff generation evaluation, not inferred from these PDBs."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path,
                        default=Path("runs/bridge_phase1_clean_20261004/comparison.json"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    summary = summarize(args.report)
    text = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    target = args.output or args.report.with_name("summary.json")
    target.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
