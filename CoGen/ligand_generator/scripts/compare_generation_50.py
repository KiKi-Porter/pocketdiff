from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Lipinski, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.spatial.distance import cdist

import utils.reconstruct as reconstruct
import utils.transforms as transforms
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from utils.evaluation import analyze, atom_num, eval_bond_length, scoring_func


DEFAULT_OLD_CHECKPOINT = Path("pretrained_models/pretrained_diffusion.pt")
DEFAULT_NEW_CHECKPOINT = Path("runs/md_dynamic_300/partial_best.pt")
DEFAULT_CACHE = Path("data/md300/prepared_targetdiff_md300_10frames.pt")
DEFAULT_OUTPUT = Path("runs/generation_compare_50_20261002")
N_PROTEIN_FEATURES = 27
N_LIGAND_CLASSES = 13
N_BOOTSTRAP = 2000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _set_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_test_indices(
    records: list[dict[str, Any]],
    test_indices: list[int],
    num_samples: int,
    seed: int,
) -> list[int]:
    """Select test frames across trajectories, with at most two per trajectory."""
    grouped: dict[str, list[int]] = defaultdict(list)
    record_by_index = {int(record["sample_index"]): record for record in records}
    for index in test_indices:
        record = record_by_index[int(index)]
        if record.get("split") != "test":
            raise ValueError(f"Sample {index} is not in the held-out test split")
        grouped[str(record["trajectory_id"])].append(int(index))
    for indices in grouped.values():
        indices.sort(
            key=lambda index: (
                int(record_by_index[index]["cluster_id"]),
                int(record_by_index[index]["cluster_frame"]),
                index,
            )
        )

    if num_samples <= 0 or num_samples > min(2 * len(grouped), len(test_indices)):
        raise ValueError(
            f"Cannot select {num_samples} samples with at most two per trajectory "
            f"from {len(grouped)} trajectories and {len(test_indices)} test frames"
        )

    rng = random.Random(seed)
    trajectory_ids = sorted(grouped)
    rng.shuffle(trajectory_ids)
    selected: list[int] = []
    first_pass = trajectory_ids[: min(num_samples, len(trajectory_ids))]
    for trajectory_id in first_pass:
        selected.append(rng.choice(grouped[trajectory_id]))
    if len(selected) == num_samples:
        return selected

    # Only trajectories already represented in the first pass can contribute
    # a second frame; this preserves the two-per-trajectory cap.
    remaining_trajectories = [
        trajectory_id
        for trajectory_id in first_pass
        if len(grouped[trajectory_id]) > 1
    ]
    rng.shuffle(remaining_trajectories)
    for trajectory_id in remaining_trajectories:
        chosen = set(
            index
            for index in selected
            if str(record_by_index[index]["trajectory_id"]) == trajectory_id
        )
        candidates = [index for index in grouped[trajectory_id] if index not in chosen]
        if candidates:
            selected.append(rng.choice(candidates))
        if len(selected) == num_samples:
            break
    if len(selected) != num_samples:
        raise ValueError(f"Selected {len(selected)} of {num_samples} requested samples")
    return selected


def selected_manifest_records(
    records: list[dict[str, Any]], selected_indices: list[int]
) -> list[dict[str, Any]]:
    record_by_index = {int(record["sample_index"]): record for record in records}
    output = []
    for ordinal, index in enumerate(selected_indices):
        record = record_by_index[index]
        output.append(
            {
                "ordinal": ordinal,
                "sample_index": index,
                "trajectory_id": record["trajectory_id"],
                "pdb_id": record["pdb_id"],
                "cluster_id": int(record["cluster_id"]),
                "cluster_frame": int(record["cluster_frame"]),
                "protein_atom_count": int(record["protein_atom_count"]),
                "source_sample_path": record["source_sample_path"],
            }
        )
    return output


def protein_only(sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return only protein coordinates, TargetDiff features, and elements."""
    required = ("protein_pos", "protein_atom_feature", "protein_element")
    missing = [key for key in required if key not in sample]
    if missing:
        raise KeyError(f"Prepared sample is missing protein inputs: {missing}")
    protein_pos = sample["protein_pos"].detach().to(dtype=torch.float32, device="cpu")
    protein_features = sample["protein_atom_feature"].detach().to(
        dtype=torch.float32, device="cpu"
    )
    protein_elements = sample["protein_element"].detach().to(
        dtype=torch.long, device="cpu"
    )
    if protein_features.ndim != 2 or protein_features.shape != (
        protein_pos.shape[0],
        N_PROTEIN_FEATURES,
    ):
        raise ValueError(
            "Protein feature contract mismatch: expected [Nprotein, 27], got "
            f"{tuple(protein_features.shape)}"
        )
    if protein_elements.shape != (protein_pos.shape[0],):
        raise ValueError("Protein element/coordinate count mismatch")
    return protein_pos, protein_features, protein_elements


def prepare_pair_condition(
    protein_pos: torch.Tensor,
    num_classes: int,
    seed: int,
    device: torch.device,
) -> tuple[int, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Sample one atom count and initial ligand state from protein-only input."""
    np.random.seed(seed)
    pocket_size = atom_num.get_space_size(protein_pos.numpy())
    num_atoms = int(atom_num.sample_atom_num(pocket_size))
    if num_atoms <= 0:
        raise ValueError(f"Atom-count prior returned invalid count {num_atoms}")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    center = protein_pos.to(device).mean(dim=0, keepdim=True)
    init_pos = center + torch.randn((num_atoms, 3), device=device)
    init_logits = torch.zeros((num_atoms, num_classes), device=device)
    init_types = log_sample_categorical(init_logits)
    state = _rng_state(device)
    input_hash = sha256_arrays(
        init_pos.detach().cpu().numpy(),
        init_types.detach().cpu().numpy(),
        np.asarray([num_atoms], dtype=np.int64),
    )
    return num_atoms, init_pos, init_types, {
        "rng_state": state,
        "input_hash": input_hash,
        "pocket_size_prior_value": float(pocket_size),
    }


def sample_one(
    model: ScorePosNet3D,
    protein_pos: torch.Tensor,
    protein_features: torch.Tensor,
    init_pos: torch.Tensor,
    init_types: torch.Tensor,
    num_steps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    protein_pos_device = protein_pos.to(device)
    protein_features_device = protein_features.to(device)
    batch_protein = torch.zeros(
        protein_pos_device.shape[0], dtype=torch.long, device=device
    )
    batch_ligand = torch.zeros(init_pos.shape[0], dtype=torch.long, device=device)
    result = model.sample_diffusion(
        protein_pos=protein_pos_device,
        protein_v=protein_features_device,
        batch_protein=batch_protein,
        init_ligand_pos=init_pos.clone(),
        init_ligand_v=init_types.clone(),
        batch_ligand=batch_ligand,
        num_steps=num_steps,
        center_pos_mode="protein",
        pos_only=False,
        return_trajectory=False,
        show_progress=False,
    )
    ligand_pos = result["pos"].detach().cpu().numpy().astype(np.float32)
    ligand_types = result["v"].detach().cpu().numpy().astype(np.int64)
    if ligand_pos.shape != (init_pos.shape[0], 3):
        raise ValueError(f"Unexpected generated coordinate shape: {ligand_pos.shape}")
    if ligand_types.shape != (init_types.shape[0],):
        raise ValueError(f"Unexpected generated atom-class shape: {ligand_types.shape}")
    if not np.isfinite(ligand_pos).all():
        raise FloatingPointError("Sampler returned non-finite ligand coordinates")
    if np.any((ligand_types < 0) | (ligand_types >= N_LIGAND_CLASSES)):
        raise ValueError("Sampler returned atom classes outside TargetDiff's 13 classes")
    return ligand_pos, ligand_types


def _pocket_contact_metrics(
    ligand_pos: np.ndarray,
    protein_pos: torch.Tensor,
    protein_elements: torch.Tensor,
) -> dict[str, float | int | None]:
    protein_xyz = protein_pos.numpy()
    heavy_mask = protein_elements.numpy() != 1
    protein_heavy = protein_xyz[heavy_mask]
    if ligand_pos.shape[0] == 0 or protein_heavy.shape[0] == 0:
        return {
            "clash_pairs_lt_2_0A": None,
            "ligand_atoms_with_clash_lt_2_0A": None,
            "mean_min_protein_distance_A": None,
            "min_protein_distance_A": None,
        }
    distances = cdist(ligand_pos, protein_heavy)
    minimum = distances.min(axis=1)
    return {
        "clash_pairs_lt_2_0A": int((distances < 2.0).sum()),
        "ligand_atoms_with_clash_lt_2_0A": int((minimum < 2.0).sum()),
        "mean_min_protein_distance_A": float(minimum.mean()),
        "min_protein_distance_A": float(minimum.min()),
    }


def _pair_distance_histograms(
    ligand_pos: np.ndarray, atom_types: list[int]
) -> dict[str, list[int]]:
    pair_distances = eval_bond_length.pair_distance_from_pos_v(ligand_pos, atom_types)
    cc_bins = np.linspace(0, 2, 100)
    all_bins = np.linspace(0, 12, 100)
    cc_distances = [
        distance
        for atom_pair, distance in pair_distances
        if atom_pair == (6, 6) and distance < 2
    ]
    all_distances = [distance for _, distance in pair_distances if distance < 12]
    return {
        "CC_2A": np.bincount(
            np.searchsorted(cc_bins, cc_distances), minlength=len(cc_bins) + 1
        ).astype(int).tolist(),
        "All_12A": np.bincount(
            np.searchsorted(all_bins, all_distances), minlength=len(all_bins) + 1
        ).astype(int).tolist(),
    }


def evaluate_generated(
    ligand_pos: np.ndarray,
    ligand_classes: np.ndarray,
    protein_pos: torch.Tensor,
    protein_elements: torch.Tensor,
) -> tuple[dict[str, Any], Chem.Mol | None]:
    atom_types = transforms.get_atomic_number_from_index(
        ligand_classes, mode="add_aromatic"
    )
    aromatic = transforms.is_aromatic_from_index(
        ligand_classes, mode="add_aromatic"
    )
    stable, stable_atoms, n_atoms = analyze.check_stability(ligand_pos, atom_types)
    metrics: dict[str, Any] = {
        "sampled_atom_count": int(len(ligand_classes)),
        "molecule_stable": bool(stable),
        "stable_atom_count": int(stable_atoms),
        "atom_stable_fraction": float(stable_atoms / max(n_atoms, 1)),
        **_pocket_contact_metrics(ligand_pos, protein_pos, protein_elements),
    }
    metrics["ligand_pair_distance_histogram"] = _pair_distance_histograms(
        ligand_pos, atom_types
    )

    molecule = None
    try:
        molecule = reconstruct.reconstruct_from_generated(
            ligand_pos.astype(np.float64, copy=False),
            atom_types,
            aromatic=aromatic,
            basic_mode=False,
        )
        Chem.SanitizeMol(molecule)
        smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        parsed = Chem.MolFromSmiles(smiles)
        if parsed is None:
            raise ValueError("RDKit could not parse reconstructed canonical SMILES")
        fragment_count = len(Chem.GetMolFrags(molecule))
        metrics.update(
            {
                "reconstruction_success": True,
                "valid_smiles": True,
                "connected": fragment_count == 1,
                "fragment_count": int(fragment_count),
                "smiles": smiles,
                "qed": float(QED.qed(molecule)),
                "sa": float(scoring_func.compute_sa_score(molecule)),
                "logp": float(Descriptors.MolLogP(molecule)),
                "molecular_weight": float(Descriptors.MolWt(molecule)),
                "h_bond_donors": int(Lipinski.NumHDonors(molecule)),
                "h_bond_acceptors": int(Lipinski.NumHAcceptors(molecule)),
                "ring_count": int(molecule.GetRingInfo().NumRings()),
                "scaffold_smiles": MurckoScaffold.MurckoScaffoldSmiles(
                    mol=molecule, includeChirality=False
                ),
            }
        )
        metrics["lipinski_score"] = int(scoring_func.obey_lipinski(molecule))
        metrics["bond_distances"] = [
            {
                "bond_type": list(bond_type),
                "distance_A": float(distance),
            }
            for bond_type, distance in eval_bond_length.bond_distance_from_mol(molecule)
        ]
    except Exception as exc:
        metrics.update(
            {
                "reconstruction_success": False,
                "valid_smiles": False,
                "connected": False,
                "fragment_count": None,
                "smiles": None,
                "reconstruction_error": f"{type(exc).__name__}: {exc}",
            }
        )
        molecule = None
    return metrics, molecule


def _mean_median(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {"n": 0, "mean": None, "median": None}
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
    }


def _rate(values: list[Any]) -> dict[str, Any]:
    if not values:
        return {"numerator": 0, "denominator": 0, "rate": None}
    numerator = sum(bool(value) for value in values)
    return {
        "numerator": int(numerator),
        "denominator": int(len(values)),
        "rate": float(numerator / len(values)),
    }


def _paired_bootstrap(
    old_values: list[Any],
    new_values: list[Any],
    seed: int,
    num_bootstrap: int = N_BOOTSTRAP,
    clusters: list[Any] | None = None,
    include: list[bool] | None = None,
) -> dict[str, Any]:
    if len(old_values) != len(new_values):
        raise ValueError("Paired arrays must have the same length")
    if clusters is not None and len(clusters) != len(old_values):
        raise ValueError("Cluster labels must match the paired arrays")
    if include is not None and len(include) != len(old_values):
        raise ValueError("Inclusion mask must match the paired arrays")

    paired: list[tuple[float, str]] = []
    for index, (old, new) in enumerate(zip(old_values, new_values)):
        if include is not None and not include[index]:
            continue
        if old is None or new is None:
            continue
        try:
            old_number, new_number = float(old), float(new)
        except (TypeError, ValueError):
            continue
        if math.isfinite(old_number) and math.isfinite(new_number):
            cluster = str(clusters[index]) if clusters is not None else str(index)
            paired.append((new_number - old_number, cluster))
    if not paired:
        return {
            "n_pairs": 0,
            "n_clusters": 0,
            "bootstrap_unit": "trajectory" if clusters is not None else "sample_pair",
            "mean_new_minus_old": None,
            "median_new_minus_old": None,
            "ci95": None,
        }

    values = np.asarray([value for value, _ in paired], dtype=float)
    rng = np.random.default_rng(seed)
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in paired:
        grouped[cluster].append(value)
    cluster_names = list(grouped)
    if clusters is None:
        draw_indices = rng.integers(0, len(values), size=(num_bootstrap, len(values)))
        means = values[draw_indices].mean(axis=1)
    else:
        means = np.empty(num_bootstrap, dtype=float)
        for draw_index in range(num_bootstrap):
            sampled_clusters = rng.integers(
                0, len(cluster_names), size=len(cluster_names)
            )
            sampled_values = [
                value
                for index in sampled_clusters
                for value in grouped[cluster_names[index]]
            ]
            means[draw_index] = float(np.mean(sampled_values))
    return {
        "n_pairs": int(len(values)),
        "n_clusters": int(len(cluster_names)),
        "bootstrap_unit": "trajectory" if clusters is not None else "sample_pair",
        "mean_new_minus_old": float(values.mean()),
        "median_new_minus_old": float(np.median(values)),
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
    }


def summarize_model(records: list[dict[str, Any]]) -> dict[str, Any]:
    sampling_ok = [record for record in records if record.get("sampling_success")]
    metrics = [record.get("metrics", {}) for record in sampling_ok]
    complete = [m for m in metrics if m.get("connected") is True]
    reconstructed = [m for m in metrics if m.get("reconstruction_success") is True]
    smiles = [m["smiles"] for m in complete if m.get("smiles")]
    scaffolds = [
        m["scaffold_smiles"] or f"ACYCLIC:{m['smiles']}"
        for m in complete
        if m.get("smiles")
    ]
    fingerprints = []
    for molecule_record in records:
        smiles_value = molecule_record.get("metrics", {}).get("smiles")
        if molecule_record.get("metrics", {}).get("connected") and smiles_value:
            molecule = Chem.MolFromSmiles(smiles_value)
            if molecule is not None:
                fingerprints.append(
                    Chem.RDKFingerprint(molecule, fpSize=2048, maxPath=4)
                )
    tanimoto_distances = []
    for index, fingerprint in enumerate(fingerprints):
        if index + 1 < len(fingerprints):
            similarities = DataStructs.BulkTanimotoSimilarity(
                fingerprint, fingerprints[index + 1 :]
            )
            tanimoto_distances.extend(1.0 - value for value in similarities)

    numeric_fields = (
        "atom_stable_fraction",
        "qed",
        "sa",
        "logp",
        "molecular_weight",
        "h_bond_donors",
        "h_bond_acceptors",
        "lipinski_score",
        "clash_pairs_lt_2_0A",
        "ligand_atoms_with_clash_lt_2_0A",
        "mean_min_protein_distance_A",
        "min_protein_distance_A",
    )
    chemistry_fields = {
        "qed",
        "sa",
        "logp",
        "molecular_weight",
        "h_bond_donors",
        "h_bond_acceptors",
        "lipinski_score",
    }
    continuous = {}
    for field in numeric_fields:
        source = complete if field in chemistry_fields else metrics
        continuous[field] = _mean_median(
            [
                float(m[field])
                for m in source
                if m.get(field) is not None and math.isfinite(float(m[field]))
            ]
        )
    rates = {
        "sampling_success": _rate([r.get("sampling_success") for r in records]),
        "molecule_stable": _rate(
            [m.get("molecule_stable") for m in metrics if m.get("molecule_stable") is not None]
        ),
        "reconstruction_success": _rate(
            [m.get("reconstruction_success") for m in metrics]
        ),
        "connected": _rate([m.get("connected") for m in metrics]),
        "valid_and_connected": _rate([m.get("connected") is True for m in metrics]),
    }
    all_bonds = [
        (tuple(item["bond_type"]), float(item["distance_A"]))
        for metric in metrics
        for item in metric.get("bond_distances", [])
    ]
    bond_jsd: dict[str, Any] = {}
    if all_bonds:
        profile = eval_bond_length.get_bond_length_profile(all_bonds)
        bond_jsd = eval_bond_length.eval_bond_length_profile(profile)
    non_null_jsd = [value for value in bond_jsd.values() if value is not None]
    pair_jsd: dict[str, Any] = {}
    pair_histograms = [
        metric.get("ligand_pair_distance_histogram", {}) for metric in metrics
    ]
    pair_profile = {}
    for key in ("CC_2A", "All_12A"):
        distributions = [
            np.asarray(histogram[key], dtype=float)
            for histogram in pair_histograms
            if key in histogram
        ]
        if distributions:
            counts = np.sum(distributions, axis=0)
            if counts.sum() > 0:
                pair_profile[key] = counts / counts.sum()
    if pair_profile:
        pair_jsd = eval_bond_length.eval_pair_length_profile(pair_profile)
    return {
        "requested_samples": len(records),
        "sampling_success_count": len(sampling_ok),
        "reconstruction_count": len(reconstructed),
        "connected_molecule_count": len(complete),
        "rates": rates,
        "continuous": continuous,
        "unique_smiles_count": len(set(smiles)),
        "unique_fraction_among_connected": (
            len(set(smiles)) / len(smiles) if smiles else None
        ),
        "unique_scaffold_count": len(set(scaffolds)),
        "scaffold_fraction_among_connected": (
            len(set(scaffolds)) / len(scaffolds) if scaffolds else None
        ),
        "mean_pairwise_fingerprint_distance": (
            float(np.mean(tanimoto_distances)) if tanimoto_distances else None
        ),
        "bond_length_jsd": bond_jsd,
        "mean_available_bond_length_jsd": (
            float(np.mean(non_null_jsd)) if non_null_jsd else None
        ),
        "protein_ligand_pair_distance_jsd": pair_jsd,
    }


def build_summary(
    pair_records: list[dict[str, Any]],
    expected_samples: int,
    seed: int,
    steps: int,
) -> dict[str, Any]:
    model_records = {
        "original_targetdiff": [pair.get("original_targetdiff", {}) for pair in pair_records],
        "aplc_finetuned": [pair.get("aplc_finetuned", {}) for pair in pair_records],
    }
    old_summary = summarize_model(model_records["original_targetdiff"])
    new_summary = summarize_model(model_records["aplc_finetuned"])

    fields = (
        "molecule_stable",
        "atom_stable_fraction",
        "reconstruction_success",
        "connected",
        "qed",
        "sa",
        "logp",
        "molecular_weight",
        "lipinski_score",
        "clash_pairs_lt_2_0A",
        "ligand_atoms_with_clash_lt_2_0A",
        "mean_min_protein_distance_A",
    )
    paired = {}
    trajectory_ids = [
        str(pair.get("trajectory_id", f"pair_{index}"))
        for index, pair in enumerate(pair_records)
    ]
    for field_index, field in enumerate(fields):
        old_values = [
            pair.get("original_targetdiff", {}).get("metrics", {}).get(field)
            for pair in pair_records
        ]
        new_values = [
            pair.get("aplc_finetuned", {}).get("metrics", {}).get(field)
            for pair in pair_records
        ]
        paired[field] = _paired_bootstrap(
            old_values,
            new_values,
            seed=seed + 31 * field_index,
            clusters=trajectory_ids,
        )

    connected_pair_mask = [
        pair.get("original_targetdiff", {}).get("metrics", {}).get("connected") is True
        and pair.get("aplc_finetuned", {}).get("metrics", {}).get("connected") is True
        for pair in pair_records
    ]
    paired_connected_chemistry = {}
    for field_index, field in enumerate(
        ("qed", "sa", "logp", "molecular_weight", "lipinski_score")
    ):
        old_values = [
            pair.get("original_targetdiff", {}).get("metrics", {}).get(field)
            for pair in pair_records
        ]
        new_values = [
            pair.get("aplc_finetuned", {}).get("metrics", {}).get(field)
            for pair in pair_records
        ]
        paired_stats = _paired_bootstrap(
            old_values,
            new_values,
            seed=seed + 997 + 31 * field_index,
            clusters=trajectory_ids,
            include=connected_pair_mask,
        )
        eligible = [
            (old, new)
            for old, new, include_pair in zip(
                old_values, new_values, connected_pair_mask
            )
            if include_pair
            and old is not None
            and new is not None
            and math.isfinite(float(old))
            and math.isfinite(float(new))
        ]
        paired_connected_chemistry[field] = {
            **paired_stats,
            "old_mean": (
                float(np.mean([old for old, _ in eligible])) if eligible else None
            ),
            "new_mean": (
                float(np.mean([new for _, new in eligible])) if eligible else None
            ),
        }

    return {
        "protocol": {
            "expected_samples": expected_samples,
            "completed_pairs": len(pair_records),
            "diffusion_steps": steps,
            "seed": seed,
            "same_test_pockets_per_pair": True,
            "same_prior_atom_count_per_pair": True,
            "same_initial_ligand_coordinates_and_classes_per_pair": True,
            "paired_reverse_diffusion_rng": True,
            "reference_ligand_fields_passed_to_model": False,
            "atom_count_source": "TargetDiff pocket-size prior",
            "reconstruction": "OpenBabel-based TargetDiff reconstruction with predicted aromatic classes",
            "docking_or_binding_affinity_evaluated": False,
        },
        "models": {
            "original_targetdiff": old_summary,
            "aplc_finetuned": new_summary,
        },
        "paired_new_minus_old": paired,
        "paired_connected_chemistry": paired_connected_chemistry,
        "interpretation_notes": [
            "Lower SA score is generally preferable; higher QED and Lipinski score are generally preferable.",
            "Molecule stability is TargetDiff's distance-based valence heuristic, not a force-field energy test.",
            "A protein-ligand contact below 2.0 A is reported as a simple severe-clash diagnostic and is not an energy calculation.",
            "Docking, binding affinity, and experimental activity were not evaluated.",
            "The 50 samples use held-out MD test pockets; frames are limited to at most two per held-out trajectory.",
            "Paired confidence intervals resample trajectories as clusters; connected-chemistry comparisons use only pairs connected in both models.",
        ],
    }


def write_report(path: Path, summary: dict[str, Any], manifest: list[dict[str, Any]]) -> None:
    models = summary["models"]
    old = models["original_targetdiff"]
    new = models["aplc_finetuned"]

    def rate(model_summary: dict[str, Any], key: str) -> str:
        value = model_summary["rates"][key]["rate"]
        count = model_summary["rates"][key]["numerator"]
        denominator = model_summary["rates"][key]["denominator"]
        if value is None:
            return "n/a"
        return f"{count}/{denominator} ({value:.1%})"

    def stat(model_summary: dict[str, Any], key: str) -> str:
        value = model_summary["continuous"][key]
        if value["mean"] is None:
            return "n/a"
        return f"{value['mean']:.3f} (median {value['median']:.3f}, n={value['n']})"

    lines = [
        "# TargetDiff 50-Pocket Generation Comparison",
        "",
        f"- Held-out pockets: {len(manifest)}",
        f"- Diffusion steps: {summary['protocol']['diffusion_steps']}",
        "- Models: original TargetDiff vs APLC partial-unfreeze fine-tune",
        "- Ligand atom count: sampled from the same pocket-size prior for each pair",
        "- Reference ligand coordinates/types/bonds are not supplied to either sampler",
        "- Docking/affinity: not evaluated",
        "",
        "## Generation and Chemistry",
        "",
        "| Metric | Original TargetDiff | APLC fine-tuned |",
        "|---|---:|---:|",
        f"| Sampling completed | {rate(old, 'sampling_success')} | {rate(new, 'sampling_success')} |",
        f"| Distance-based molecule stability | {rate(old, 'molecule_stable')} | {rate(new, 'molecule_stable')} |",
        f"| Reconstruction success | {rate(old, 'reconstruction_success')} | {rate(new, 'reconstruction_success')} |",
        f"| Connected reconstructed molecules | {rate(old, 'connected')} | {rate(new, 'connected')} |",
        f"| Unique fraction among connected | {old['unique_fraction_among_connected']} | {new['unique_fraction_among_connected']} |",
        f"| QED mean (median) | {stat(old, 'qed')} | {stat(new, 'qed')} |",
        f"| SA mean (median; lower is better) | {stat(old, 'sa')} | {stat(new, 'sa')} |",
        f"| LogP mean (median) | {stat(old, 'logp')} | {stat(new, 'logp')} |",
        f"| Mean severe clash pairs per molecule (<2.0 A) | {stat(old, 'clash_pairs_lt_2_0A')} | {stat(new, 'clash_pairs_lt_2_0A')} |",
        f"| Mean nearest protein distance (A) | {stat(old, 'mean_min_protein_distance_A')} | {stat(new, 'mean_min_protein_distance_A')} |",
        "",
        "## Paired Changes",
        "",
        "The table reports fine-tuned minus original. Confidence intervals use a paired cluster bootstrap over trajectories. Binary and geometric metrics use all generated pairs; chemistry descriptors here use pairs where both outputs reconstructed, including disconnected outputs.",
        "",
        "| Metric | N pairs | N trajectories | Mean difference | 95% cluster-bootstrap CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, value in summary["paired_new_minus_old"].items():
        ci = value["ci95"]
        ci_text = "n/a" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        mean_text = (
            "n/a"
            if value["mean_new_minus_old"] is None
            else f"{value['mean_new_minus_old']:.4f}"
        )
        lines.append(
            f"| {key} | {value['n_pairs']} | {value['n_clusters']} | {mean_text} | {ci_text} |"
        )
    lines.extend(
        [
            "",
            "## Connected-Pair Chemistry",
            "",
            "These paired chemistry summaries use only samples for which both models produced connected reconstructed molecules. This is the like-for-like subset; means below are calculated on the same paired samples.",
            "",
            "| Metric | N pairs | N trajectories | Original mean | APLC mean | Mean difference | 95% cluster-bootstrap CI |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, value in summary["paired_connected_chemistry"].items():
        ci = value["ci95"]
        ci_text = "n/a" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        mean_text = (
            "n/a"
            if value["mean_new_minus_old"] is None
            else f"{value['mean_new_minus_old']:.4f}"
        )
        old_text = "n/a" if value["old_mean"] is None else f"{value['old_mean']:.3f}"
        new_text = "n/a" if value["new_mean"] is None else f"{value['new_mean']:.3f}"
        lines.append(
            f"| {key} | {value['n_pairs']} | {value['n_clusters']} | {old_text} | {new_text} | {mean_text} | {ci_text} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "- These are 50 samples from 30 held-out trajectories, with at most two selected frames per trajectory; frames from the same trajectory are not fully independent.",
            "- Validity, reconstruction, diversity, QED, SA, and geometric diagnostics do not establish binding affinity or activity.",
            "- The pocket clash cutoff is a screening diagnostic only. No docking engine or receptor preparation was run.",
            "- Generated compounds have no atom mapping to the MD reference ligand, so atom-wise RMSD is not a meaningful comparison.",
            "",
            "Full per-sample results and raw generated coordinates are stored beside this report.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _load_models(
    old_checkpoint: Path, new_checkpoint: Path, device: torch.device
) -> tuple[dict[str, ScorePosNet3D], dict[str, Any]]:
    checkpoints = {
        "original_targetdiff": torch.load(
            old_checkpoint, map_location="cpu", weights_only=False
        ),
        "aplc_finetuned": torch.load(
            new_checkpoint, map_location="cpu", weights_only=False
        ),
    }
    models: dict[str, ScorePosNet3D] = {}
    model_info = {}
    for label, checkpoint in checkpoints.items():
        config = checkpoint["config"]
        model = ScorePosNet3D(
            config.model,
            protein_atom_feature_dim=N_PROTEIN_FEATURES,
            ligand_atom_feature_dim=N_LIGAND_CLASSES,
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device)
        model.eval()
        models[label] = model
        model_info[label] = {
            "iteration": int(checkpoint.get("iteration", -1)),
            "num_timesteps": int(model.num_timesteps),
            "num_classes": int(model.num_classes),
            "aplc_enabled": bool(getattr(config.model, "aplc_enabled", False)),
        }
    if model_info["original_targetdiff"]["num_timesteps"] != model_info[
        "aplc_finetuned"
    ]["num_timesteps"]:
        raise ValueError("Checkpoints have different diffusion timestep counts")
    if model_info["original_targetdiff"]["num_classes"] != N_LIGAND_CLASSES:
        raise ValueError("Original checkpoint does not expose 13 ligand atom classes")
    if model_info["aplc_finetuned"]["num_classes"] != N_LIGAND_CLASSES:
        raise ValueError("Fine-tuned checkpoint does not expose 13 ligand atom classes")
    return models, model_info


def _save_molecule(path: Path, molecule: Chem.Mol | None, name: str) -> None:
    if molecule is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    molecule.SetProp("_Name", name)
    writer = Chem.SDWriter(str(path))
    writer.write(molecule)
    writer.close()


def _evaluate_output(
    output_dir: Path,
    label: str,
    ordinal: int,
    ligand_pos: np.ndarray,
    ligand_classes: np.ndarray,
    protein_pos: torch.Tensor,
    protein_elements: torch.Tensor,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    model_dir = output_dir / label
    model_dir.mkdir(parents=True, exist_ok=True)
    raw_path = model_dir / f"sample_{ordinal:03d}.npz"
    np.savez_compressed(
        raw_path,
        ligand_pos=ligand_pos,
        ligand_classes=ligand_classes,
    )
    metrics, molecule = evaluate_generated(
        ligand_pos, ligand_classes, protein_pos, protein_elements
    )
    sdf_path = model_dir / f"sample_{ordinal:03d}.sdf"
    _save_molecule(sdf_path, molecule, f"{label}_{ordinal:03d}")
    record = {
        **metadata,
        "model": label,
        "sampling_success": True,
        "evaluation_success": True,
        "raw_path": str(raw_path.relative_to(output_dir)),
        "sdf_path": str(sdf_path.relative_to(output_dir)) if molecule is not None else None,
        "metrics": metrics,
    }
    write_json(model_dir / f"sample_{ordinal:03d}.json", record)
    return record


def _load_completed_pair(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text())
        if (
            record.get("original_targetdiff", {}).get("sampling_success")
            and record.get("aplc_finetuned", {}).get("sampling_success")
            and record.get("original_targetdiff", {}).get("evaluation_success")
            and record.get("aplc_finetuned", {}).get("evaluation_success")
        ):
            return record
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _validate_output(output_dir: Path) -> dict[str, Any]:
    config_path = output_dir / "run_config.json"
    manifest_path = output_dir / "selection_manifest.json"
    summary_path = output_dir / "summary.json"
    for path in (config_path, manifest_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing validation artifact: {path}")
    config = json.loads(config_path.read_text())
    manifest_payload = json.loads(manifest_path.read_text())
    manifest = manifest_payload.get("samples", [])
    expected = int(config["num_samples"])
    pair_paths = sorted((output_dir / "pairs").glob("pair_*.json"))
    if len(pair_paths) != expected:
        raise ValueError(f"Expected {expected} pairs, found {len(pair_paths)}")
    checked = 0
    for ordinal, pair_path in enumerate(pair_paths):
        pair = json.loads(pair_path.read_text())
        if pair.get("ordinal") != ordinal:
            raise ValueError(f"Pair order mismatch in {pair_path}")
        old = pair["original_targetdiff"]
        new = pair["aplc_finetuned"]
        if not old.get("sampling_success") or not new.get("sampling_success"):
            raise ValueError(f"Sampling failure in pair {ordinal}")
        if not old.get("evaluation_success") or not new.get("evaluation_success"):
            raise ValueError(f"Evaluation failure in pair {ordinal}")
        if old["metrics"]["sampled_atom_count"] != new["metrics"]["sampled_atom_count"]:
            raise ValueError(f"Atom-count mismatch in pair {ordinal}")
        if old["paired_input_hash"] != new["paired_input_hash"]:
            raise ValueError(f"Initial condition mismatch in pair {ordinal}")
        for result in (old, new):
            raw_path = output_dir / result["raw_path"]
            if not raw_path.is_file():
                raise FileNotFoundError(raw_path)
            with np.load(raw_path) as raw:
                if not np.isfinite(raw["ligand_pos"]).all():
                    raise ValueError(f"Non-finite output coordinates in {raw_path}")
                if raw["ligand_pos"].shape[0] != result["metrics"]["sampled_atom_count"]:
                    raise ValueError(f"Atom count mismatch in {raw_path}")
            if result.get("sdf_path") and not (output_dir / result["sdf_path"]).is_file():
                raise FileNotFoundError(output_dir / result["sdf_path"])
        initial_state_path = output_dir / pair["initial_state_path"]
        if not initial_state_path.is_file():
            raise FileNotFoundError(initial_state_path)
        with np.load(initial_state_path) as initial:
            if initial["init_ligand_pos"].shape[0] != old["metrics"]["sampled_atom_count"]:
                raise ValueError(f"Initial state atom-count mismatch in {initial_state_path}")
        checked += 1
    if len(manifest) != expected:
        raise ValueError(f"Expected {expected} manifest entries, found {len(manifest)}")
    summary = json.loads(summary_path.read_text())
    return {
        "valid": True,
        "checked_pairs": checked,
        "manifest_entries": len(manifest),
        "summary_models": list(summary["models"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired 1000-step TargetDiff generation comparison."
    )
    parser.add_argument("--old-checkpoint", type=Path, default=DEFAULT_OLD_CHECKPOINT)
    parser.add_argument("--new-checkpoint", type=Path, default=DEFAULT_NEW_CHECKPOINT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--rebuild-summary", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else project_root / path

    output_dir = resolve(args.output_dir)
    if args.smoke:
        output_dir = output_dir / "smoke"
    if args.validate_only:
        print(json.dumps(_validate_output(output_dir), indent=2))
        return
    if args.rebuild_summary:
        validation = _validate_output(output_dir)
        pair_paths = sorted(
            (output_dir / "pairs").glob("pair_[0-9][0-9][0-9].json")
        )
        pair_records = [json.loads(path.read_text()) for path in pair_paths]
        manifest = json.loads(
            (output_dir / "selection_manifest.json").read_text()
        ).get("samples", [])
        config = json.loads((output_dir / "run_config.json").read_text())
        summary = build_summary(
            pair_records,
            expected_samples=int(config["num_samples"]),
            seed=int(config["seed"]),
            steps=int(config["steps"]),
        )
        write_json(output_dir / "summary.json", summary)
        write_report(output_dir / "report.md", summary, manifest)
        print(json.dumps(validation, indent=2))
        print(f"Rebuilt summary and report from {len(pair_records)} saved pairs.")
        return

    old_checkpoint = resolve(args.old_checkpoint)
    new_checkpoint = resolve(args.new_checkpoint)
    cache_path = resolve(args.cache)
    for path in (old_checkpoint, new_checkpoint, cache_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    num_samples = args.num_samples if args.num_samples is not None else (
        1 if args.smoke else 50
    )
    if not 1 <= num_samples <= 50:
        raise ValueError("--num-samples must be between 1 and 50")
    if args.steps != 1000:
        raise ValueError("This comparison is fixed to the official full 1000-step sampler")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pairs").mkdir(exist_ok=True)
    (output_dir / "old").mkdir(exist_ok=True)
    (output_dir / "new").mkdir(exist_ok=True)
    free_bytes = __import__("shutil").disk_usage(output_dir).free
    if free_bytes < 2 * 1024**3:
        raise RuntimeError(f"Less than 2 GiB free under output filesystem: {free_bytes}")

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    test_indices = [int(value) for value in cache["indices_by_split"]["test"]]
    selected_indices = select_test_indices(
        cache["records"], test_indices, num_samples=num_samples, seed=args.seed
    )
    manifest = selected_manifest_records(cache["records"], selected_indices)
    write_json(output_dir / "selection_manifest.json", {"samples": manifest})
    manifest_for_summary = manifest

    checkpoint_hashes = {
        "original_targetdiff": sha256_file(old_checkpoint),
        "aplc_finetuned": sha256_file(new_checkpoint),
    }
    run_config = {
        "num_samples": num_samples,
        "seed": args.seed,
        "steps": args.steps,
        "device": str(device),
        "cache": str(cache_path.relative_to(project_root)),
        "cache_sha256": sha256_file(cache_path),
        "checkpoints": {
            "original_targetdiff": {
                "path": str(old_checkpoint.relative_to(project_root)),
                "sha256": checkpoint_hashes["original_targetdiff"],
            },
            "aplc_finetuned": {
                "path": str(new_checkpoint.relative_to(project_root)),
                "sha256": checkpoint_hashes["aplc_finetuned"],
            },
        },
        "feature_contract": {
            "protein_atom_features": N_PROTEIN_FEATURES,
            "ligand_atom_classes": N_LIGAND_CLASSES,
        },
        "reference_ligand_fields_passed_to_model": False,
        "sampling": {
            "atom_count": "TargetDiff pocket-size prior",
            "coordinates": "protein centroid + standard normal",
            "atom_classes": "uniform over 13 classes",
            "rng_pairing": "same initialization and restored reverse-sampling RNG state",
            "steps": args.steps,
        },
    }
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        identity_keys = (
            "num_samples",
            "seed",
            "steps",
            "cache_sha256",
            "checkpoints",
        )
        differences = [
            key for key in identity_keys if previous.get(key) != run_config.get(key)
        ]
        if differences:
            raise ValueError(
                f"Existing output belongs to a different run; changed fields: {differences}"
            )
    else:
        write_json(config_path, run_config)

    RDLogger.DisableLog("rdApp.*")
    models, model_info = _load_models(old_checkpoint, new_checkpoint, device)
    if model_info["original_targetdiff"]["num_timesteps"] != args.steps:
        raise ValueError(
            f"Requested {args.steps} steps, checkpoint has "
            f"{model_info['original_targetdiff']['num_timesteps']} diffusion steps"
        )
    write_json(output_dir / "model_info.json", model_info)

    for ordinal, (sample_index, sample_manifest) in enumerate(
        zip(selected_indices, manifest_for_summary)
    ):
        pair_path = output_dir / "pairs" / f"pair_{ordinal:03d}.json"
        completed = _load_completed_pair(pair_path)
        if completed is not None:
            print(f"[{ordinal + 1}/{num_samples}] resume: pair already complete")
            continue

        sample = cache["samples"][sample_index]
        protein_pos, protein_features, protein_elements = protein_only(sample)
        pair_seed = int(args.seed + ordinal * 100_003)
        atom_count, init_pos, init_types, condition = prepare_pair_condition(
            protein_pos=protein_pos,
            num_classes=model_info["original_targetdiff"]["num_classes"],
            seed=pair_seed,
            device=device,
        )
        initial_npz = output_dir / "pairs" / f"pair_{ordinal:03d}_initial.npz"
        np.savez_compressed(
            initial_npz,
            init_ligand_pos=init_pos.detach().cpu().numpy().astype(np.float32),
            init_ligand_classes=init_types.detach().cpu().numpy().astype(np.int64),
        )
        protein_hash = sha256_arrays(
            protein_pos.numpy(), protein_features.numpy(), protein_elements.numpy()
        )
        common_metadata = {
            "ordinal": ordinal,
            "sample_index": sample_index,
            "trajectory_id": sample_manifest["trajectory_id"],
            "pdb_id": sample_manifest["pdb_id"],
            "cluster_id": sample_manifest["cluster_id"],
            "cluster_frame": sample_manifest["cluster_frame"],
            "pair_seed": pair_seed,
            "paired_input_hash": condition["input_hash"],
            "protein_input_hash": protein_hash,
            "sampled_atom_count": atom_count,
            "diffusion_steps": args.steps,
        }
        pair_result: dict[str, Any] = {
            **sample_manifest,
            "ordinal": ordinal,
            "pair_seed": pair_seed,
            "paired_input_hash": condition["input_hash"],
            "initial_state_path": str(initial_npz.relative_to(output_dir)),
        }
        for label, model in models.items():
            _set_rng_state(condition["rng_state"], device)
            start = time.perf_counter()
            try:
                ligand_pos, ligand_classes = sample_one(
                    model=model,
                    protein_pos=protein_pos,
                    protein_features=protein_features,
                    init_pos=init_pos,
                    init_types=init_types,
                    num_steps=args.steps,
                    device=device,
                )
            except Exception as exc:
                pair_result[label] = {
                    **common_metadata,
                    "sampling_success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                continue

            sampling_seconds = time.perf_counter() - start
            output_label = "old" if label == "original_targetdiff" else "new"
            result_metadata = {
                **common_metadata,
                "sampling_seconds": float(sampling_seconds),
                "paired_input_hash": condition["input_hash"],
            }
            try:
                pair_result[label] = _evaluate_output(
                    output_dir=output_dir,
                    label=output_label,
                    ordinal=ordinal,
                    ligand_pos=ligand_pos,
                    ligand_classes=ligand_classes,
                    protein_pos=protein_pos,
                    protein_elements=protein_elements,
                    metadata=result_metadata,
                )
            except Exception as exc:
                raw_path = output_dir / output_label / f"sample_{ordinal:03d}.npz"
                np.savez_compressed(
                    raw_path,
                    ligand_pos=ligand_pos,
                    ligand_classes=ligand_classes,
                )
                pair_result[label] = {
                    **result_metadata,
                    "model": output_label,
                    "sampling_success": True,
                    "evaluation_success": False,
                    "raw_path": str(raw_path.relative_to(output_dir)),
                    "evaluation_error": f"{type(exc).__name__}: {exc}",
                }
            if pair_result[label].get("sampling_success"):
                pair_result[label]["paired_input_hash"] = condition["input_hash"]
        write_json(pair_path, pair_result)
        status = {
            key: (
                "ok"
                if value.get("sampling_success") and value.get("evaluation_success")
                else "failed"
            )
            for key, value in pair_result.items()
            if key in ("original_targetdiff", "aplc_finetuned")
        }
        print(
            f"[{ordinal + 1}/{num_samples}] sample={sample_index} "
            f"atoms={atom_count} old={status.get('original_targetdiff')} "
            f"new={status.get('aplc_finetuned')}",
            flush=True,
        )
        if not all(value == "ok" for value in status.values()):
            raise RuntimeError(f"Sampling failed for pair {ordinal}: {status}")

        all_pairs = []
        for current_path in sorted((output_dir / "pairs").glob("pair_*.json")):
            if "_initial" in current_path.stem:
                continue
            try:
                all_pairs.append(json.loads(current_path.read_text()))
            except json.JSONDecodeError:
                continue
        # Convert stored aliases used by the output tree into stable summary keys.
        normalized_pairs = []
        for pair in all_pairs:
            normalized = dict(pair)
            for source, target in (("old", "original_targetdiff"), ("new", "aplc_finetuned")):
                if source in normalized:
                    normalized[target] = normalized.pop(source)
            normalized_pairs.append(normalized)
        summary = build_summary(
            normalized_pairs,
            expected_samples=num_samples,
            seed=args.seed,
            steps=args.steps,
        )
        write_json(output_dir / "summary.json", summary)
        write_report(output_dir / "report.md", summary, manifest_for_summary)

    pair_files = sorted((output_dir / "pairs").glob("pair_[0-9][0-9][0-9].json"))
    final_pairs = [json.loads(path.read_text()) for path in pair_files]
    normalized_pairs = []
    for pair in final_pairs:
        normalized = dict(pair)
        for source, target in (("old", "original_targetdiff"), ("new", "aplc_finetuned")):
            if source in normalized:
                normalized[target] = normalized.pop(source)
        normalized_pairs.append(normalized)
    final_summary = build_summary(
        normalized_pairs,
        expected_samples=num_samples,
        seed=args.seed,
        steps=args.steps,
    )
    write_json(output_dir / "summary.json", final_summary)
    write_report(output_dir / "report.md", final_summary, manifest_for_summary)
    print(json.dumps(_validate_output(output_dir), indent=2))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
