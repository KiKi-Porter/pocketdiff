"""Reusable training and inference pipeline for the independent PocketDiff diffusion core."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pickle
import random
import math

import torch

from pocketdiff.data.apo2mol_adapter import AdapterError, Apo2MolAdapter
from pocketdiff.data.schema import PocketComplex
from pocketdiff.diffusion import (
    build_diffusion_state_from_current,
    collate_diffusion_states,
    DiffusionMotionAdapter,
    ReverseTrajectory,
    diffusion_motion_loss,
    sample_diffusion_state,
    sample_reverse_trajectory,
)
from pocketdiff.geometry.chi import apply_chi_updates, build_chi_update_metadata
from pocketdiff.geometry.current_state import build_current_chi_state
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.diffusion.se3 import apply_local_se3_update
from pocketdiff.geometry.oracle import oracle_rigid_chi_reconstruction


DEFAULT_DATA_ROOT = Path("Apo2Mol-main/Apo2MOl-dataset/data_folder")
DEFAULT_SPLIT_PATH = DEFAULT_DATA_ROOT.parent / "split_druglike_dict.pkl"
DEFAULT_TIMES = (0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _move_complex_to_device(
    value: PocketComplex,
    device: torch.device,
) -> PocketComplex:
    kwargs = {}
    for field in value.__dataclass_fields__.values():
        item = getattr(value, field.name)
        kwargs[field.name] = item.to(device) if isinstance(item, torch.Tensor) else item
    return PocketComplex(**kwargs)


def _generator_for_device(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device.type).manual_seed(int(seed))


@dataclass(frozen=True)
class DiffusionTrainConfig:
    """Configuration for a finite, auditable PocketDiff training run."""

    data_root: str = str(DEFAULT_DATA_ROOT)
    split_path: str = str(DEFAULT_SPLIT_PATH)
    source_cache: Optional[str] = None
    prepared_cache: Optional[str] = None
    device: str = "cpu"
    train_split: str = "train"
    valid_split: str = "valid"
    test_split: str = "test"
    holdout_split: str = "valid"
    train_count: int = 24
    valid_count: Optional[int] = None
    test_count: int = 0
    holdout_count: int = 16
    train_seed: int = 5700
    holdout_seed: int = 5701
    diffusion_seed: int = 5700
    updates: int = 480
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    self_state_probability: float = 0.0
    self_state_steps: int = 1
    self_state_rollout_steps: int = 8
    self_state_warmup: int = 0
    apo_boundary_probability: float = 0.25
    cache_manifest: Optional[str] = None
    verify_cache_sources: bool = True
    times: Tuple[float, ...] = DEFAULT_TIMES
    time_min: float = 0.02
    time_max: float = 1.0
    learning_rate: float = 2.0e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    hidden_dim: int = 128
    dropout: float = 0.0
    encoder_backend: str = "scalar"
    architecture_variant: str = "v47"
    endpoint_weight: float = 1.0
    next_state_weight: float = 0.0
    next_state_steps: int = 20
    chi_loss_weight: float = 1.0
    backbone_only: bool = False
    use_chi: bool = True
    score_normalize: bool = False
    score_floor: float = 0.1
    motion_parameterization: str = "remaining"
    sampler_steps: Tuple[int, ...] = (4, 8)
    remaining_step_offset: float = 0.0
    translation_noise_scale: float = 0.10
    rotation_noise_scale: float = 0.05
    chi_noise_scale: float = 0.05
    prediction_type: str = "bridge_velocity"
    target_parameterization: str = "bridge_velocity"
    target_steps: int = 20
    backbone_endpoint_weight: float = 2.0
    continuity_weight: float = 0.10
    direction_weight: float = 0.0
    direction_threshold: float = 0.05
    backbone_progress_direction_weight: float = 0.0
    backbone_progress_magnitude_weight: float = 0.0
    backbone_progress_threshold: float = 0.05
    motion_bucket_count: int = 4
    motion_bucket_balance: bool = True

    def __post_init__(self) -> None:
        _resolve_device(self.device)
        valid_count = self.holdout_count if self.valid_count is None else self.valid_count
        if self.train_count <= 0 or valid_count < 0 or self.test_count < 0:
            raise ValueError("train_count must be positive and holdout_count non-negative")
        if self.updates <= 0:
            raise ValueError("updates must be positive")
        if self.batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("batch_size and gradient_accumulation_steps must be positive")
        if not 0.0 <= self.self_state_probability <= 1.0:
            raise ValueError("self_state_probability must lie in [0, 1]")
        if not 0.0 <= self.apo_boundary_probability <= 1.0:
            raise ValueError("apo_boundary_probability must lie in [0, 1]")
        if self.self_state_steps < 0 or self.self_state_rollout_steps <= 0:
            raise ValueError("self_state_steps must be non-negative and rollout positive")
        if self.self_state_steps > self.self_state_rollout_steps:
            raise ValueError("self_state_steps cannot exceed self_state_rollout_steps")
        if self.self_state_warmup < 0:
            raise ValueError("self_state_warmup must be non-negative")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.dropout < 0.0:
            raise ValueError("dropout must be non-negative")
        if self.encoder_backend not in ("scalar", "targetdiff", "dynamicbind"):
            raise ValueError(
                "encoder_backend must be scalar, targetdiff, or dynamicbind"
            )
        if self.encoder_backend in ("targetdiff", "dynamicbind") and self.hidden_dim != 128:
            raise ValueError(
                "targetdiff and dynamicbind encoder backends require hidden_dim=128"
            )
        if not self.times or any(not 0.0 <= float(t) <= 1.0 for t in self.times):
            raise ValueError("times must be non-empty values in [0, 1]")
        if not 0.0 < self.time_min < self.time_max <= 1.0:
            raise ValueError("time_min/time_max must satisfy 0 < min < max <= 1")
        if any(int(step) <= 0 for step in self.sampler_steps):
            raise ValueError("sampler_steps must contain positive integers")
        if self.motion_parameterization not in ("remaining", "bridge_rate"):
            raise ValueError("motion_parameterization must be remaining or bridge_rate")
        if self.architecture_variant not in ("v45", "v46", "v47"):
            raise ValueError("architecture_variant must be v45, v46, or v47")
        if self.prediction_type not in ("remaining", "velocity", "bridge_velocity"):
            raise ValueError(
                "prediction_type must be remaining, velocity, or bridge_velocity"
            )
        if self.target_parameterization not in (
            "remaining",
            "velocity",
            "bridge_velocity",
        ):
            raise ValueError("unsupported target_parameterization")
        if self.prediction_type != self.target_parameterization:
            raise ValueError(
                "prediction_type and target_parameterization must match"
            )
        if self.target_steps <= 0:
            raise ValueError("target_steps must be positive")
        if self.prediction_type == "bridge_velocity" and self.motion_parameterization != "remaining":
            raise ValueError(
                "bridge_velocity requires motion_parameterization='remaining'"
            )
        if self.backbone_endpoint_weight < 0.0 or self.continuity_weight < 0.0:
            raise ValueError("endpoint auxiliary weights must be non-negative")
        if (
            self.direction_weight < 0.0
            or self.direction_threshold < 0.0
            or self.backbone_progress_direction_weight < 0.0
            or self.backbone_progress_magnitude_weight < 0.0
            or self.backbone_progress_threshold < 0.0
            or self.next_state_weight < 0.0
            or self.chi_loss_weight < 0.0
        ):
            raise ValueError("direction settings must be non-negative")
        if self.next_state_steps <= 0:
            raise ValueError("next_state_steps must be positive")
        if self.backbone_only and self.use_chi:
            raise ValueError("backbone_only training cannot enable use_chi")
        if self.motion_bucket_count <= 0:
            raise ValueError("motion_bucket_count must be positive")


@dataclass(frozen=True)
class SelectedComplexes:
    values: List[PocketComplex]
    indices: List[int]
    rejected: List[Dict[str, object]]

    @property
    def sample_ids(self) -> List[str]:
        return [value.sample_id for value in self.values]


@dataclass(frozen=True)
class LoadedDiffusionCheckpoint:
    model: DiffusionMotionAdapter
    payload: Dict[str, object]


def config_to_dict(config: DiffusionTrainConfig) -> Dict[str, object]:
    result = asdict(config)
    result["times"] = list(config.times)
    result["sampler_steps"] = list(config.sampler_steps)
    return result


def _select_cached_complexes(
    manifest_path: Path,
    *,
    split: str,
    count: int,
    verify_sources: bool,
) -> SelectedComplexes:
    from pocketdiff.preprocessing.cache import load_manifest, load_sample_cache

    manifest = load_manifest(manifest_path)
    entries = [entry for entry in manifest["entries"] if entry.get("split") == split]
    if count < 0:
        raise ValueError("count must be non-negative")
    if count > len(entries):
        raise RuntimeError(
            f"cache manifest has {len(entries)} entries for split {split!r}, "
            f"but {count} were requested"
        )
    values: List[PocketComplex] = []
    indices: List[int] = []
    rejected: List[Dict[str, object]] = []
    for index, entry in enumerate(entries[:count]):
        try:
            cache_path = Path(str(entry["cache_path"]))
            if not cache_path.is_absolute():
                candidate = manifest_path.parent / cache_path
                cache_path = candidate if candidate.is_file() else cache_path
            cached = load_sample_cache(cache_path, verify_sources=verify_sources)
            values.append(cached.complex_value)
            indices.append(index)
        except Exception as exc:
            rejected.append(
                {
                    "index": index,
                    "sample_id": entry.get("sample_id"),
                    "reason_code": "cache_error",
                    "error": str(exc),
                }
            )
    if len(values) != count:
        raise RuntimeError(
            f"could not load {count} cached samples for split {split!r}; got {len(values)}"
        )
    return SelectedComplexes(values=values, indices=indices, rejected=rejected)


def load_split_records(split_path: Path, split_name: str) -> List[object]:
    with Path(split_path).open("rb") as handle:
        split = pickle.load(handle)
    if split_name not in split:
        raise KeyError(f"split {split_name!r} not found in {split_path}")
    return list(split[split_name])


def select_convertible_complexes(
    adapter: Apo2MolAdapter,
    records: Sequence[object],
    *,
    seed: int,
    count: int,
) -> SelectedComplexes:
    """Pick a deterministic convertible slice and record rejected raw rows."""

    if count < 0:
        raise ValueError("count must be non-negative")
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    values: List[PocketComplex] = []
    selected: List[int] = []
    rejected: List[Dict[str, object]] = []
    for index in indices:
        try:
            value = adapter.convert_record(records[index])
        except AdapterError as exc:
            sample_id = None
            try:
                sample_id = Path(str(records[index][0])).parent.name
            except Exception:
                sample_id = str(index)
            rejected.append(
                {
                    "index": index,
                    "sample_id": sample_id,
                    "reason_code": getattr(exc, "reason_code", "adapter_error"),
                    "error": str(exc),
                }
            )
            continue
        values.append(value)
        selected.append(index)
        if len(values) == count:
            break
    if len(values) != count:
        raise RuntimeError(f"could not collect {count} convertible samples; got {len(values)}")
    return SelectedComplexes(values=values, indices=selected, rejected=rejected)


def load_apo2mol_slices(config: DiffusionTrainConfig) -> Dict[str, SelectedComplexes]:
    valid_count = config.holdout_count if config.valid_count is None else config.valid_count
    if config.prepared_cache:
        return load_prepared_diffusion_slices(Path(config.prepared_cache), config)
    if config.source_cache:
        payload = torch.load(Path(config.source_cache), map_location="cpu", weights_only=False)
        if payload.get("format") != "pocketdiff-medium-cache-v1":
            raise ValueError("source_cache must be a pocketdiff-medium-cache-v1 cache")
        result = {}
        for split, count, seed in (
            (config.train_split, config.train_count, config.train_seed),
            (config.valid_split, valid_count, config.holdout_seed),
            (config.test_split, config.test_count, config.holdout_seed + 1),
        ):
            values = list(payload["values"].get(split, []))
            if count > len(values):
                raise RuntimeError(
                    f"source_cache has {len(values)} {split} samples, requested {count}"
                )
            order = list(range(len(values)))
            random.Random(seed).shuffle(order)
            chosen = [values[index] for index in order[:count]]
            result[split] = SelectedComplexes(chosen, order[:count], [])
        result["holdout"] = result[config.valid_split]
        _assert_disjoint_slices(result)
        return result
    if config.cache_manifest:
        manifest_path = Path(config.cache_manifest)
        train = _select_cached_complexes(
            manifest_path,
            split=config.train_split,
            count=config.train_count,
            verify_sources=config.verify_cache_sources,
        )
        valid = _select_cached_complexes(
            manifest_path,
            split=config.valid_split,
            count=valid_count,
            verify_sources=config.verify_cache_sources,
        )
        test = _select_cached_complexes(
            manifest_path,
            split=config.test_split,
            count=config.test_count,
            verify_sources=config.verify_cache_sources,
        ) if config.test_count else SelectedComplexes([], [], [])
        slices = {"train": train, "valid": valid, "holdout": valid, "test": test}
        _assert_disjoint_slices(slices)
        return slices
    adapter = Apo2MolAdapter(Path(config.data_root))
    train_records = load_split_records(Path(config.split_path), config.train_split)
    train = select_convertible_complexes(
        adapter,
        train_records,
        seed=config.train_seed,
        count=config.train_count,
    )
    if valid_count:
        valid_records = load_split_records(Path(config.split_path), config.valid_split)
        valid = select_convertible_complexes(
            adapter,
            valid_records,
            seed=config.holdout_seed,
            count=valid_count,
        )
    else:
        valid = SelectedComplexes(values=[], indices=[], rejected=[])
    if config.test_count:
        test_records = load_split_records(Path(config.split_path), config.test_split)
        test = select_convertible_complexes(
            adapter,
            test_records,
            seed=config.holdout_seed + 1,
            count=config.test_count,
        )
    else:
        test = SelectedComplexes(values=[], indices=[], rejected=[])
    slices = {"train": train, "valid": valid, "holdout": valid, "test": test}
    _assert_disjoint_slices(slices)
    return slices


def load_prepared_diffusion_slices(
    cache_path: Path,
    config: DiffusionTrainConfig,
) -> Dict[str, SelectedComplexes]:
    """Load the v4.5 CPU cache without rerunning adapter preprocessing."""

    from pocketdiff.preprocessing.diffusion_cache import load_diffusion_cache

    payload = load_diffusion_cache(Path(cache_path))
    result: Dict[str, SelectedComplexes] = {}
    valid_count = config.holdout_count if config.valid_count is None else config.valid_count
    for split, count, seed in (
        (config.train_split, config.train_count, config.train_seed),
        (config.valid_split, valid_count, config.holdout_seed),
        (config.test_split, config.test_count, config.holdout_seed + 1),
    ):
        all_values = list(payload["splits"].get(split, []))
        # Some legacy prepared caches contain pocket records whose N-CA-C
        # frame is unavailable for every residue. They cannot provide a
        # rigid-motion supervision signal and must not enter a training batch.
        values = [value for value in all_values if bool(value.frame_valid.any())]
        if count > len(values):
            raise RuntimeError(
                f"prepared cache has {len(values)} usable {split} samples "
                f"({len(all_values)} total), requested {count}"
            )
        order = list(range(len(values)))
        random.Random(seed).shuffle(order)
        chosen = [values[index] for index in order[:count]]
        result[split] = SelectedComplexes(chosen, order[:count], [])
    result["holdout"] = result[config.valid_split]
    _assert_disjoint_slices(result)
    return result


def _assert_disjoint_slices(slices: Dict[str, SelectedComplexes]) -> None:
    seen: Dict[str, str] = {}
    for split in ("train", "valid", "test"):
        for sample_id in slices[split].sample_ids:
            previous = seen.get(sample_id)
            if previous is not None:
                raise ValueError(
                    f"sample {sample_id!r} appears in both {previous} and {split}"
                )
            seen[sample_id] = split


def coordinate_rmsd(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.shape != second.shape:
        raise ValueError(f"RMSD tensors must have same shape, got {first.shape} and {second.shape}")
    return float(torch.sqrt((first - second).square().sum(dim=-1).mean()))


def masked_coordinate_rmsd(
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    if first.shape != second.shape:
        raise ValueError("RMSD tensors must have identical shapes")
    if mask.shape != (first.shape[0],) or mask.dtype != torch.bool:
        raise ValueError("RMSD mask must be BoolTensor [num_atoms]")
    if not bool(mask.any()):
        return float("nan")
    return coordinate_rmsd(first[mask], second[mask])


def _atom_mask(value: PocketComplex, names: set[str]) -> torch.Tensor:
    return torch.tensor(
        [name in names for name in value.protein_atom_name],
        dtype=torch.bool,
        device=value.protein_pos_apo.device,
    )


def _oracle_rigid_backbone_rmsd(value: PocketComplex) -> float:
    oracle = oracle_rigid_chi_reconstruction(
        value.protein_pos_apo,
        value.protein_pos_holo,
        value.atom_to_residue,
        value.protein_atom_name,
        [
            value.protein_residue_name[
                next(
                    index
                    for index, residue_id in enumerate(
                        value.atom_to_residue.detach().cpu().tolist()
                    )
                    if residue_id == residue_index
                )
            ]
            for residue_index in range(value.num_residues)
        ],
        value.frame_valid,
    )
    return float(oracle.rigid_metrics.backbone_rmsd)


def _backbone_motion_score(value: PocketComplex) -> float:
    mask = torch.tensor(
        [name in {"N", "CA", "C", "O"} for name in value.protein_atom_name],
        dtype=torch.bool,
        device=value.protein_pos_apo.device,
    )
    if not bool(mask.any()):
        return 0.0
    return coordinate_rmsd(
        value.protein_pos_apo[mask],
        value.protein_pos_holo[mask],
    )


def _trajectory_motion_metrics(
    value: PocketComplex,
    positions: torch.Tensor,
) -> Dict[str, float]:
    """Measure whether a trajectory state moves toward the holo backbone."""

    backbone_mask = _atom_mask(value, {"N", "CA", "C", "O"})
    apo = value.protein_pos_apo[backbone_mask]
    holo = value.protein_pos_holo[backbone_mask]
    current = positions[backbone_mask]
    predicted_delta = current - apo
    target_delta = holo - apo
    target_flat = target_delta.reshape(-1)
    predicted_flat = predicted_delta.reshape(-1)
    target_norm_sq = torch.sum(target_flat.square())
    predicted_norm = torch.linalg.vector_norm(predicted_flat)
    target_norm = torch.linalg.vector_norm(target_flat)
    if float(target_norm_sq) > 1.0e-12:
        projection = torch.sum(predicted_flat * target_flat) / target_norm_sq
        residual = predicted_flat - projection * target_flat
        projection_value = float(projection)
        orthogonal_value = float(torch.linalg.vector_norm(residual))
    else:
        projection_value = float("nan")
        orthogonal_value = float("nan")
    if float(predicted_norm) > 1.0e-8 and float(target_norm) > 1.0e-8:
        direction_cosine = float(
            torch.sum(predicted_flat * target_flat)
            / (predicted_norm * target_norm)
        )
    else:
        direction_cosine = float("nan")

    residue_atoms: Dict[int, Dict[str, int]] = {}
    for atom_index, (residue_id, atom_name) in enumerate(
        zip(
            value.atom_to_residue.detach().cpu().tolist(),
            value.protein_atom_name,
        )
    ):
        if atom_name in {"N", "CA", "C"}:
            residue_atoms.setdefault(int(residue_id), {})[atom_name] = atom_index
    ca_errors = []
    peptide_errors = []
    chain_ids = list(value.residue_chain_id)
    for residue_id in range(value.num_residues - 1):
        if chain_ids[residue_id] != chain_ids[residue_id + 1]:
            continue
        left = residue_atoms.get(residue_id, {})
        right = residue_atoms.get(residue_id + 1, {})
        if "CA" in left and "CA" in right:
            pred_distance = torch.linalg.vector_norm(
                positions[left["CA"]] - positions[right["CA"]]
            )
            holo_distance = torch.linalg.vector_norm(
                value.protein_pos_holo[left["CA"]]
                - value.protein_pos_holo[right["CA"]]
            )
            ca_errors.append(torch.abs(pred_distance - holo_distance))
        if "C" in left and "N" in right:
            pred_distance = torch.linalg.vector_norm(
                positions[left["C"]] - positions[right["N"]]
            )
            holo_distance = torch.linalg.vector_norm(
                value.protein_pos_holo[left["C"]]
                - value.protein_pos_holo[right["N"]]
            )
            peptide_errors.append(torch.abs(pred_distance - holo_distance))
    return {
        "holo_rmsd": coordinate_rmsd(positions, value.protein_pos_holo),
        "ca_rmsd": masked_coordinate_rmsd(
            positions,
            value.protein_pos_holo,
            _atom_mask(value, {"CA"}),
        ),
        "backbone_rmsd": coordinate_rmsd(current, holo),
        "apo_distance_rmsd": coordinate_rmsd(positions, value.protein_pos_apo),
        "direction_cosine": direction_cosine,
        "projection_coefficient": projection_value,
        "orthogonal_error": orthogonal_value,
        "mean_ca_ca_error": (
            float(torch.stack(ca_errors).mean()) if ca_errors else float("nan")
        ),
        "mean_peptide_bond_error": (
            float(torch.stack(peptide_errors).mean())
            if peptide_errors
            else float("nan")
        ),
    }


def _build_motion_buckets(
    values: Sequence[PocketComplex],
    bucket_count: int,
) -> Tuple[List[List[int]], List[float]]:
    if not values:
        return [], []
    scores = [_backbone_motion_score(value) for value in values]
    order = sorted(range(len(values)), key=lambda index: (scores[index], index))
    bucket_count = min(int(bucket_count), len(values))
    buckets = [[] for _ in range(bucket_count)]
    for rank, index in enumerate(order):
        bucket_id = min(bucket_count - 1, (rank * bucket_count) // len(values))
        buckets[bucket_id].append(index)
    return buckets, scores


@torch.no_grad()
def _sample_batched_reverse_trajectory(
    model: DiffusionMotionAdapter,
    states: Sequence,
    *,
    steps: int,
    remaining_step_offset: float,
    motion_parameterization: str,
    prediction_type: str,
    use_chi: bool,
) -> Tuple[Tuple[torch.Tensor, ...], ...]:
    """Run one deterministic trajectory for several complexes at once.

    The model already supports disjoint batched graphs.  The old evaluator
    ignored that contract and invoked the TargetDiff encoder once per sample
    and per step.  This helper keeps the same Euler/SE(3) update while making
    one model call for the whole batch at each step.
    """

    if not states:
        raise ValueError("at least one state is required")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if remaining_step_offset < 0.0 or not math.isfinite(float(remaining_step_offset)):
        raise ValueError("remaining_step_offset must be finite and non-negative")
    if motion_parameterization not in ("remaining", "bridge_rate"):
        raise ValueError("motion_parameterization must be remaining or bridge_rate")
    if prediction_type not in ("remaining", "velocity", "bridge_velocity"):
        raise ValueError("unsupported prediction_type")

    batched_input, _ = collate_diffusion_states(tuple(states))
    current = batched_input.protein_pos.detach().clone()
    current_chi = batched_input.chi_current.detach().clone()
    chi_metadata = build_chi_update_metadata(
        batched_input.protein_atom_name,
        batched_input.atom_to_residue,
        batched_input.protein_residue_name,
    )
    atom_counts = [
        int(state.model_input.protein_pos.shape[0]) for state in states
    ]
    atom_offsets = [0]
    for count in atom_counts:
        atom_offsets.append(atom_offsets[-1] + count)

    trajectories = [[
        current[atom_offsets[index] : atom_offsets[index + 1]].clone()
    ] for index in range(len(states))]
    times = torch.full(
        (len(states),),
        1.0,
        dtype=current.dtype,
        device=current.device,
    )
    from dataclasses import replace

    for index in range(steps):
        current_t = 1.0 - float(index) / float(steps)
        next_t = 1.0 - float(index + 1) / float(steps)
        inp = replace(
            batched_input,
            protein_pos=current,
            chi_current=current_chi,
            diffusion_time=times.fill_(current_t),
        )
        prediction = model(inp)
        frames = build_residue_frames(
            current,
            inp.atom_to_residue,
            inp.protein_atom_name,
            num_residues=inp.residue_type.shape[0],
            atom_indices=inp.frame_atom_indices,
        )
        remaining_steps = steps - index
        if prediction_type in ("velocity", "bridge_velocity"):
            fraction = 1.0 / float(steps)
        elif motion_parameterization == "bridge_rate":
            fraction = 1.0 / float(steps + remaining_step_offset)
        else:
            fraction = 1.0 / float(remaining_steps + remaining_step_offset)
        current = apply_local_se3_update(
            current,
            inp.atom_to_residue,
            frames.origins,
            frames.frames,
            prediction.translation_local,
            prediction.rotation_local,
            fraction=fraction,
            frame_valid=inp.frame_valid & frames.valid,
        )
        if use_chi:
            chi_valid = (
                inp.chi_mask
                & chi_metadata.valid
                & inp.frame_valid[:, None]
            )
            chi_update = apply_chi_updates(
                current,
                chi_metadata.axis_start,
                chi_metadata.axis_end,
                chi_metadata.downstream_atom_mask,
                prediction.chi * fraction,
                valid=chi_valid,
            )
            current = chi_update.positions
            current_chi = torch.atan2(
                torch.sin(current_chi + chi_update.applied_chi),
                torch.cos(current_chi + chi_update.applied_chi),
            )
        for graph_index in range(len(states)):
            trajectories[graph_index].append(
                current[
                    atom_offsets[graph_index] : atom_offsets[graph_index + 1]
                ].clone()
            )
    return tuple(tuple(item) for item in trajectories)


def summarize_diffusion_rows(
    rows: Sequence[Dict[str, object]],
    *,
    step_count: int,
    remaining_step_offset: float,
    motion_parameterization: str,
    prediction_type: str,
) -> Dict[str, object]:
    """Aggregate one fixed-step evaluation from serialized sample rows."""

    if not rows:
        raise ValueError("cannot summarize an empty evaluation")
    key = str(int(step_count))
    final_values = [row["trajectories"][key]["holo_rmsd"] for row in rows]
    baseline_values = [row["apo_rmsd"] for row in rows]
    final_ca_values = [row["trajectories"][key]["ca_rmsd"] for row in rows]
    final_backbone_values = [
        row["trajectories"][key]["backbone_rmsd"] for row in rows
    ]
    final_direction_values = [
        row["trajectories"][key]["direction_cosine"] for row in rows
    ]
    final_projection_values = [
        row["trajectories"][key]["projection_coefficient"] for row in rows
    ]
    final_orthogonal_values = [
        row["trajectories"][key]["orthogonal_error"] for row in rows
    ]
    final_ca_geometry_values = [
        row["trajectories"][key]["mean_ca_ca_error"] for row in rows
    ]
    final_peptide_geometry_values = [
        row["trajectories"][key]["mean_peptide_bond_error"] for row in rows
    ]

    def _finite_mean(values: Sequence[float]) -> float:
        finite = [float(value) for value in values if math.isfinite(float(value))]
        return float(sum(finite) / len(finite)) if finite else float("nan")

    baseline_ca_values = [row["apo_ca_rmsd"] for row in rows]
    baseline_backbone_values = [row["apo_backbone_rmsd"] for row in rows]
    oracle_backbone_values = [row["rigid_oracle_backbone_rmsd"] for row in rows]
    return {
        "rows": list(rows),
        "remaining_step_offset": remaining_step_offset,
        "motion_parameterization": motion_parameterization,
        "prediction_type": prediction_type,
        "mean_apo_rmsd": float(sum(baseline_values) / len(baseline_values)),
        "mean_final_holo_rmsd": float(sum(final_values) / len(final_values)),
        "mean_apo_ca_rmsd": float(sum(baseline_ca_values) / len(baseline_ca_values)),
        "mean_final_ca_rmsd": float(sum(final_ca_values) / len(final_ca_values)),
        "mean_apo_backbone_rmsd": float(
            sum(baseline_backbone_values) / len(baseline_backbone_values)
        ),
        "mean_final_backbone_rmsd": float(
            sum(final_backbone_values) / len(final_backbone_values)
        ),
        "mean_final_direction_cosine": _finite_mean(final_direction_values),
        "mean_final_projection_coefficient": _finite_mean(final_projection_values),
        "mean_final_orthogonal_error": _finite_mean(final_orthogonal_values),
        "mean_final_ca_ca_error": _finite_mean(final_ca_geometry_values),
        "mean_final_peptide_bond_error": _finite_mean(
            final_peptide_geometry_values
        ),
        "final_backbone_improved_fraction": float(
            sum(
                final < baseline
                for final, baseline in zip(
                    final_backbone_values, baseline_backbone_values
                )
            )
            / len(rows)
        ),
        "mean_rigid_oracle_backbone_rmsd": float(
            sum(oracle_backbone_values) / len(oracle_backbone_values)
        ),
        "all_start_matches_apo": all(row["start_matches_apo"] for row in rows),
        "all_trajectories_finite": all(
            trajectory["finite"]
            for row in rows
            for trajectory in row["trajectories"].values()
        ),
    }


@torch.no_grad()
def evaluate_diffusion_sampler(
    model: DiffusionMotionAdapter,
    values: Sequence[PocketComplex],
    *,
    steps: Sequence[int] = (8,),
    remaining_step_offset: float = 0.0,
    motion_parameterization: str = "remaining",
    prediction_type: str = "velocity",
    ligand_conditioner=None,
    ligand_condition_seed: Optional[int] = None,
    batch_size: int = 8,
    progress_label: str = "",
) -> Dict[str, object]:
    """Evaluate deterministic apo-start reverse trajectories for a sample list."""

    if not values:
        return {
            "rows": [],
            "remaining_step_offset": remaining_step_offset,
            "motion_parameterization": motion_parameterization,
            "prediction_type": prediction_type,
            "mean_apo_rmsd": 0.0,
            "mean_final_holo_rmsd": 0.0,
            "all_start_matches_apo": True,
            "all_trajectories_finite": True,
        }
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model_was_training = model.training
    model.eval()
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    rows = []
    try:
        use_chi = bool(getattr(model, "_pocketdiff_use_chi", True))
        for batch_start in range(0, len(values), batch_size):
            raw_batch = values[batch_start : batch_start + batch_size]
            device_values = [
                _move_complex_to_device(raw_value, model_device)
                for raw_value in raw_batch
            ]
            states = [
                sample_diffusion_state(
                    value,
                    1.0,
                    translation_noise_scale=0.0,
                    rotation_noise_scale=0.0,
                    chi_noise_scale=0.0,
                    ligand_conditioner=ligand_conditioner,
                    generator=(
                        _generator_for_device(
                            model_device,
                            int(ligand_condition_seed) + batch_start + offset,
                        )
                        if ligand_condition_seed is not None
                        else None
                    ),
                )
                for offset, value in enumerate(device_values)
            ]
            row_batch = []
            for value, state in zip(device_values, states):
                start = state.model_input.protein_pos
                row_batch.append(
                    {
                        "sample_id": value.sample_id,
                        "apo_rmsd": coordinate_rmsd(
                            value.protein_pos_apo, value.protein_pos_holo
                        ),
                        "apo_ca_rmsd": masked_coordinate_rmsd(
                            value.protein_pos_apo,
                            value.protein_pos_holo,
                            _atom_mask(value, {"CA"}),
                        ),
                        "apo_backbone_rmsd": masked_coordinate_rmsd(
                            value.protein_pos_apo,
                            value.protein_pos_holo,
                            _atom_mask(value, {"N", "CA", "C", "O"}),
                        ),
                        "rigid_oracle_backbone_rmsd": _oracle_rigid_backbone_rmsd(
                            value
                        ),
                        "start_matches_apo": bool(
                            torch.equal(start, value.protein_pos_apo)
                        ),
                        "trajectories": {},
                    }
                )
            for step_count in steps:
                if ligand_conditioner is None:
                    batched_trajectories = _sample_batched_reverse_trajectory(
                        model,
                        states,
                        steps=int(step_count),
                        remaining_step_offset=remaining_step_offset,
                        motion_parameterization=motion_parameterization,
                        prediction_type=prediction_type,
                        use_chi=use_chi,
                    )
                else:
                    batched_trajectories = tuple(
                        sample_reverse_trajectory(
                            model,
                            state,
                            steps=int(step_count),
                            remaining_step_offset=remaining_step_offset,
                            motion_parameterization=motion_parameterization,
                            prediction_type=prediction_type,
                            use_chi=use_chi,
                            ligand_conditioner=ligand_conditioner,
                            ligand_generator=_generator_for_device(
                                state.model_input.protein_pos.device,
                                910000 + int(step_count),
                            ),
                        ).states
                        for state in states
                    )
                for row, value, state, trajectory_states in zip(
                    row_batch, device_values, states, batched_trajectories
                ):
                    start = state.model_input.protein_pos
                    step_metrics = [
                        _trajectory_motion_metrics(value, trajectory_state)
                        for trajectory_state in trajectory_states
                    ]
                    final = trajectory_states[-1]
                    row["trajectories"][str(int(step_count))] = {
                        "finite": all(
                            bool(torch.isfinite(item).all())
                            for item in trajectory_states
                        ),
                        "state_count": len(trajectory_states),
                        "times": [
                            1.0 - float(index) / float(step_count)
                            for index in range(len(trajectory_states))
                        ],
                        **step_metrics[-1],
                        "step_metrics": step_metrics,
                        "mean_displacement_from_apo": float(
                            torch.linalg.vector_norm(final - start, dim=-1).mean()
                        ),
                    }
            rows.extend(row_batch)
            if progress_label:
                print(
                    f"[evaluate] {progress_label} "
                    f"{min(batch_start + len(raw_batch), len(values))}/{len(values)}",
                    flush=True,
                )
    finally:
        model.train(model_was_training)

    return summarize_diffusion_rows(
        rows,
        step_count=max(int(step) for step in steps),
        remaining_step_offset=remaining_step_offset,
        motion_parameterization=motion_parameterization,
        prediction_type=prediction_type,
    )


def _advance_autonomous_state(
    model: DiffusionMotionAdapter,
    model_input,
    current: torch.Tensor,
    current_chi: torch.Tensor,
    *,
    fraction: float,
    use_chi: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply one detached reverse update using the sampler's exact contract."""

    frames = build_residue_frames(
        current,
        model_input.atom_to_residue,
        model_input.protein_atom_name,
        num_residues=model_input.residue_type.shape[0],
        atom_indices=model_input.frame_atom_indices,
    )
    with torch.no_grad():
        prediction = model(model_input)
    updated = apply_local_se3_update(
        current,
        model_input.atom_to_residue,
        frames.origins,
        frames.frames,
        prediction.translation_local,
        prediction.rotation_local,
        fraction=fraction,
        frame_valid=model_input.frame_valid & frames.valid,
    )
    if use_chi:
        chi_state = build_current_chi_state(
            updated,
            model_input.protein_atom_name,
            model_input.atom_to_residue,
            model_input.protein_residue_name,
        )
        chi_valid = (
            model_input.chi_mask
            & chi_state.geometry_rotatable_mask
            & model_input.frame_valid[:, None]
        )
        updated = apply_chi_updates(
            updated,
            chi_state.axis_start,
            chi_state.axis_end,
            chi_state.downstream_atom_mask,
            prediction.chi * fraction,
            valid=chi_valid,
        ).positions
    next_chi = build_current_chi_state(
        updated,
        model_input.protein_atom_name,
        model_input.atom_to_residue,
        model_input.protein_residue_name,
    ).angles
    return updated.detach(), next_chi.detach()


def build_scheduled_self_state(
    model: DiffusionMotionAdapter,
    complex_value: PocketComplex,
    exposed_steps: int,
    rollout_steps: int,
    *,
    ligand_conditioner=None,
    generator: Optional[torch.Generator] = None,
    remaining_step_offset: float = 0.0,
    motion_parameterization: str = "remaining",
    prediction_type: str = "bridge_velocity",
    target_parameterization: str = "bridge_velocity",
    target_steps: int = 20,
    use_chi: bool = True,
) :
    """Build a detached state from the model's own prefix rollout.

    ``exposed_steps=0`` returns the apo boundary.  For positive values, the
    model is called at exactly the same times as ``sample_reverse_trajectory``
    and the resulting coordinates are detached before supervised targets are
    constructed.  Holo coordinates therefore remain target-only.
    """

    if not isinstance(exposed_steps, int) or exposed_steps < 0:
        raise ValueError("exposed_steps must be a non-negative integer")
    if not isinstance(rollout_steps, int) or rollout_steps <= 0:
        raise ValueError("rollout_steps must be a positive integer")
    if exposed_steps > rollout_steps:
        raise ValueError("exposed_steps cannot exceed rollout_steps")
    if (
        remaining_step_offset < 0.0
        or not torch.isfinite(torch.tensor(remaining_step_offset))
    ):
        raise ValueError("remaining_step_offset must be finite and non-negative")
    if motion_parameterization not in ("remaining", "bridge_rate"):
        raise ValueError(
            "motion_parameterization must be remaining or bridge_rate"
        )
    if prediction_type not in ("remaining", "velocity", "bridge_velocity"):
        raise ValueError("unsupported prediction_type")
    if target_parameterization not in ("remaining", "velocity", "bridge_velocity"):
        raise ValueError("unsupported target_parameterization")
    if prediction_type != target_parameterization:
        raise ValueError("prediction_type and target_parameterization must match")
    state = sample_diffusion_state(
        complex_value,
        1.0,
        translation_noise_scale=0.0,
        rotation_noise_scale=0.0,
        chi_noise_scale=0.0,
        generator=generator,
        ligand_conditioner=ligand_conditioner,
        target_parameterization=target_parameterization,
        target_steps=target_steps,
    )
    if exposed_steps == 0:
        return state
    was_training = model.training
    model.eval()
    current = state.model_input.protein_pos.detach().clone()
    current_chi = state.model_input.chi_current.detach().clone()
    try:
        for index in range(exposed_steps):
            current_t = 1.0 - float(index) / float(rollout_steps)
            inp = replace(
                state.model_input,
                protein_pos=current,
                chi_current=current_chi,
                diffusion_time=torch.tensor(
                    [current_t], dtype=current.dtype, device=current.device
                ),
            )
            if ligand_conditioner is not None:
                condition = ligand_conditioner.condition_from_input(
                    inp,
                    current_t,
                    generator=generator,
                )
                inp = replace(
                    inp,
                    ligand_pos=condition.ligand_pos.to(
                        device=current.device, dtype=torch.float32
                    ),
                    ligand_type=condition.ligand_type.to(
                        device=current.device, dtype=torch.long
                    ),
                    ligand_condition_t=torch.tensor(
                        [condition.targetdiff_t],
                        dtype=torch.long,
                        device=current.device,
                    ),
                )
            remaining_steps = rollout_steps - index
            if prediction_type in ("velocity", "bridge_velocity"):
                # Velocity is integrated over a fixed normalized interval.
                # This must match sample_reverse_trajectory; using the
                # remaining-step denominator here would over-advance
                # scheduled self-states near t=0.
                fraction = 1.0 / float(rollout_steps)
            elif motion_parameterization == "bridge_rate":
                fraction = 1.0 / float(rollout_steps + remaining_step_offset)
            else:
                fraction = 1.0 / float(remaining_steps + remaining_step_offset)
            current, current_chi = _advance_autonomous_state(
                model,
                inp,
                current,
                current_chi,
                fraction=fraction,
                use_chi=use_chi,
            )
        next_t = 1.0 - float(exposed_steps) / float(rollout_steps)
        final_ligand_pos = state.model_input.ligand_pos.detach()
        final_ligand_type = state.model_input.ligand_type.detach()
        final_condition_t = (
            int(state.model_input.ligand_condition_t.reshape(-1)[0])
            if state.model_input.ligand_condition_t is not None
            else None
        )
        if ligand_conditioner is not None:
            condition = ligand_conditioner.condition_from_input(
                replace(
                    state.model_input,
                    protein_pos=current,
                    chi_current=current_chi,
                    diffusion_time=torch.tensor(
                        [next_t], dtype=current.dtype, device=current.device
                    ),
                ),
                next_t,
                generator=generator,
            )
            final_ligand_pos = condition.ligand_pos.detach()
            final_ligand_type = condition.ligand_type.detach()
            final_condition_t = condition.targetdiff_t
        return build_diffusion_state_from_current(
            complex_value,
            current,
            next_t,
            ligand_pos=final_ligand_pos,
            ligand_type=final_ligand_type,
            ligand_condition_t=final_condition_t,
            target_parameterization=target_parameterization,
            target_steps=target_steps,
        )
    finally:
        model.train(was_training)


def train_diffusion_model(
    train_values: Sequence[PocketComplex],
    config: DiffusionTrainConfig,
    *,
    model: Optional[DiffusionMotionAdapter] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ligand_conditioner=None,
    update_callback=None,
) -> Tuple[DiffusionMotionAdapter, torch.optim.Optimizer, List[Dict[str, object]]]:
    if not train_values:
        raise ValueError("train_values must be non-empty")
    if model is None:
        model = DiffusionMotionAdapter(
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            encoder_backend=config.encoder_backend,
            prediction_type=config.prediction_type,
            architecture_variant=config.architecture_variant,
        ).to(_resolve_device(config.device))
    model._pocketdiff_use_chi = bool(config.use_chi)
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = _resolve_device(config.device)
    # Keep the cache on host memory. Moving all 3000 complexes to the GPU
    # before the first update creates a long opaque startup phase and consumes
    # several GB of device memory. Individual complexes are moved when they
    # are selected into the current micro-batch.
    if optimizer is None:
        optimizer = torch.optim.Adam(
            (
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    motion_buckets, motion_scores = _build_motion_buckets(
        train_values,
        config.motion_bucket_count,
    )
    rows: List[Dict[str, object]] = []
    for step in range(config.updates):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        micro_objectives = []
        row_sample_ids = []
        row_times = []
        self_state_count = 0
        for micro_step in range(config.gradient_accumulation_steps):
            states = []
            for batch_offset in range(config.batch_size):
                draw_id = (
                    step * config.gradient_accumulation_steps * config.batch_size
                    + micro_step * config.batch_size
                    + batch_offset
                )
                if config.motion_bucket_balance and motion_buckets:
                    bucket_id = draw_id % len(motion_buckets)
                    bucket = motion_buckets[bucket_id]
                    sample_index = bucket[
                        (draw_id // len(motion_buckets)) % len(bucket)
                    ]
                else:
                    sample_index = draw_id % len(train_values)
                value = _move_complex_to_device(
                    train_values[sample_index],
                    model_device,
                )
                generator = _generator_for_device(
                    value.protein_pos_apo.device,
                    config.diffusion_seed
                    + step * config.gradient_accumulation_steps
                    + micro_step * config.batch_size
                    + batch_offset
                )
                use_self_state = (
                    config.self_state_probability > 0.0
                    and step >= config.self_state_warmup
                    and config.self_state_steps > 0
                    and random.Random(
                        config.diffusion_seed
                        + step * config.gradient_accumulation_steps
                        + micro_step * config.batch_size
                        + batch_offset
                    ).random() < config.self_state_probability
                )
                if use_self_state:
                    exposed_steps = 1 + (
                        (step + micro_step + batch_offset) % config.self_state_steps
                    )
                    state = build_scheduled_self_state(
                        model,
                        value,
                        exposed_steps,
                        config.self_state_rollout_steps,
                        ligand_conditioner=ligand_conditioner,
                        generator=generator,
                        remaining_step_offset=config.remaining_step_offset,
                        motion_parameterization=config.motion_parameterization,
                        prediction_type=config.prediction_type,
                        target_parameterization=config.target_parameterization,
                        target_steps=config.target_steps,
                        use_chi=config.use_chi,
                    )
                    self_state_count += 1
                else:
                    time_rng = random.Random(
                        config.diffusion_seed
                        + 1000003 * step
                        + 1009 * micro_step
                        + batch_offset
                    )
                    # The deployment trajectory always starts exactly at apo
                    # (t=1).  Uniform interior-time sampling never exposes
                    # that boundary, so reserve a fixed fraction of clean
                    # training states for the actual inference start.
                    if time_rng.random() < config.apo_boundary_probability:
                        time_value = 1.0
                    else:
                        time_value = time_rng.uniform(config.time_min, config.time_max)
                    state = sample_diffusion_state(
                        value,
                        time_value,
                        translation_noise_scale=config.translation_noise_scale,
                        rotation_noise_scale=config.rotation_noise_scale,
                        chi_noise_scale=config.chi_noise_scale,
                        generator=generator,
                        ligand_conditioner=ligand_conditioner,
                        target_parameterization=config.target_parameterization,
                        target_steps=config.target_steps,
                    )
                states.append(state)
                row_sample_ids.append(value.sample_id)
                row_times.append(float(state.t))
            model_input, target = collate_diffusion_states(states)
            prediction = model(model_input)
            objective = diffusion_motion_loss(
                prediction,
                target,
                model_input=model_input,
                endpoint_weight=config.endpoint_weight,
                next_state_weight=config.next_state_weight,
                next_state_steps=config.next_state_steps,
                chi_loss_weight=config.chi_loss_weight,
                backbone_only=config.backbone_only,
                score_normalize=config.score_normalize,
                score_floor=config.score_floor,
                motion_parameterization=config.motion_parameterization,
                prediction_type=config.prediction_type,
                backbone_endpoint_weight=config.backbone_endpoint_weight,
                continuity_weight=config.continuity_weight,
                direction_weight=config.direction_weight,
                direction_threshold=config.direction_threshold,
                backbone_progress_direction_weight=(
                    config.backbone_progress_direction_weight
                ),
                backbone_progress_magnitude_weight=(
                    config.backbone_progress_magnitude_weight
                ),
                backbone_progress_threshold=config.backbone_progress_threshold,
            )
            if not bool(torch.isfinite(objective.loss)):
                raise FloatingPointError(f"non-finite diffusion loss at step {step}")
            (objective.loss / config.gradient_accumulation_steps).backward()
            micro_objectives.append(objective)
        objective = micro_objectives[-1]
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm))
        optimizer.step()
        rows.append(
            {
                "step": step,
                "sample_id": row_sample_ids[0] if len(row_sample_ids) == 1 else row_sample_ids,
                "sample_ids": row_sample_ids,
                "time": row_times[0] if len(row_times) == 1 else row_times,
                "times": row_times,
                "graph_count": len(row_sample_ids),
                "self_state_count": self_state_count,
                "loss": float(objective.loss.detach()),
                "translation_loss": float(objective.translation_loss.detach()),
                "rotation_loss": float(objective.rotation_loss.detach()),
                "chi_loss": float(objective.chi_loss.detach()),
                "endpoint_loss": float(objective.endpoint_loss.detach()),
                "backbone_loss": float(objective.backbone_loss.detach()),
                "next_state_loss": float(objective.next_state_loss.detach()),
                "continuity_loss": float(objective.continuity_loss.detach()),
                "direction_loss": float(objective.direction_loss.detach()),
                "backbone_progress_direction_loss": float(
                    objective.backbone_progress_direction_loss.detach()
                ),
                "backbone_progress_magnitude_loss": float(
                    objective.backbone_progress_magnitude_loss.detach()
                ),
                "gradient_norm_pre_clip": gradient,
                "finite": True,
            }
        )
        if update_callback is not None:
            update_callback(step, rows[-1], model, optimizer)
    return model, optimizer, rows


@torch.no_grad()
def evaluate_diffusion_loss(
    model: DiffusionMotionAdapter,
    values: Sequence[PocketComplex],
    config: DiffusionTrainConfig,
    *,
    batch_size: int = 4,
    ligand_conditioner=None,
) -> Dict[str, object]:
    """Evaluate the fixed validation objective used for checkpoint selection.

    This deliberately does not run an apo-start rollout or inspect holo RMSD.
    It evaluates the same local bridge/geometry objective used for training on
    a deterministic sequence of validation states, then averages by complex
    count rather than by batch count.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not values:
        return {
            "sample_count": 0,
            "total_loss": 0.0,
            "translation_loss": 0.0,
            "rotation_loss": 0.0,
            "chi_loss": 0.0,
            "endpoint_loss": 0.0,
            "backbone_loss": 0.0,
            "next_state_loss": 0.0,
            "continuity_loss": 0.0,
            "direction_loss": 0.0,
            "backbone_progress_direction_loss": 0.0,
            "backbone_progress_magnitude_loss": 0.0,
        }
    model_was_training = model.training
    model.eval()
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device(config.device)
    sums = {
        "total_loss": 0.0,
        "translation_loss": 0.0,
        "rotation_loss": 0.0,
        "chi_loss": 0.0,
        "endpoint_loss": 0.0,
        "backbone_loss": 0.0,
        "next_state_loss": 0.0,
        "continuity_loss": 0.0,
        "direction_loss": 0.0,
        "backbone_progress_direction_loss": 0.0,
        "backbone_progress_magnitude_loss": 0.0,
    }
    count = 0
    for batch_start in range(0, len(values), batch_size):
        raw_batch = values[batch_start : batch_start + batch_size]
        states = []
        for offset, raw_value in enumerate(raw_batch):
            value = _move_complex_to_device(raw_value, model_device)
            time_value = float(config.times[(batch_start + offset) % len(config.times)])
            generator = _generator_for_device(
                model_device,
                config.diffusion_seed + 7000003 + batch_start + offset,
            )
            states.append(
                sample_diffusion_state(
                    value,
                    time_value,
                    translation_noise_scale=config.translation_noise_scale,
                    rotation_noise_scale=config.rotation_noise_scale,
                    chi_noise_scale=config.chi_noise_scale,
                    generator=generator,
                    ligand_conditioner=ligand_conditioner,
                    target_parameterization=config.target_parameterization,
                    target_steps=config.target_steps,
                )
            )
        model_input, target = collate_diffusion_states(states)
        prediction = model(model_input)
        objective = diffusion_motion_loss(
            prediction,
            target,
            model_input=model_input,
            endpoint_weight=config.endpoint_weight,
            next_state_weight=config.next_state_weight,
            next_state_steps=config.next_state_steps,
            chi_loss_weight=config.chi_loss_weight,
            backbone_only=config.backbone_only,
            score_normalize=config.score_normalize,
            score_floor=config.score_floor,
            motion_parameterization=config.motion_parameterization,
            prediction_type=config.prediction_type,
            backbone_endpoint_weight=config.backbone_endpoint_weight,
            continuity_weight=config.continuity_weight,
            direction_weight=config.direction_weight,
            direction_threshold=config.direction_threshold,
            backbone_progress_direction_weight=(
                config.backbone_progress_direction_weight
            ),
            backbone_progress_magnitude_weight=(
                config.backbone_progress_magnitude_weight
            ),
            backbone_progress_threshold=config.backbone_progress_threshold,
        )
        weight = len(states)
        for key, value in (
            ("total_loss", objective.loss),
            ("translation_loss", objective.translation_loss),
            ("rotation_loss", objective.rotation_loss),
            ("chi_loss", objective.chi_loss),
            ("endpoint_loss", objective.endpoint_loss),
            ("backbone_loss", objective.backbone_loss),
            ("next_state_loss", objective.next_state_loss),
            ("continuity_loss", objective.continuity_loss),
            ("direction_loss", objective.direction_loss),
            (
                "backbone_progress_direction_loss",
                objective.backbone_progress_direction_loss,
            ),
            (
                "backbone_progress_magnitude_loss",
                objective.backbone_progress_magnitude_loss,
            ),
        ):
            if not bool(torch.isfinite(value)):
                raise FloatingPointError(
                    f"non-finite validation loss component: {key}"
                )
            sums[key] += float(value.detach()) * weight
        count += weight
    if count == 0:
        raise RuntimeError("validation loss received no complexes")
    result = {"sample_count": count}
    result.update({key: value / count for key, value in sums.items()})
    if model_was_training:
        model.train()
    return result


def save_diffusion_checkpoint(
    path: Path,
    model: DiffusionMotionAdapter,
    optimizer: torch.optim.Optimizer,
    *,
    config: DiffusionTrainConfig,
    report: Dict[str, object],
    experiment_metadata: Optional[Dict[str, object]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_config = {
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "encoder_backend": config.encoder_backend,
        "prediction_type": config.prediction_type,
        "architecture_variant": config.architecture_variant,
        "target_parameterization": config.target_parameterization,
        "target_steps": config.target_steps,
        "max_translation": (
            2.5 if config.architecture_variant == "v47" else None
        ),
        "architecture_contract": (
            "targetdiff-atom-contact-sequence-constrained-v47-directed-backbone"
            if (
                config.architecture_variant == "v47"
                and (
                    config.backbone_progress_direction_weight > 0.0
                    or config.backbone_progress_magnitude_weight > 0.0
                )
            )
            else (
                "targetdiff-atom-contact-sequence-constrained-v47"
                if config.architecture_variant == "v47"
                else None
            )
        ),
    }
    torch.save(
        {
            "format": "pocketdiff-diffusion-v48-directed-backbone",
            "checkpoint_version": 5,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config,
            "sampler_config": {
                "motion_parameterization": config.motion_parameterization,
                "prediction_type": config.prediction_type,
                "sampler_steps": list(config.sampler_steps),
                "remaining_step_offset": config.remaining_step_offset,
                "use_chi": config.use_chi,
                "target_steps": config.target_steps,
                "apo_boundary_probability": config.apo_boundary_probability,
            },
            "training_config": config_to_dict(config),
            "experiment_metadata": experiment_metadata or {},
            "report": report,
        },
        path,
    )


def load_diffusion_checkpoint(path: Path, *, map_location: str = "cpu") -> LoadedDiffusionCheckpoint:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("format") not in (
        "pocketdiff-diffusion-v1",
        "pocketdiff-diffusion-v45",
        "pocketdiff-diffusion-v46",
        "pocketdiff-diffusion-v47",
        "pocketdiff-diffusion-v48-directed-backbone",
    ):
        raise ValueError(f"unsupported checkpoint format: {payload.get('format')!r}")
    model_config = payload.get("model_config", {})
    model = DiffusionMotionAdapter(
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        dropout=float(model_config.get("dropout", 0.0)),
        encoder_backend=str(model_config.get("encoder_backend", "scalar")),
        prediction_type=str(model_config.get("prediction_type", "velocity")),
        architecture_variant=str(model_config.get("architecture_variant", "v45")),
    ).to(map_location)
    model.load_state_dict(payload["model_state_dict"])
    model._pocketdiff_use_chi = bool(
        payload.get("training_config", {}).get("use_chi", True)
    )
    model.eval()
    return LoadedDiffusionCheckpoint(model=model, payload=payload)


def run_diffusion_training(
    config: DiffusionTrainConfig,
    output_dir: Path,
    *,
    ligand_conditioner=None,
) -> Dict[str, object]:
    """Run training, evaluation, checkpoint save and strict reload validation."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slices = load_apo2mol_slices(config)
    model = DiffusionMotionAdapter(
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        encoder_backend=config.encoder_backend,
        prediction_type=config.prediction_type,
        architecture_variant=config.architecture_variant,
    ).to(_resolve_device(config.device))
    model._pocketdiff_use_chi = bool(config.use_chi)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model, optimizer, training_rows = train_diffusion_model(
        slices["train"].values,
        config,
        model=model,
        ligand_conditioner=ligand_conditioner,
    )
    parameter_changed = any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )
    train_report = evaluate_diffusion_sampler(
        model,
        slices["train"].values,
        steps=config.sampler_steps,
        remaining_step_offset=config.remaining_step_offset,
        motion_parameterization=config.motion_parameterization,
        prediction_type=config.prediction_type,
        ligand_conditioner=ligand_conditioner,
    )
    valid_report = evaluate_diffusion_sampler(
        model,
        slices["valid"].values,
        steps=config.sampler_steps,
        remaining_step_offset=config.remaining_step_offset,
        motion_parameterization=config.motion_parameterization,
        prediction_type=config.prediction_type,
        ligand_conditioner=ligand_conditioner,
    )
    test_report = evaluate_diffusion_sampler(
        model,
        slices["test"].values,
        steps=config.sampler_steps,
        remaining_step_offset=config.remaining_step_offset,
        motion_parameterization=config.motion_parameterization,
        prediction_type=config.prediction_type,
        ligand_conditioner=ligand_conditioner,
    )
    train_improved = train_report["mean_final_holo_rmsd"] < train_report["mean_apo_rmsd"]
    valid_improved = (
        True if not slices["valid"].values
        else valid_report["mean_final_holo_rmsd"] < valid_report["mean_apo_rmsd"]
    )
    report: Dict[str, object] = {
        "format": "pocketdiff-diffusion-run-v1",
        "passed": bool(
            parameter_changed
            and all(row["finite"] for row in training_rows)
            and train_report["all_start_matches_apo"]
            and train_report["all_trajectories_finite"]
            and valid_report["all_start_matches_apo"]
            and valid_report["all_trajectories_finite"]
            and test_report["all_start_matches_apo"]
            and test_report["all_trajectories_finite"]
        ),
        "learning_goal_met": bool(train_improved and valid_improved),
        "config": config_to_dict(config),
        "protocol": {
            "train_samples": slices["train"].sample_ids,
            "valid_samples": slices["valid"].sample_ids,
            "holdout_samples": slices["holdout"].sample_ids,
            "test_samples": slices["test"].sample_ids,
            "train_indices": slices["train"].indices,
            "valid_indices": slices["valid"].indices,
            "holdout_indices": slices["holdout"].indices,
            "test_indices": slices["test"].indices,
            "train_rejected": slices["train"].rejected,
            "valid_rejected": slices["valid"].rejected,
            "holdout_rejected": slices["holdout"].rejected,
            "test_rejected": slices["test"].rejected,
        },
        "parameter_changed": parameter_changed,
        "initial_loss": training_rows[0]["loss"],
        "final_loss": training_rows[-1]["loss"],
        "max_gradient_norm_pre_clip": max(row["gradient_norm_pre_clip"] for row in training_rows),
        "training_rows": training_rows,
        "train": train_report,
        "valid": valid_report,
        "holdout": valid_report,
        "test": test_report,
        "train_mean_improved": bool(train_improved),
        "valid_mean_improved": bool(valid_improved),
        "holdout_mean_improved": bool(valid_improved),
    }
    checkpoint = output_dir / "checkpoint.pt"
    save_diffusion_checkpoint(checkpoint, model, optimizer, config=config, report=report)
    model_device = next(model.parameters()).device
    loaded = load_diffusion_checkpoint(
        checkpoint,
        map_location=str(model_device),
    )
    reload_value = _move_complex_to_device(
        slices["train"].values[0],
        model_device,
    )
    reload_state = sample_diffusion_state(
        reload_value,
        0.5,
        # This smoke state is deterministic because all noise scales are zero.
        # Do not pass a CUDA generator here: some PyTorch geometry kernels
        # internally require a CPU generator even when no random draw occurs.
        generator=None,
    )
    model.eval()
    with torch.no_grad():
        expected = model(reload_state.model_input)
        actual = loaded.model(reload_state.model_input)
    reload_exact = (
        torch.equal(expected.translation_local, actual.translation_local)
        and torch.equal(expected.rotation_local, actual.rotation_local)
        and torch.equal(expected.chi, actual.chi)
    )
    reload_close = (
        torch.allclose(expected.translation_local, actual.translation_local, atol=2e-6, rtol=2e-6)
        and torch.allclose(expected.rotation_local, actual.rotation_local, atol=2e-6, rtol=2e-6)
        and torch.allclose(expected.chi, actual.chi, atol=2e-6, rtol=2e-6)
    )
    report["checkpoint"] = str(checkpoint)
    report["checkpoint_reload_exact"] = bool(reload_exact)
    report["checkpoint_reload_close"] = bool(reload_close)
    report["checkpoint_reload_tolerance"] = (
        0.0 if model_device.type == "cpu" else 2e-6
    )
    report["passed"] = bool(
        report["passed"]
        and (reload_exact if model_device.type == "cpu" else reload_close)
    )
    save_diffusion_checkpoint(checkpoint, model, optimizer, config=config, report=report)
    (output_dir / "report.json").write_text(json_dumps(report))
    return report

def sample_checkpoint_on_complex(
    checkpoint: Path,
    value: PocketComplex,
    *,
    steps: int = 8,
    remaining_step_offset: float = 0.0,
    motion_parameterization: Optional[str] = None,
    prediction_type: Optional[str] = None,
    ligand_conditioner=None,
) -> ReverseTrajectory:
    loaded = load_diffusion_checkpoint(checkpoint)
    if motion_parameterization is None:
        training_config = loaded.payload.get("training_config", {})
        motion_parameterization = str(training_config.get("motion_parameterization", "remaining"))
    if prediction_type is None:
        model_config = loaded.payload.get("model_config", {})
        prediction_type = str(model_config.get("prediction_type", "velocity"))
    state = sample_diffusion_state(
        value,
        1.0,
        translation_noise_scale=0.0,
        rotation_noise_scale=0.0,
        chi_noise_scale=0.0,
        ligand_conditioner=ligand_conditioner,
    )
    return sample_reverse_trajectory(
        loaded.model,
        state,
        steps=steps,
        remaining_step_offset=remaining_step_offset,
        motion_parameterization=motion_parameterization,
        prediction_type=prediction_type,
        use_chi=bool(
            loaded.payload.get("training_config", {}).get("use_chi", True)
        ),
        ligand_conditioner=ligand_conditioner,
    )

def json_dumps(value: Dict[str, object]) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True) + "\n"

__all__ = [
    "DiffusionTrainConfig",
    "LoadedDiffusionCheckpoint",
    "SelectedComplexes",
    "config_to_dict",
    "coordinate_rmsd",
    "evaluate_diffusion_sampler",
    "load_apo2mol_slices",
    "load_diffusion_checkpoint",
    "load_split_records",
    "run_diffusion_training",
    "sample_checkpoint_on_complex",
    "save_diffusion_checkpoint",
    "select_convertible_complexes",
    "train_diffusion_model",
    "build_scheduled_self_state",
]
