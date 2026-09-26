from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence

import torch

from .geometry import apply_motion
from .constants import NUM_CHI


def sample_seed(seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return (seed + int.from_bytes(digest[:8], "little")) % (2**63 - 1)


def initial_state_from_apo(
    inputs: Dict[str, object],
    sample_ids: Sequence[str],
    *,
    seed: int,
    initial_noise_scale: float = 1.0,
) -> torch.Tensor:
    """Draw deterministic local rigid noise from apo-only inputs."""
    if not 0.0 <= initial_noise_scale <= 1.0:
        raise ValueError("initial_noise_scale must be in [0, 1]")
    apo = inputs["apo_pos"]
    if initial_noise_scale == 0.0:
        return apo.clone()
    device = apo.device
    translation = torch.zeros(
        (inputs["residue_type"].shape[0], 3),
        dtype=apo.dtype,
        device="cpu",
    )
    rotation = torch.zeros_like(translation)
    residue_ptr = inputs["residue_ptr"].detach().cpu().tolist()

    for graph_id, sample_id in enumerate(sample_ids):
        generator = torch.Generator(device="cpu").manual_seed(
            sample_seed(seed, sample_id)
        )
        sampled_noise_scale = (
            0.05 + 0.20 * torch.rand((), generator=generator).item()
        )
        start, end = residue_ptr[graph_id], residue_ptr[graph_id + 1]
        translation[start:end] = torch.randn(
            (end - start, 3), generator=generator, dtype=apo.dtype
        ) * (sampled_noise_scale * initial_noise_scale)
        rotation[start:end] = torch.randn(
            (end - start, 3), generator=generator, dtype=apo.dtype
        ) * (0.5 * sampled_noise_scale * initial_noise_scale)

    return apply_motion(
        inputs,
        apo,
        translation.to(device=device, dtype=apo.dtype),
        rotation.to(device=device, dtype=apo.dtype),
        apo.new_zeros((translation.shape[0], NUM_CHI)),
    )


def schedule_fraction(
    remaining_steps: int,
    total_steps: int,
    *,
    schedule_type: str = "remaining",
    max_fraction: float | None = None,
    fixed_fraction: float | None = None,
    damping: float = 0.25,
) -> float:
    """Return the rollout fraction applied to a predicted remaining transform."""
    if remaining_steps <= 0 or total_steps <= 0:
        raise ValueError("remaining_steps and total_steps must be positive")
    if remaining_steps > total_steps:
        raise ValueError("remaining_steps cannot exceed total_steps")
    if damping < 0.0:
        raise ValueError("damping must be non-negative")
    if max_fraction is not None and max_fraction <= 0.0:
        raise ValueError("max_fraction must be positive when provided")
    if fixed_fraction is not None and fixed_fraction <= 0.0:
        raise ValueError("fixed_fraction must be positive when provided")

    if schedule_type == "remaining":
        fraction = 1.0 / float(remaining_steps)
    elif schedule_type in {"clipped", "capped"}:
        if max_fraction is None:
            raise ValueError("max_fraction is required for clipped schedule")
        fraction = min(1.0 / float(remaining_steps), float(max_fraction))
    elif schedule_type == "damped":
        fraction = 1.0 / (
            float(remaining_steps) + float(damping) * float(total_steps)
        )
    elif schedule_type == "fixed":
        if fixed_fraction is None:
            raise ValueError("fixed_fraction is required for fixed schedule")
        fraction = float(fixed_fraction)
    else:
        raise ValueError("unknown schedule_type: " + schedule_type)

    if max_fraction is not None and schedule_type not in {"clipped", "capped"}:
        fraction = min(fraction, float(max_fraction))
    return fraction


@torch.inference_mode()
def sample_complexes(
    model,
    inputs: Dict[str, object],
    sample_ids: Sequence[str],
    *,
    steps: int = 4,
    seed: int = 20260924,
    motion_scale: float = 1.0,
    initial_noise_scale: float = 0.0,
    disable_chi: bool = False,
    schedule_type: str = "remaining",
    max_fraction: float | None = None,
    fixed_fraction: float | None = None,
    damping: float = 0.25,
) -> torch.Tensor:
    """Run the training-compatible rigid/chi rollout without target access."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 <= motion_scale <= 1.0:
        raise ValueError("motion_scale must be in [0, 1]")
    if not 0.0 <= initial_noise_scale <= 1.0:
        raise ValueError("initial_noise_scale must be in [0, 1]")
    if len(sample_ids) != inputs["atom_ptr"].numel() - 1:
        raise ValueError("sample_ids do not match the collated graph count")
    current = initial_state_from_apo(
        inputs,
        sample_ids,
        seed=seed,
        initial_noise_scale=initial_noise_scale,
    )
    if motion_scale == 0.0:
        if not torch.isfinite(current).all():
            raise FloatingPointError("sampler input contains non-finite coordinates")
        return current.clone()
    for step in range(steps):
        remaining_steps = steps - step
        remaining = float(remaining_steps) / float(steps)
        fraction = schedule_fraction(
            remaining_steps,
            steps,
            schedule_type=schedule_type,
            max_fraction=max_fraction,
            fixed_fraction=fixed_fraction,
            damping=damping,
        )
        prediction = model(
            {"input": inputs},
            current,
            current.new_tensor(remaining),
        )
        current = apply_motion(
            inputs,
            current,
            prediction["remaining_translation_local"] * motion_scale,
            prediction["remaining_rotvec_local"] * motion_scale,
            torch.zeros_like(prediction["remaining_chi"])
            if disable_chi
            else prediction["remaining_chi"] * motion_scale,
            fraction=fraction,
        )
    if not torch.isfinite(current).all():
        raise FloatingPointError("sampler produced non-finite coordinates")
    return current


@torch.inference_mode()
def sample_complexes_with_trajectory(
    model,
    inputs: Dict[str, object],
    sample_ids: Sequence[str],
    *,
    steps: int = 4,
    seed: int = 20260924,
    motion_scale: float = 1.0,
    initial_noise_scale: float = 0.0,
    disable_chi: bool = False,
    schedule_type: str = "remaining",
    max_fraction: float | None = None,
    fixed_fraction: float | None = None,
    damping: float = 0.25,
):
    """Return the final state and every state after initialization/each step."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    current = initial_state_from_apo(
        inputs,
        sample_ids,
        seed=seed,
        initial_noise_scale=initial_noise_scale,
    )
    trajectory = [current.clone()]
    if motion_scale == 0.0:
        if not torch.isfinite(current).all():
            raise FloatingPointError("sampler input contains non-finite coordinates")
        trajectory.extend(current.clone() for _ in range(steps))
        return current.clone(), trajectory
    for step in range(steps):
        remaining_steps = steps - step
        remaining = float(remaining_steps) / float(steps)
        fraction = schedule_fraction(
            remaining_steps,
            steps,
            schedule_type=schedule_type,
            max_fraction=max_fraction,
            fixed_fraction=fixed_fraction,
            damping=damping,
        )
        prediction = model(
            {"input": inputs},
            current,
            current.new_tensor(remaining),
        )
        current = apply_motion(
            inputs,
            current,
            prediction["remaining_translation_local"] * motion_scale,
            prediction["remaining_rotvec_local"] * motion_scale,
            torch.zeros_like(prediction["remaining_chi"])
            if disable_chi
            else prediction["remaining_chi"] * motion_scale,
            fraction=fraction,
        )
        trajectory.append(current.clone())
    if not torch.isfinite(current).all():
        raise FloatingPointError("sampler produced non-finite coordinates")
    return current, trajectory
