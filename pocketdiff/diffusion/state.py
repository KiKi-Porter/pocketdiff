"""Apo2Mol/DynamicBind-style diffusion states for protein conformational change.

The contract uses ``t=0`` for the clean holo endpoint and ``t=1`` for the apo
side (maximum deformation/noise).  Training constructs a noisy state by
interpolating holo -> apo in residue SE(3)+chi space and adding independent
translation, rotation-vector, and chi noise.  Holo coordinates are retained
only in the returned supervision object; callers should pass only
``model_input`` to a model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple

import torch

from pocketdiff.data.schema import PocketComplex
from pocketdiff.geometry.bridge import build_bridge_state
from pocketdiff.geometry.chi import apply_chi_updates, periodic_chi_delta
from pocketdiff.geometry.current_state import build_current_chi_state
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.diffusion.se3 import apply_local_se3_update
from pocketdiff.geometry.bridge import remaining_transform_current_to_holo

Tensor = torch.Tensor


@dataclass(frozen=True)
class DiffusionStateInput:
    """The label-independent tensors allowed into the model."""

    protein_pos: Tensor
    protein_feature: Tensor
    atom_to_residue: Tensor
    residue_type: Tensor
    frame_valid: Tensor
    chi_current: Tensor
    chi_mask: Tensor
    ligand_pos: Tensor
    ligand_type: Tensor
    diffusion_time: Tensor
    protein_atom_name: Tuple[str, ...]
    protein_residue_name: Tuple[str, ...]
    # ``None`` keeps hand-built legacy test inputs valid. Production state
    # builders always populate the apo reference so the model can detect
    # drift from the inference anchor.
    apo_pos_ref: Optional[Tensor] = None
    batch_protein: Optional[Tensor] = None
    batch_residue: Optional[Tensor] = None
    batch_ligand: Optional[Tensor] = None
    ligand_condition_t: Optional[Tensor] = None
    ligand_pos_ref: Optional[Tensor] = None
    ligand_type_ref: Optional[Tensor] = None
    # Explicit chain-aware positional features. These are label-independent
    # and let the motion head distinguish peptide neighbors from spatial-only
    # contacts without relying on atom order inside a cached graph.
    residue_position: Optional[Tensor] = None
    residue_chain_break: Optional[Tensor] = None


@dataclass(frozen=True)
class DiffusionStateTarget:
    """Reverse targets and clean endpoint supervision."""

    translation_target_local: Tensor
    rotation_target_local: Tensor
    chi_target: Tensor
    target_valid: Tensor
    protein_pos_holo: Tensor


@dataclass(frozen=True)
class DiffusionState:
    """One sampled noisy state plus separated input/target views."""

    model_input: DiffusionStateInput
    target: DiffusionStateTarget
    t: float
    noise_translation_local: Tensor
    noise_rotation_local: Tensor
    noise_chi: Tensor

    def __post_init__(self) -> None:
        if not (0.0 <= self.t <= 1.0):
            raise ValueError("t must lie in [0, 1]")
        tensors = (
            self.model_input.protein_pos,
            self.model_input.chi_current,
            self.target.translation_target_local,
            self.target.rotation_target_local,
            self.target.chi_target,
            self.target.protein_pos_holo,
        )
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("diffusion state contains non-finite tensors")
        if self.model_input.apo_pos_ref is not None:
            if self.model_input.apo_pos_ref.shape != self.model_input.protein_pos.shape:
                raise ValueError("apo_pos_ref must match protein_pos")
            if not torch.isfinite(self.model_input.apo_pos_ref).all():
                raise ValueError("apo_pos_ref contains non-finite values")
        if self.model_input.ligand_pos_ref is not None:
            if self.model_input.ligand_pos_ref.shape != self.model_input.ligand_pos.shape:
                raise ValueError("ligand_pos_ref must match ligand_pos")
            if not torch.isfinite(self.model_input.ligand_pos_ref).all():
                raise ValueError("ligand_pos_ref contains non-finite values")
        if self.model_input.ligand_type_ref is not None:
            if self.model_input.ligand_type_ref.shape != self.model_input.ligand_type.shape:
                raise ValueError("ligand_type_ref must match ligand_type")
            if self.model_input.ligand_type_ref.dtype != torch.long:
                raise TypeError("ligand_type_ref must be LongTensor")
        batch_values = (
            self.model_input.batch_protein,
            self.model_input.batch_residue,
            self.model_input.batch_ligand,
        )
        if any(value is not None for value in batch_values):
            if any(value is None for value in batch_values):
                raise ValueError(
                    "batch_protein, batch_residue, and batch_ligand must be "
                    "provided together"
                )
            batch_protein, batch_residue, batch_ligand = batch_values
            if batch_protein.dtype != torch.long or batch_protein.shape != (
                self.model_input.protein_pos.shape[0],
            ):
                raise ValueError("batch_protein must match protein atom count")
            if batch_residue.dtype != torch.long or batch_residue.shape != (
                self.model_input.residue_type.shape[0],
            ):
                raise ValueError("batch_residue must match residue count")
            if batch_ligand.dtype != torch.long or batch_ligand.shape != (
                self.model_input.ligand_pos.shape[0],
            ):
                raise ValueError("batch_ligand must match ligand atom count")
            if any(
                value.numel() and int(value.min()) < 0
                for value in (batch_protein, batch_residue, batch_ligand)
            ):
                raise ValueError("batch metadata cannot contain negative graph ids")
            graph_ids = torch.unique(
                torch.cat((batch_protein, batch_residue, batch_ligand)), sorted=True
            )
            if graph_ids.numel() == 0 or not torch.equal(
                graph_ids,
                torch.arange(
                    graph_ids.numel(), device=graph_ids.device, dtype=torch.long
                ),
            ):
                raise ValueError("batch graph ids must be contiguous from zero")
            if self.model_input.diffusion_time.reshape(-1).numel() != graph_ids.numel():
                raise ValueError("diffusion_time must contain one value per graph")
        nr = self.model_input.residue_type.shape[0]
        if len(self.model_input.protein_atom_name) != self.model_input.protein_pos.shape[0]:
            raise ValueError("protein_atom_name must describe every protein atom")
        if len(self.model_input.protein_residue_name) != nr:
            raise ValueError("protein_residue_name must describe every residue")
        if self.model_input.chi_current.shape != (nr, 5):
            raise ValueError("chi_current must have shape [Nr, 5]")
        for name, value in (
            ("translation_target_local", self.target.translation_target_local),
            ("rotation_target_local", self.target.rotation_target_local),
            ("noise_translation_local", self.noise_translation_local),
            ("noise_rotation_local", self.noise_rotation_local),
        ):
            if value.shape != (nr, 3):
                raise ValueError(f"{name} must have shape [Nr, 3]")
        for name, value in (("chi_target", self.target.chi_target), ("noise_chi", self.noise_chi)):
            if value.shape != (nr, 5):
                raise ValueError(f"{name} must have shape [Nr, 5]")
        if self.target.target_valid.shape != (nr,) or self.target.target_valid.dtype != torch.bool:
            raise ValueError("target_valid must be BoolTensor [Nr]")


def _noise_like(shape, *, scale: float, generator: Optional[torch.Generator], device: torch.device) -> Tensor:
    if scale < 0.0:
        raise ValueError("noise scales must be non-negative")
    return torch.randn(shape, generator=generator, device=device, dtype=torch.float32) * float(scale)


def _residue_names(complex_value: PocketComplex) -> Tuple[str, ...]:
    """Return one residue name per residue from canonical atom metadata."""
    atom_to_residue = complex_value.atom_to_residue.detach().cpu().tolist()
    names = []
    for residue_id in range(complex_value.num_residues):
        atom_index = next(
            index for index, mapped_residue in enumerate(atom_to_residue)
            if mapped_residue == residue_id
        )
        names.append(complex_value.protein_residue_name[atom_index])
    return tuple(names)


def _residue_sequence_features(
    complex_value: PocketComplex,
) -> Tuple[Tensor, Tensor]:
    """Encode per-chain residue order and chain starts as CPU tensors."""

    positions = []
    chain_break = []
    previous_chain = None
    running = 0
    for chain in complex_value.residue_chain_id:
        if previous_chain is None or chain != previous_chain:
            running = 0
            chain_break.append(1.0)
        else:
            chain_break.append(0.0)
        positions.append(float(running))
        running += 1
        previous_chain = chain
    return (
        torch.tensor(positions, dtype=torch.float32),
        torch.tensor(chain_break, dtype=torch.float32),
    )


def _endpoint_transforms(complex_value: PocketComplex):
    apo = build_residue_frames(
        complex_value.protein_pos_apo,
        complex_value.atom_to_residue,
        complex_value.protein_atom_name,
        num_residues=complex_value.num_residues,
    )
    holo = build_residue_frames(
        complex_value.protein_pos_holo,
        complex_value.atom_to_residue,
        complex_value.protein_atom_name,
        num_residues=complex_value.num_residues,
    )
    valid = complex_value.frame_valid & apo.valid & holo.valid
    # The noisy state is built from holo toward apo.  Local reverse targets
    # are derived later from the actual noisy frames, using the same
    # current-frame convention as the sampler.
    chi_holo_to_apo = periodic_chi_delta(
        complex_value.chi_holo, complex_value.chi_apo, complex_value.chi_mask
    )
    return apo, holo, valid, chi_holo_to_apo


class LigandConditionProvider(Protocol):
    """Optional provider for time-dependent ligand coordinates and types."""

    def condition(
        self,
        complex_value: PocketComplex,
        pocket_time: float,
        *,
        generator: Optional[torch.Generator] = None,
    ):
        ...

    def condition_from_input(
        self,
        model_input: DiffusionStateInput,
        pocket_time: float,
        *,
        generator: Optional[torch.Generator] = None,
    ):
        ...


def _resolve_ligand_condition(
    complex_value: PocketComplex,
    t_value: float,
    *,
    conditioner: Optional[LigandConditionProvider],
    generator: Optional[torch.Generator],
) -> Tuple[Tensor, Tensor, Optional[int]]:
    if conditioner is None:
        return (
            complex_value.ligand_pos_ref,
            complex_value.ligand_type_ref,
            None,
        )
    result = conditioner.condition(
        complex_value,
        t_value,
        generator=generator,
    )
    if isinstance(result, tuple):
        if len(result) == 2:
            ligand_pos, ligand_type = result
            condition_t = None
        elif len(result) == 3:
            ligand_pos, ligand_type, condition_t = result
        else:
            raise ValueError("ligand condition tuple must contain 2 or 3 values")
    else:
        try:
            ligand_pos = result.ligand_pos
            ligand_type = result.ligand_type
            condition_t = getattr(result, "targetdiff_t", None)
        except AttributeError as exc:
            raise TypeError(
                "ligand condition provider must return a tuple or an object "
                "with ligand_pos and ligand_type"
            ) from exc
    if not isinstance(ligand_pos, torch.Tensor) or ligand_pos.shape != complex_value.ligand_pos_ref.shape:
        raise ValueError("ligand condition positions must match the reference ligand shape")
    if not isinstance(ligand_type, torch.Tensor) or ligand_type.shape != complex_value.ligand_type_ref.shape:
        raise ValueError("ligand condition types must match the reference ligand shape")
    if ligand_type.dtype != torch.long:
        raise TypeError("ligand condition types must be LongTensor")
    if not torch.isfinite(ligand_pos).all():
        raise ValueError("ligand condition positions must be finite")
    if condition_t is not None and (
        not isinstance(condition_t, int) or isinstance(condition_t, bool) or condition_t < 0
    ):
        raise ValueError("ligand condition timestep must be a non-negative integer")
    return ligand_pos, ligand_type, condition_t


def build_diffusion_state_from_current(
    complex_value: PocketComplex,
    protein_pos: Tensor,
    t: float,
    *,
    ligand_pos: Optional[Tensor] = None,
    ligand_type: Optional[Tensor] = None,
    ligand_condition_t: Optional[int] = None,
    ligand_conditioner: Optional[LigandConditionProvider] = None,
    generator: Optional[torch.Generator] = None,
) -> DiffusionState:
    """Build a supervised state from an arbitrary detached current structure.

    This is the shared state constructor for teacher-forced samples and
    scheduled autonomous exposure. It never puts holo tensors in
    ``DiffusionStateInput``; holo is used only to construct the separated
    target.
    """

    if not isinstance(complex_value, PocketComplex):
        raise TypeError("complex_value must be PocketComplex")
    if not isinstance(t, (float, int)) or not 0.0 <= float(t) <= 1.0:
        raise ValueError("t must be a scalar in [0, 1]")
    if protein_pos.shape != complex_value.protein_pos_apo.shape:
        raise ValueError("protein_pos must match the complex protein shape")
    if not protein_pos.is_floating_point() or not torch.isfinite(protein_pos).all():
        raise ValueError("protein_pos must be finite floating point")
    t_value = float(t)
    device = protein_pos.device
    residue_names = _residue_names(complex_value)
    residue_position, residue_chain_break = _residue_sequence_features(complex_value)
    _, holo, endpoint_valid, _ = _endpoint_transforms(complex_value)
    atom_to_residue = complex_value.atom_to_residue.to(device=device)
    current_frames = build_residue_frames(
        protein_pos,
        atom_to_residue,
        complex_value.protein_atom_name,
        num_residues=complex_value.num_residues,
    )
    valid = endpoint_valid.to(device=device) & current_frames.valid
    current_chi_state = build_current_chi_state(
        protein_pos,
        complex_value.protein_atom_name,
        atom_to_residue,
        residue_names,
    )
    chi_mask = (
        complex_value.chi_mask.to(device=device)
        & current_chi_state.geometry_rotatable_mask
        & valid[:, None]
    )
    current_chi = torch.where(
        chi_mask,
        current_chi_state.angles,
        torch.zeros_like(current_chi_state.angles),
    )
    remaining = remaining_transform_current_to_holo(
        current_frames.origins,
        current_frames.frames,
        holo.origins.to(device=device),
        holo.frames.to(device=device),
        frame_valid=valid,
    )
    target_chi = periodic_chi_delta(
        current_chi,
        complex_value.chi_holo.to(device=device),
        chi_mask,
    )
    if ligand_pos is None or ligand_type is None:
        ligand_pos, ligand_type, resolved_t = _resolve_ligand_condition(
            complex_value,
            t_value,
            conditioner=ligand_conditioner,
            generator=generator,
        )
        ligand_condition_t = resolved_t
    else:
        if ligand_conditioner is not None:
            raise ValueError("provide either explicit ligand tensors or a conditioner")
    ligand_pos = ligand_pos.to(device=device, dtype=torch.float32)
    ligand_type = ligand_type.to(device=device, dtype=torch.long)
    model_input = DiffusionStateInput(
        protein_pos=protein_pos,
        protein_feature=complex_value.protein_feature.to(device=device),
        atom_to_residue=atom_to_residue,
        residue_type=complex_value.residue_type.to(device=device),
        frame_valid=valid,
        chi_current=current_chi,
        chi_mask=chi_mask,
        ligand_pos=ligand_pos,
        ligand_type=ligand_type,
        diffusion_time=torch.tensor([t_value], dtype=torch.float32, device=device),
        protein_atom_name=tuple(complex_value.protein_atom_name),
        protein_residue_name=residue_names,
        apo_pos_ref=complex_value.protein_pos_apo.to(device=device),
        batch_protein=torch.zeros(
            complex_value.num_protein_atoms, dtype=torch.long, device=device
        ),
        batch_residue=torch.zeros(
            complex_value.num_residues, dtype=torch.long, device=device
        ),
        batch_ligand=torch.zeros(
            complex_value.num_ligand_atoms, dtype=torch.long, device=device
        ),
        ligand_condition_t=(
            torch.tensor([ligand_condition_t], dtype=torch.long, device=device)
            if ligand_condition_t is not None
            else None
        ),
        ligand_pos_ref=complex_value.ligand_pos_ref.to(device=device),
        ligand_type_ref=complex_value.ligand_type_ref.to(device=device),
        residue_position=residue_position.to(device=device),
        residue_chain_break=residue_chain_break.to(device=device),
    )
    target = DiffusionStateTarget(
        translation_target_local=remaining.translation_local,
        rotation_target_local=remaining.rotvec_local,
        chi_target=target_chi,
        target_valid=valid,
        protein_pos_holo=complex_value.protein_pos_holo.to(device=device),
    )
    zeros = torch.zeros_like(target_chi)
    return DiffusionState(
        model_input=model_input,
        target=target,
        t=t_value,
        noise_translation_local=torch.zeros_like(remaining.translation_local),
        noise_rotation_local=torch.zeros_like(remaining.rotvec_local),
        noise_chi=zeros,
    )


def sample_diffusion_state(
    complex_value: PocketComplex,
    t: float,
    *,
    translation_noise_scale: float = 0.10,
    rotation_noise_scale: float = 0.05,
    chi_noise_scale: float = 0.05,
    generator: Optional[torch.Generator] = None,
    ligand_conditioner: Optional[LigandConditionProvider] = None,
) -> DiffusionState:
    """Sample a finite protein diffusion state from one apo/holo pair.

    ``t=0`` is clean holo and ``t=1`` is apo.  Noise is scaled by ``t`` so
    that the clean endpoint is deterministic.  Targets are reverse local
    transforms from the sampled state toward holo, plus the clean holo coords.
    """
    if not isinstance(complex_value, PocketComplex):
        raise TypeError("complex_value must be PocketComplex")
    if not isinstance(t, (float, int)) or not 0.0 <= float(t) <= 1.0:
        raise ValueError("t must be a scalar in [0, 1]")
    t_value = float(t)
    ligand_pos_condition, ligand_type_condition, ligand_condition_t = _resolve_ligand_condition(
        complex_value,
        t_value,
        conditioner=ligand_conditioner,
        generator=generator,
    )
    residue_names = _residue_names(complex_value)
    apo, holo, valid, chi_holo_to_apo = _endpoint_transforms(complex_value)
    # Fraction from holo toward apo.  At t=0 this is exactly holo; at t=1 apo.
    fraction = torch.tensor(t_value, dtype=torch.float32, device=complex_value.protein_pos_apo.device)
    bridge = build_bridge_state(
        complex_value.protein_pos_holo,
        complex_value.atom_to_residue,
        holo.origins,
        holo.frames,
        apo.origins,
        apo.frames,
        fraction=fraction,
        frame_valid=valid,
    )
    noise_translation = _noise_like(
        (complex_value.num_residues, 3),
        scale=t_value * translation_noise_scale,
        generator=generator,
        device=bridge.protein_pos.device,
    )
    noise_rotation = _noise_like(
        (complex_value.num_residues, 3),
        scale=t_value * rotation_noise_scale,
        generator=generator,
        device=bridge.protein_pos.device,
    )
    noise_chi = _noise_like(
        chi_holo_to_apo.shape,
        scale=t_value * chi_noise_scale,
        generator=generator,
        device=bridge.protein_pos.device,
    )

    # Match Apo2Mol's transform order: residue rigid motion first, followed
    # by explicit side-chain χ rotations.  The rigid+χ construction is an
    # expressive bridge, but it does not reproduce every non-rigid apo atom.
    # Add that residual along the path so the stated t=1 endpoint is the real
    # apo structure rather than an approximation.
    bridge_chi_state = build_current_chi_state(
        bridge.protein_pos,
        complex_value.protein_atom_name,
        complex_value.atom_to_residue,
        residue_names,
    )
    chi_mask = (
        complex_value.chi_mask
        & bridge_chi_state.geometry_rotatable_mask
        & valid[:, None]
    )
    deterministic_chi = apply_chi_updates(
        bridge.protein_pos,
        bridge_chi_state.axis_start,
        bridge_chi_state.axis_end,
        bridge_chi_state.downstream_atom_mask,
        t_value * chi_holo_to_apo,
        valid=chi_mask,
    ).positions
    apo_bridge = build_bridge_state(
        complex_value.protein_pos_holo,
        complex_value.atom_to_residue,
        holo.origins,
        holo.frames,
        apo.origins,
        apo.frames,
        fraction=1.0,
        frame_valid=valid,
    )
    apo_bridge_chi_state = build_current_chi_state(
        apo_bridge.protein_pos,
        complex_value.protein_atom_name,
        complex_value.atom_to_residue,
        residue_names,
    )
    apo_chi_mask = (
        complex_value.chi_mask
        & apo_bridge_chi_state.geometry_rotatable_mask
        & valid[:, None]
    )
    apo_bridge_with_chi = apply_chi_updates(
        apo_bridge.protein_pos,
        apo_bridge_chi_state.axis_start,
        apo_bridge_chi_state.axis_end,
        apo_bridge_chi_state.downstream_atom_mask,
        chi_holo_to_apo,
        valid=apo_chi_mask,
    ).positions
    apo_residual = complex_value.protein_pos_apo - apo_bridge_with_chi
    deterministic_pos = deterministic_chi + t_value * apo_residual
    if t_value == 0.0:
        deterministic_pos = complex_value.protein_pos_holo.clone()
    elif t_value == 1.0:
        deterministic_pos = complex_value.protein_pos_apo.clone()

    deterministic_frames = build_residue_frames(
        deterministic_pos,
        complex_value.atom_to_residue,
        complex_value.protein_atom_name,
        num_residues=complex_value.num_residues,
    )
    noisy_pos = apply_local_se3_update(
        deterministic_pos,
        complex_value.atom_to_residue,
        deterministic_frames.origins,
        deterministic_frames.frames,
        noise_translation,
        noise_rotation,
        frame_valid=valid & deterministic_frames.valid,
    )
    # χ noise is applied after the rigid noise, matching the coordinate
    # update order used by the sampler.
    chi_state = build_current_chi_state(
        noisy_pos,
        complex_value.protein_atom_name,
        complex_value.atom_to_residue,
        residue_names,
    )
    chi_mask = (
        complex_value.chi_mask
        & chi_state.geometry_rotatable_mask
        & valid[:, None]
    )
    noisy_pos = apply_chi_updates(
        noisy_pos,
        chi_state.axis_start,
        chi_state.axis_end,
        chi_state.downstream_atom_mask,
        noise_chi,
        valid=chi_mask,
    ).positions
    chi_state = build_current_chi_state(
        noisy_pos,
        complex_value.protein_atom_name,
        complex_value.atom_to_residue,
        residue_names,
    )
    chi_mask = (
        complex_value.chi_mask
        & chi_state.geometry_rotatable_mask
        & valid[:, None]
    )
    chi_current = torch.where(chi_mask, chi_state.angles, torch.zeros_like(chi_state.angles))
    # Reverse target is the exact local transform from the sampled state to holo.
    current = build_residue_frames(noisy_pos, complex_value.atom_to_residue,
                                   complex_value.protein_atom_name, num_residues=complex_value.num_residues)
    remaining = remaining_transform_current_to_holo(
        current.origins,
        current.frames,
        holo.origins,
        holo.frames,
        frame_valid=valid & current.valid,
    )
    target_chi = periodic_chi_delta(chi_current, complex_value.chi_holo, chi_mask)
    residue_position, residue_chain_break = _residue_sequence_features(complex_value)
    model_input = DiffusionStateInput(
        protein_pos=noisy_pos,
        protein_feature=complex_value.protein_feature,
        atom_to_residue=complex_value.atom_to_residue,
        residue_type=complex_value.residue_type,
        frame_valid=valid,
        chi_current=chi_current,
        chi_mask=chi_mask,
        ligand_pos=ligand_pos_condition.to(device=noisy_pos.device, dtype=torch.float32),
        ligand_type=ligand_type_condition.to(device=noisy_pos.device, dtype=torch.long),
        diffusion_time=torch.tensor([t_value], dtype=torch.float32, device=noisy_pos.device),
        protein_atom_name=tuple(complex_value.protein_atom_name),
        protein_residue_name=residue_names,
        apo_pos_ref=complex_value.protein_pos_apo.to(device=noisy_pos.device),
        batch_protein=torch.zeros(
            complex_value.num_protein_atoms, dtype=torch.long, device=noisy_pos.device
        ),
        batch_residue=torch.zeros(
            complex_value.num_residues, dtype=torch.long, device=noisy_pos.device
        ),
        batch_ligand=torch.zeros(
            complex_value.num_ligand_atoms, dtype=torch.long, device=noisy_pos.device
        ),
        ligand_condition_t=(
            torch.tensor([ligand_condition_t], dtype=torch.long, device=noisy_pos.device)
            if ligand_condition_t is not None
            else None
        ),
        ligand_pos_ref=complex_value.ligand_pos_ref.to(device=noisy_pos.device),
        ligand_type_ref=complex_value.ligand_type_ref.to(device=noisy_pos.device),
        residue_position=residue_position.to(device=noisy_pos.device),
        residue_chain_break=residue_chain_break.to(device=noisy_pos.device),
    )
    target = DiffusionStateTarget(
        translation_target_local=remaining.translation_local,
        rotation_target_local=remaining.rotvec_local,
        chi_target=target_chi,
        target_valid=valid,
        protein_pos_holo=complex_value.protein_pos_holo,
    )
    return DiffusionState(model_input, target, t_value, noise_translation, noise_rotation, noise_chi)


def collate_diffusion_states(
    states: Tuple[DiffusionState, ...] | list[DiffusionState],
) -> Tuple[DiffusionStateInput, DiffusionStateTarget]:
    """Concatenate single-graph states into one graph-balanced model input."""

    if not states:
        raise ValueError("at least one diffusion state is required")
    device = states[0].model_input.protein_pos.device
    if any(state.model_input.protein_pos.device != device for state in states):
        raise ValueError("all diffusion states must use the same device")
    protein_parts = []
    apo_parts = []
    feature_parts = []
    atom_to_residue_parts = []
    residue_parts = []
    frame_parts = []
    chi_parts = []
    chi_mask_parts = []
    ligand_pos_parts = []
    ligand_type_parts = []
    protein_batches = []
    residue_batches = []
    ligand_batches = []
    protein_names = []
    residue_names = []
    residue_position_parts = []
    residue_chain_break_parts = []
    target_translation_parts = []
    target_rotation_parts = []
    target_chi_parts = []
    target_valid_parts = []
    holo_parts = []
    times = []
    condition_times = []
    residue_offset = 0
    has_condition_time = any(
        state.model_input.ligand_condition_t is not None for state in states
    )
    for graph_id, state in enumerate(states):
        value = state.model_input
        target = state.target
        nr = int(value.residue_type.shape[0])
        np_atoms = int(value.protein_pos.shape[0])
        nl = int(value.ligand_pos.shape[0])
        protein_parts.append(value.protein_pos)
        apo_parts.append(
            value.apo_pos_ref if value.apo_pos_ref is not None else value.protein_pos
        )
        feature_parts.append(value.protein_feature)
        atom_to_residue_parts.append(value.atom_to_residue + residue_offset)
        residue_parts.append(value.residue_type)
        frame_parts.append(value.frame_valid)
        chi_parts.append(value.chi_current)
        chi_mask_parts.append(value.chi_mask)
        ligand_pos_parts.append(value.ligand_pos)
        ligand_type_parts.append(value.ligand_type)
        protein_batches.append(torch.full((np_atoms,), graph_id, dtype=torch.long, device=device))
        residue_batches.append(torch.full((nr,), graph_id, dtype=torch.long, device=device))
        ligand_batches.append(torch.full((nl,), graph_id, dtype=torch.long, device=device))
        protein_names.extend(value.protein_atom_name)
        residue_names.extend(value.protein_residue_name)
        residue_position_parts.append(
            value.residue_position
            if value.residue_position is not None
            else torch.arange(nr, dtype=torch.float32, device=device)
        )
        residue_chain_break_parts.append(
            value.residue_chain_break
            if value.residue_chain_break is not None
            else torch.zeros(nr, dtype=torch.float32, device=device)
        )
        target_translation_parts.append(target.translation_target_local)
        target_rotation_parts.append(target.rotation_target_local)
        target_chi_parts.append(target.chi_target)
        target_valid_parts.append(target.target_valid)
        holo_parts.append(target.protein_pos_holo)
        times.append(float(state.t))
        if has_condition_time:
            condition_times.append(
                int(value.ligand_condition_t.reshape(-1)[0])
                if value.ligand_condition_t is not None
                else -1
            )
        residue_offset += nr
    model_input = DiffusionStateInput(
        protein_pos=torch.cat(protein_parts, dim=0),
        protein_feature=torch.cat(feature_parts, dim=0),
        atom_to_residue=torch.cat(atom_to_residue_parts, dim=0),
        residue_type=torch.cat(residue_parts, dim=0),
        frame_valid=torch.cat(frame_parts, dim=0),
        chi_current=torch.cat(chi_parts, dim=0),
        chi_mask=torch.cat(chi_mask_parts, dim=0),
        ligand_pos=torch.cat(ligand_pos_parts, dim=0),
        ligand_type=torch.cat(ligand_type_parts, dim=0),
        diffusion_time=torch.tensor(times, dtype=torch.float32, device=device),
        protein_atom_name=tuple(protein_names),
        protein_residue_name=tuple(residue_names),
        apo_pos_ref=torch.cat(apo_parts, dim=0),
        batch_protein=torch.cat(protein_batches, dim=0),
        batch_residue=torch.cat(residue_batches, dim=0),
        batch_ligand=torch.cat(ligand_batches, dim=0),
        ligand_condition_t=(
            torch.tensor(condition_times, dtype=torch.long, device=device)
            if has_condition_time
            else None
        ),
        ligand_pos_ref=torch.cat(
            [
                value.ligand_pos_ref
                if value.ligand_pos_ref is not None
                else value.ligand_pos
                for value in [state.model_input for state in states]
            ],
            dim=0,
        ),
        ligand_type_ref=torch.cat(
            [
                value.ligand_type_ref
                if value.ligand_type_ref is not None
                else value.ligand_type
                for value in [state.model_input for state in states]
            ],
            dim=0,
        ),
        residue_position=torch.cat(residue_position_parts, dim=0).to(device=device),
        residue_chain_break=torch.cat(residue_chain_break_parts, dim=0).to(device=device),
    )
    target = DiffusionStateTarget(
        translation_target_local=torch.cat(target_translation_parts, dim=0),
        rotation_target_local=torch.cat(target_rotation_parts, dim=0),
        chi_target=torch.cat(target_chi_parts, dim=0),
        target_valid=torch.cat(target_valid_parts, dim=0),
        protein_pos_holo=torch.cat(holo_parts, dim=0),
    )
    return model_input, target


__all__ = [
    "DiffusionState",
    "DiffusionStateInput",
    "DiffusionStateTarget",
    "LigandConditionProvider",
    "build_diffusion_state_from_current",
    "collate_diffusion_states",
    "sample_diffusion_state",
]
