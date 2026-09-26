"""Single protein-only PocketDiff update on a TargetDiff tensor state."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import torch

from pocketdiff.data.schema import PocketStepOutput, ResidueMetadata
from pocketdiff.geometry.bridge import apply_fractional_update
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.models import PocketDiffModel
from pocketdiff.targetdiff.state import TargetDiffState


Tensor = torch.Tensor


def _graph_schedule(value: Union[int, Tensor], *, name: str, batch_size: int, device: torch.device) -> Tensor:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer or LongTensor [B]")
    if isinstance(value, int):
        result = torch.full((batch_size,), value, dtype=torch.long, device=device)
    elif isinstance(value, torch.Tensor):
        if value.dtype != torch.long or value.ndim != 1 or value.shape[0] != batch_size:
            raise ValueError(f"{name} must be LongTensor [{batch_size}]")
        result = value.to(device=device)
    else:
        raise TypeError(f"{name} must be an integer or LongTensor [B]")
    return result


def _metadata_on_state(
    state: TargetDiffState,
    metadata: ResidueMetadata,
) -> tuple[Tensor, Tensor, Tensor, List[str]]:
    """Validate global residue metadata and return tensors/string names for model input."""

    device = state.protein_pos.device
    num_residues = metadata.residue_type.numel()
    if metadata.atom_to_residue_global.shape[0] != state.protein_pos.shape[0]:
        raise ValueError("metadata atom mapping and state protein_pos have different lengths")
    if metadata.batch_residue.shape[0] != num_residues:
        raise ValueError("metadata batch_residue and residue_type have different lengths")
    if metadata.frame_valid_reference.shape[0] != num_residues:
        raise ValueError("metadata frame_valid_reference and residue_type have different lengths")
    if metadata.batch_residue.numel() == 0:
        raise ValueError("metadata must contain at least one residue")
    if metadata.batch_residue.dtype != torch.long or metadata.atom_to_residue_global.dtype != torch.long:
        raise TypeError("residue metadata mappings must be LongTensor")
    if int(metadata.batch_residue.min()) < 0 or int(metadata.batch_residue.max()) >= state.num_graphs:
        raise ValueError("metadata batch_residue contains an out-of-range graph id")
    atom_to_residue_cpu = metadata.atom_to_residue_global.detach().cpu()
    batch_residue_cpu = metadata.batch_residue.detach().cpu()
    if not torch.equal(batch_residue_cpu[atom_to_residue_cpu], state.batch_protein.detach().cpu()):
        raise ValueError("metadata atom_to_residue_global crosses state graph boundaries")
    graph_atom_counts = torch.bincount(state.batch_protein, minlength=state.num_graphs).tolist()
    if len(metadata.protein_atom_name) != state.num_graphs:
        raise ValueError("protein_atom_name must contain one list per state graph")
    if any(len(names) != graph_atom_counts[index] for index, names in enumerate(metadata.protein_atom_name)):
        raise ValueError("protein_atom_name lengths do not match state graph atom counts")
    names: List[str] = [name for graph_names in metadata.protein_atom_name for name in graph_names]
    if len(names) != state.protein_pos.shape[0] or any(not isinstance(name, str) for name in names):
        raise ValueError("protein_atom_name must contain one string per protein atom")
    return (
        metadata.atom_to_residue_global.to(device=device),
        metadata.residue_type.to(device=device),
        metadata.batch_residue.to(device=device),
        names,
    )


class PocketStepSolver:
    """Apply one PocketDiff prediction while changing protein coordinates only."""

    def __init__(self, model: PocketDiffModel):
        if not callable(model):
            raise TypeError("model must be callable with PocketDiff forward arguments")
        self.model = model

    def step(
        self,
        state: TargetDiffState,
        residue_metadata: ResidueMetadata,
        k: Union[int, Tensor],
    ) -> PocketStepOutput:
        return pocket_step(self.model, state, residue_metadata, k)


def pocket_step(
    model: PocketDiffModel,
    state: TargetDiffState,
    residue_metadata: ResidueMetadata,
    k: Union[int, Tensor],
) -> PocketStepOutput:
    """Predict and apply one protein-only update at pocket step ``k``.

    ``k=0`` means twenty remaining PocketDiff steps and ``k=19`` means the
    final full remaining transform.  The model receives the aligned TargetDiff
    ligand state as a read-only condition; no tensor in ``state`` is modified.
    """

    if not isinstance(state, TargetDiffState):
        raise TypeError("state must be a TargetDiffState")
    if not isinstance(residue_metadata, ResidueMetadata):
        raise TypeError("residue_metadata must be a ResidueMetadata")
    device = state.protein_pos.device
    k_graph = _graph_schedule(k, name="k", batch_size=state.num_graphs, device=device)
    if bool((k_graph < 0).any()) or bool((k_graph > 19).any()):
        raise ValueError("k must lie in [0, 19]")
    targetdiff_t = 199 - 10 * k_graph
    remaining_steps_graph = 20 - k_graph
    atom_to_residue, residue_type, batch_residue, protein_atom_name = _metadata_on_state(
        state, residue_metadata
    )
    frame_valid = residue_metadata.frame_valid_reference.to(device=device)
    prediction = model(
        protein_pos=state.protein_pos,
        apo_pos_ref=state.apo_pos_ref,
        protein_feature=state.protein_v,
        atom_to_residue_global=atom_to_residue,
        residue_type=residue_type,
        frame_valid=frame_valid,
        batch_protein=state.batch_protein,
        batch_residue=batch_residue,
        ligand_pos=state.ligand_pos,
        ligand_v=state.ligand_v,
        batch_ligand=state.batch_ligand,
        targetdiff_t=targetdiff_t,
        pocket_k=k_graph,
        protein_atom_name=protein_atom_name,
    )
    current_frames = build_residue_frames(
        state.protein_pos,
        atom_to_residue,
        protein_atom_name,
        num_residues=residue_type.shape[0],
    )
    valid = prediction.frame_valid & current_frames.valid & frame_valid
    remaining_steps_residue = remaining_steps_graph[batch_residue]
    protein_pos_next = apply_fractional_update(
        state.protein_pos,
        atom_to_residue,
        current_frames.origins,
        current_frames.frames,
        prediction.remaining_translation_local,
        prediction.remaining_rotvec_local,
        remaining_steps=remaining_steps_residue,
        frame_valid=valid,
    )
    applied_translation = prediction.remaining_translation_local / remaining_steps_residue.to(
        dtype=prediction.remaining_translation_local.dtype
    ).unsqueeze(-1)
    applied_rotvec = prediction.remaining_rotvec_local / remaining_steps_residue.to(
        dtype=prediction.remaining_rotvec_local.dtype
    ).unsqueeze(-1)
    applied_translation = torch.where(valid[:, None], applied_translation, torch.zeros_like(applied_translation))
    applied_rotvec = torch.where(valid[:, None], applied_rotvec, torch.zeros_like(applied_rotvec))
    diagnostics: Dict[str, Tensor] = {
        "k": k_graph,
        "targetdiff_t": targetdiff_t,
        "remaining_steps": remaining_steps_graph,
        "valid_frame_fraction": valid.to(dtype=torch.float32).mean(),
        "max_applied_translation": torch.linalg.vector_norm(applied_translation, dim=-1).max(),
        "max_applied_rotvec": torch.linalg.vector_norm(applied_rotvec, dim=-1).max(),
    }
    output = PocketStepOutput(
        protein_pos_next=protein_pos_next,
        prediction=prediction,
        applied_translation_local=applied_translation,
        applied_rotvec_local=applied_rotvec,
        applied_chi=None,
        diagnostics=diagnostics,
    )
    return output


__all__ = ["PocketStepSolver", "pocket_step"]
