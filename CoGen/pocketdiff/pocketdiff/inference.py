"""Independent PocketDiff inputs and one joint protein update (no TargetDiff)."""
from dataclasses import dataclass, fields
from typing import Sequence

import torch

from pocketdiff.data.apo2mol_adapter import AA_NAMES
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.geometry.joint_update import apply_joint_update
from pocketdiff.models import PocketDiffModel


@dataclass(frozen=True)
class PocketInputs:
    """Inference-only inputs. Holo positions/chi/supervision masks are absent.

    Use dataclasses.replace(inputs, protein_pos=next_pos, pocket_k=next_k, ...)
    to form a subsequent state. Frames and chi are recomputed from coordinates.
    Tensors stay on the supplied device; float32 is the current model contract.
    """
    protein_pos: torch.Tensor
    apo_pos_ref: torch.Tensor
    protein_feature: torch.Tensor
    atom_to_residue_global: torch.Tensor
    residue_type: torch.Tensor
    batch_protein: torch.Tensor
    batch_residue: torch.Tensor
    ligand_pos: torch.Tensor
    ligand_v: torch.Tensor
    batch_ligand: torch.Tensor
    targetdiff_t: torch.Tensor
    pocket_k: torch.Tensor
    protein_atom_name: Sequence[str]

    def model_kwargs(self):
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        tensors = [v for v in values.values() if isinstance(v, torch.Tensor)]
        if any(v.device != self.protein_pos.device for v in tensors):
            raise ValueError("PocketInputs tensors must share a device")
        for name in ('protein_pos', 'apo_pos_ref', 'protein_feature', 'ligand_pos'):
            value = values[name]
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(name + ' must be finite float32')
        # Validate mappings before frame indexing. Validity is derived below,
        # so no caller-supplied label-dependent mask can enter this API.
        values['frame_valid'] = torch.ones(self.residue_type.shape, dtype=torch.bool,
                                          device=self.protein_pos.device)
        PocketDiffModel._validate_batch(**values)
        values['frame_valid'] = build_residue_frames(
            self.apo_pos_ref, self.atom_to_residue_global, self.protein_atom_name,
            num_residues=self.residue_type.numel(),
        ).valid
        return values


def predict_joint_step(model: PocketDiffModel, inputs: PocketInputs):
    """Predict physical remaining motion and apply a per-graph 1/(20-k) step.

    Gradients and model train/eval mode are preserved; the caller controls
    no_grad/eval for inference. Ligand, apo anchor and metadata remain unchanged.
    """
    if not isinstance(model, PocketDiffModel) or model.chi_input_mode != 'current':
        raise ValueError("predict_joint_step requires the explicit current-chi model")
    values = inputs.model_kwargs()
    prediction = model(**values)
    return apply_joint_update(
        inputs.protein_pos, inputs.apo_pos_ref, inputs.atom_to_residue_global,
        inputs.protein_atom_name, [AA_NAMES[i] for i in inputs.residue_type.detach().cpu().tolist()],
        prediction, remaining_steps=(20 - inputs.pocket_k)[inputs.batch_residue],
    )


__all__ = ['PocketInputs', 'predict_joint_step']
