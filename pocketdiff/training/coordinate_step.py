"""Independent joint-coordinate training; targets never enter model inputs."""
from dataclasses import dataclass
import math
from typing import Dict

import torch

from pocketdiff.data.schema import PocketStepOutput
from pocketdiff.inference import PocketInputs, predict_joint_step
from pocketdiff.models import PocketDiffModel
from .coordinate import NextXYZLoss, masked_next_xyz_loss


@dataclass(frozen=True)
class NextXYZBatch:
    """A current state and explicitly supplied next-step supervision.

    The caller constructs targets in the fixed apo frame for the supplied k.
    This interface neither substitutes the holo endpoint for the next step nor
    guesses symmetry-equivalent labels. The mask is used only by the loss.
    """
    inputs: PocketInputs
    target_pos_next: torch.Tensor
    supervision_frame_valid: torch.Tensor

    def validate(self):
        pos = self.inputs.protein_pos
        target, mask = self.target_pos_next, self.supervision_frame_valid
        if target.shape != pos.shape or target.dtype != torch.float32:
            raise ValueError('target_pos_next must be float32 with shape [Np,3]')
        if mask.dtype != torch.bool or mask.shape != self.inputs.residue_type.shape:
            raise ValueError('supervision_frame_valid must be BoolTensor [Nr]')
        if target.device != pos.device or mask.device != pos.device:
            raise ValueError('supervision and inputs must share a device')
        if not torch.isfinite(target).all():
            raise ValueError('target_pos_next must be finite')
        if not mask.any():
            raise ValueError('coordinate supervision has no valid residue')


@dataclass(frozen=True)
class NextXYZObjective:
    step_output: PocketStepOutput
    coordinate: NextXYZLoss


@dataclass(frozen=True)
class NextXYZTrainRecord:
    loss_before_update: float
    valid_atom_count: int
    valid_graph_count: int
    gradient_norm_before_clip: float
    gradient_norms: Dict[str, float]


def next_xyz_objective(model: PocketDiffModel, batch: NextXYZBatch) -> NextXYZObjective:
    """Use the inference solver unchanged; detach labels only at the loss."""
    batch.validate()
    output = predict_joint_step(model, batch.inputs)
    valid = output.diagnostics['frame_valid'] & batch.supervision_frame_valid
    if bool((valid & ~output.diagnostics['frame_valid_next']).any()):
        raise ValueError('predicted step invalidated a supervised frame')
    loss = masked_next_xyz_loss(
        output.protein_pos_next, batch.target_pos_next.detach(),
        batch.inputs.atom_to_residue_global, valid, batch.inputs.batch_protein,
    )
    return NextXYZObjective(output, loss)


def _gradient_norm(parameters):
    values = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def train_next_xyz_step(model: PocketDiffModel, batch: NextXYZBatch,
                        optimizer: torch.optim.Optimizer, *,
                        max_grad_norm: float = 1.) -> NextXYZTrainRecord:
    """One optimization step, with optimizer state owned by the caller.

    Preserve the caller's model mode. A nonfinite loss/gradient is rejected
    before optimizer.step(). Head gradients can be nonzero at zero init while
    the encoder starts receiving gradients only after head weights move.
    """
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError('max_grad_norm must be finite and positive')
    batch.validate()
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = {id(p) for group in optimizer.param_groups for p in group['params'] if p.requires_grad}
    if actual != expected:
        raise ValueError('optimizer must own exactly the trainable model parameters')
    was_training = model.training
    model.train()
    try:
        optimizer.zero_grad(set_to_none=True)
        objective = next_xyz_objective(model, batch)
        loss = objective.coordinate
        if not torch.isfinite(loss.loss):
            raise FloatingPointError('coordinate loss is non-finite')
        loss.loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('coordinate gradient is non-finite')
        rigid = model.motion_head.network[-1]
        gradient_norms = dict(
            encoder=_gradient_norm(model.encoder.parameters()),
            translation=float(torch.cat((rigid.weight.grad[:3].flatten(), rigid.bias.grad[:3])).norm()),
            rotation=float(torch.cat((rigid.weight.grad[3:].flatten(), rigid.bias.grad[3:])).norm()),
            chi=_gradient_norm(model.current_chi_head.parameters()),
        )
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm,
                                             error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('non-finite parameter after coordinate optimizer step')
        return NextXYZTrainRecord(float(loss.loss.detach()), loss.valid_atom_count,
                                  loss.valid_graph_count, float(norm), gradient_norms)
    finally:
        model.train(was_training)


__all__ = ['NextXYZBatch', 'NextXYZObjective', 'NextXYZTrainRecord',
           'next_xyz_objective', 'train_next_xyz_step']
