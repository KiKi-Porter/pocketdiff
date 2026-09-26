import math
import random

import torch

from pocketdiff_v4.batching import collate_complexes
from pocketdiff_v4.geometry import apply_motion, axis_angle_matrix, residue_frames
from pocketdiff_v4.model import PocketDiffV4Model
from pocketdiff_v4.train import (
    _bridge_step_targets,
    _graph_balanced_mean,
    _rollout_loss,
)
from pocketdiff.geometry.bridge import remaining_transform_current_to_holo


def _record(sample_id="rollout"):
    apo = torch.tensor(
        [
            [-0.4, 0.2, 0.1],
            [0.0, 0.0, 0.0],
            [0.2, 0.8, -0.1],
            [0.3, -0.5, 0.4],
            [0.8, -0.2, 0.7],
        ],
        dtype=torch.float32,
    )
    rotation = axis_angle_matrix(torch.tensor([[0.13, -0.09, 0.17]]))[0]
    shift = torch.tensor([0.25, -0.12, 0.2])
    holo = (apo - apo[1]) @ rotation.T + apo[1] + shift
    return {
        "sample_id": sample_id,
        "input": {
            "apo_pos": apo,
            "protein_feature": torch.zeros((5, 27)),
            "ligand_pos": torch.tensor([[2.0, 0.4, -0.2], [2.7, 0.6, 0.1]]),
            "ligand_type": torch.tensor([1, 4]),
            "atom_to_residue": torch.zeros(5, dtype=torch.long),
            "residue_type": torch.tensor([5]),
            "residue_feature": torch.zeros((1, 27)),
            "residue_center_apo": apo.mean(0, keepdim=True),
            "frame_index": torch.tensor([[0, 1, 2]]),
            "chi_geometry_mask": torch.zeros((1, 5), dtype=torch.bool),
            "chi_axis": torch.full((1, 5, 2), -1, dtype=torch.long),
            "chi_ptr": torch.zeros(6, dtype=torch.long),
            "chi_downstream": torch.empty(0, dtype=torch.long),
            "rr_edge_index": torch.empty((2, 0), dtype=torch.long),
            "lr_edge_index": torch.tensor([[0, 1], [0, 0]]),
            "ll_edge_index": torch.tensor([[0, 1], [1, 0]]),
        },
        "target": {
            "holo_pos": holo,
            "chi_apo": torch.zeros((1, 5)),
            "chi_holo": torch.tensor([[0.7, 0.0, 0.0, 0.0, 0.0]]),
            "chi_supervision_mask": torch.tensor(
                [[True, False, False, False, False]]
            ),
        },
    }


def _model():
    return PocketDiffV4Model(
        hidden=32, vector_channels=4, layers=1, radial_count=8
    )


def test_bridge_targets_are_complete_remaining_transforms():
    record = _record()
    batch = collate_complexes([record])
    inputs, targets = batch["input"], batch["target"]
    apo = inputs["apo_pos"]
    far_holo = targets["holo_pos"] + torch.tensor([8.0, -3.0, 2.0])
    targets = dict(targets)
    targets["holo_pos"] = far_holo
    translation, rotation, chi, valid, chi_mask = _bridge_step_targets(
        _model(),
        inputs,
        targets,
        apo,
        torch.zeros((1, 5)),
        remaining_steps=1,
    )
    assert valid.tolist() == [True]
    expected = remaining_transform_current_to_holo(
        *residue_frames(apo, inputs["frame_index"])[:2],
        *residue_frames(far_holo, inputs["frame_index"])[:2],
        frame_valid=valid,
    )
    assert torch.allclose(translation, expected.translation_local)
    assert torch.allclose(rotation, expected.rotvec_local)
    assert chi_mask[0, 0]
    assert torch.allclose(chi[0, 0], torch.tensor(0.7), atol=1e-6)


def test_remaining_target_is_independent_of_rollout_length():
    record = _record()
    batch = collate_complexes([record])
    inputs, targets = batch["input"], batch["target"]
    current = inputs["apo_pos"]
    outputs = [
        _bridge_step_targets(
            _model(),
            inputs,
            targets,
            current,
            torch.zeros((1, 5)),
            remaining_steps=steps,
        )[:3]
        for steps in (1, 2, 4, 20)
    ]
    for actual in outputs[1:]:
        for reference, candidate in zip(outputs[0], actual):
            assert torch.allclose(reference, candidate)


def test_bridge_solver_reaches_rigid_endpoint_across_rotated_local_frames():
    record = _record()
    sample = record["input"]
    current = sample["apo_pos"].clone()
    target = record["target"]["holo_pos"]
    target_origin, target_frame, target_valid = residue_frames(
        target, sample["frame_index"]
    )
    for remaining in range(4, 0, -1):
        current_origin, current_frame, current_valid = residue_frames(
            current, sample["frame_index"]
        )
        bridge = remaining_transform_current_to_holo(
            current_origin,
            current_frame,
            target_origin,
            target_frame,
            frame_valid=current_valid & target_valid,
        )
        current = apply_motion(
            sample,
            current,
            bridge.translation_local,
            bridge.rotvec_local,
            torch.zeros((1, 5)),
            fraction=1.0 / remaining,
        )
    assert torch.allclose(current, target, atol=2e-5, rtol=2e-5)


def test_bridge_solver_reaches_same_endpoint_for_multiple_step_counts():
    record = _record()
    sample = record["input"]
    target = record["target"]["holo_pos"]
    target_origin, target_frame, target_valid = residue_frames(
        target, sample["frame_index"]
    )
    endpoints = []
    for steps in (1, 2, 4, 20):
        current = sample["apo_pos"].clone()
        for step in range(steps):
            current_origin, current_frame, current_valid = residue_frames(
                current, sample["frame_index"]
            )
            bridge = remaining_transform_current_to_holo(
                current_origin,
                current_frame,
                target_origin,
                target_frame,
                frame_valid=current_valid & target_valid,
            )
            current = apply_motion(
                sample,
                current,
                bridge.translation_local,
                bridge.rotvec_local,
                torch.zeros((1, 5)),
                fraction=1.0 / (steps - step),
            )
        endpoints.append(current)
    for endpoint in endpoints[1:]:
        assert torch.allclose(endpoint, endpoints[0], atol=3e-5, rtol=3e-5)


def test_graph_balanced_loss_weights_complexes_equally():
    values = torch.tensor([1.0, 1.0, 3.0, 3.0, 3.0, 3.0])
    graph_index = torch.tensor([0, 0, 1, 1, 1, 1])
    balanced = _graph_balanced_mean(values, graph_index, graph_count=2)
    assert torch.allclose(balanced, torch.tensor(2.0))


def test_multistep_rollout_loss_has_finite_backward():
    random.seed(19)
    torch.manual_seed(19)
    batch = collate_complexes([_record()])
    model = _model()
    loss, metrics = _rollout_loss(model, batch, max_steps=2)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["endpoint_loss"])
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_final_endpoint_weight_is_outside_step_average():
    torch.manual_seed(31)
    batch = collate_complexes([_record("endpoint-weight")])
    batch["input"]["backbone_mask"] = torch.ones(
        batch["input"]["apo_pos"].shape[0], dtype=torch.bool
    )
    model = _model()
    common = {
        "step_count": 2,
        "disable_chi": True,
        "schedule_type": "fixed",
        "fixed_fraction": 0.2,
    }
    without_endpoint, _ = _rollout_loss(
        model, batch, max_steps=2, final_endpoint_weight=0.0, **common
    )
    with_endpoint, metrics = _rollout_loss(
        model, batch, max_steps=2, final_endpoint_weight=1.0, **common
    )
    assert metrics["endpoint_loss"] > 0
    assert torch.allclose(
        with_endpoint - without_endpoint,
        metrics["endpoint_loss"] + metrics["truncated_endpoint_loss"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_backbone_only_objective_ignores_sidechain_holo_changes():
    torch.manual_seed(37)
    record = _record("backbone-mask")
    batch = collate_complexes([record])
    batch["input"]["backbone_mask"] = torch.tensor(
        [True, True, True, True, False], dtype=torch.bool
    )
    changed = {
        "input": batch["input"],
        "target": dict(batch["target"]),
    }
    changed["target"]["holo_pos"] = batch["target"]["holo_pos"].clone()
    changed["target"]["holo_pos"][-1] += torch.tensor([7.0, -4.0, 3.0])
    kwargs = {
        "max_steps": 2,
        "step_count": 2,
        "disable_chi": True,
        "direction_weight": 0.0,
        "ca_direction_weight": 0.0,
        "backbone_only_objective": True,
        "schedule_type": "fixed",
        "fixed_fraction": 0.2,
    }
    model = _model()
    torch.manual_seed(101)
    baseline_loss, baseline_metrics = _rollout_loss(model, batch, **kwargs)
    torch.manual_seed(101)
    changed_loss, changed_metrics = _rollout_loss(model, changed, **kwargs)
    assert torch.allclose(baseline_loss, changed_loss, atol=1e-6, rtol=1e-6)
    for key in ("bridge_loss", "final_endpoint_loss", "truncated_endpoint_loss"):
        assert torch.allclose(
            baseline_metrics[key], changed_metrics[key], atol=1e-6, rtol=1e-6
        )


def test_truncated_bptt_reaches_the_preceding_rollout_step():
    torch.manual_seed(43)
    batch = collate_complexes([_record("bptt")])
    batch["input"]["backbone_mask"] = torch.ones(
        batch["input"]["apo_pos"].shape[0], dtype=torch.bool
    )

    detached_model = _model()
    detached_loss, _ = _rollout_loss(
        detached_model,
        batch,
        max_steps=2,
        step_count=2,
        disable_chi=True,
        direction_weight=0.0,
        final_endpoint_weight=1.0,
        bptt_steps=0,
        schedule_type="remaining",
    )
    detached_loss.backward()
    detached_grad = detached_model.translation_gate[-1].weight.grad.detach().clone()

    bptt_model = _model()
    bptt_model.load_state_dict(detached_model.state_dict())
    bptt_loss, _ = _rollout_loss(
        bptt_model,
        batch,
        max_steps=2,
        step_count=2,
        disable_chi=True,
        direction_weight=0.0,
        final_endpoint_weight=1.0,
        bptt_steps=2,
        schedule_type="remaining",
    )
    bptt_loss.backward()
    bptt_grad = bptt_model.translation_gate[-1].weight.grad.detach()

    assert torch.isfinite(bptt_grad).all()
    assert not torch.allclose(detached_grad, bptt_grad)


def test_rollout_loss_reports_consistent_component_decomposition():
    torch.manual_seed(37)
    batch = collate_complexes([_record("loss-components")])
    batch["input"]["backbone_mask"] = torch.ones(
        batch["input"]["apo_pos"].shape[0], dtype=torch.bool
    )
    model = _model()
    loss, metrics = _rollout_loss(
        model,
        batch,
        max_steps=2,
        step_count=2,
        disable_chi=True,
        direction_weight=0.0,
        final_endpoint_weight=1.0,
        schedule_type="remaining",
    )
    expected = (
        0.2 * metrics["rigid_loss"]
        + 0.75 * metrics["bridge_loss"]
        + 0.1 * metrics["ca_motion_loss"]
        + 0.05 * metrics["ca_direction_loss"]
        + metrics["final_endpoint_loss"]
        + metrics["truncated_endpoint_loss"]
    )
    assert torch.allclose(loss.detach(), expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        metrics["endpoint_loss"], metrics["final_endpoint_loss"]
    )
    assert torch.equal(metrics["direction_loss"], torch.zeros_like(metrics["direction_loss"]))


def test_ca_motion_and_direction_losses_reach_rigid_heads():
    torch.manual_seed(41)
    batch = collate_complexes([_record("ca-motion-loss")])
    batch["input"]["backbone_mask"] = torch.ones(
        batch["input"]["apo_pos"].shape[0], dtype=torch.bool
    )
    model = _model()
    loss, metrics = _rollout_loss(
        model,
        batch,
        max_steps=2,
        step_count=2,
        disable_chi=True,
        direction_weight=0.0,
        ca_motion_weight=0.1,
        ca_direction_weight=0.05,
        schedule_type="remaining",
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["ca_motion_loss"] > 0
    assert metrics["ca_direction_loss"] > 0
    assert model.translation_gate[-1].weight.grad is not None
    assert model.rotation_gate[-1].weight.grad is not None
    assert model.translation_gate[-1].weight.grad.abs().sum() > 0
    assert model.rotation_gate[-1].weight.grad.abs().sum() > 0


def test_oracle_rigid_loss_trains_rigid_heads_without_chi_gradients():
    random.seed(23)
    torch.manual_seed(23)
    batch = collate_complexes([_record()])
    model = _model()
    loss, metrics = _rollout_loss(
        model,
        batch,
        max_steps=2,
        oracle_rollout=True,
        disable_chi=True,
        direction_weight=0.1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["endpoint_loss"])
    assert model.translation_gate[-1].weight.grad is not None
    assert model.rotation_gate[-1].weight.grad is not None
    assert model.translation_gate[-1].weight.grad.abs().sum() > 0
    assert model.rotation_gate[-1].weight.grad.abs().sum() > 0
    chi_grad = model.chi_head[-1].weight.grad
    assert chi_grad is None or torch.allclose(chi_grad, torch.zeros_like(chi_grad))
