import torch

from pocketdiff_v4.model import PocketDiffV4Model


def _sample():
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
    return {
        "input": {
            "apo_pos": apo,
            "residue_type": torch.tensor([5]),
            "residue_feature": torch.randn(1, 27),
            "ligand_pos": torch.tensor([[2.0, 0.4, -0.2], [2.7, 0.6, 0.1]]),
            "ligand_type": torch.tensor([1, 4]),
            "frame_index": torch.tensor([[0, 1, 2]]),
            "atom_to_residue": torch.zeros(5, dtype=torch.long),
            "rr_edge_index": torch.empty((2, 0), dtype=torch.long),
            "lr_edge_index": torch.tensor([[0, 1], [0, 0]]),
            "ll_edge_index": torch.tensor([[0, 1], [1, 0]]),
            "chi_geometry_mask": torch.zeros((1, 5), dtype=torch.bool),
        }
    }


def _rotation():
    axis = torch.tensor([0.3, -0.7, 0.2])
    axis = axis / torch.linalg.vector_norm(axis)
    angle = torch.tensor(0.83)
    x, y, z = axis
    skew = torch.tensor(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=torch.float32
    )
    return torch.eye(3) + torch.sin(angle) * skew + (1 - torch.cos(angle)) * (skew @ skew)


def test_model_forward_backward_and_rotation_equivariance():
    torch.manual_seed(7)
    sample = _sample()
    model = PocketDiffV4Model(hidden=48, vector_channels=6, layers=2, radial_count=8)
    with torch.no_grad():
        model.translation_gate[-1].weight.normal_(std=0.04)
        model.rotation_gate[-1].weight.normal_(std=0.04)
        model.chi_head[-1].weight.normal_(std=0.04)

    apo = sample["input"]["apo_pos"]
    current = apo.clone()
    current[3:] += torch.tensor([0.1, -0.2, 0.15])
    output = model(sample, current, torch.tensor(0.4))
    loss = sum(output[key].square().mean() for key in ("translation_local", "rotation_local", "chi_delta"))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    q = _rotation()
    shift = torch.tensor([-0.8, 0.3, 1.1])
    transformed_sample = {"input": dict(sample["input"])}
    transformed_sample["input"]["apo_pos"] = apo @ q.T + shift
    transformed_sample["input"]["ligand_pos"] = sample["input"]["ligand_pos"] @ q.T + shift
    transformed = current @ q.T + shift
    output_rotated = model(transformed_sample, transformed, torch.tensor(0.4))
    assert torch.allclose(
        output["translation_local"], output_rotated["translation_local"], atol=2e-5
    )
    assert torch.allclose(
        output["rotation_local"], output_rotated["rotation_local"], atol=2e-5
    )
    assert torch.allclose(output["chi_delta"], output_rotated["chi_delta"], atol=2e-5)


def test_default_rigid_heads_have_nonzero_gradients():
    torch.manual_seed(11)
    sample = _sample()
    model = PocketDiffV4Model(hidden=48, vector_channels=6, layers=2, radial_count=8)

    apo = sample["input"]["apo_pos"]
    current = apo.clone()
    current[3:] += torch.tensor([0.1, -0.2, 0.15])
    output = model(sample, current, torch.tensor(0.4))
    target_translation = torch.tensor([[0.15, -0.08, 0.05]])
    target_rotation = torch.tensor([[0.04, -0.02, 0.03]])
    loss = (
        (output["translation_local"] - target_translation).square().sum()
        + (output["rotation_local"] - target_rotation).square().sum()
    )
    loss.backward()

    translation_grad = model.translation_gate[-1].weight.grad
    rotation_grad = model.rotation_gate[-1].weight.grad
    assert translation_grad is not None
    assert rotation_grad is not None
    assert torch.isfinite(translation_grad).all()
    assert torch.isfinite(rotation_grad).all()
    assert translation_grad.abs().sum() > 0
    assert rotation_grad.abs().sum() > 0
