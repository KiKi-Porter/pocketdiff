import copy
import pickle
from pathlib import Path

import pytest
import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.models import PocketDiffModel
from pocketdiff.models.motion_head import ResidueMotionHead


def _batch_inputs():
    protein_pos = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    apo_pos_ref = protein_pos.clone()
    apo_pos_ref[:3] += torch.tensor([0.2, 0.1, 0.0])
    apo_pos_ref[3:] += torch.tensor([-0.1, 0.2, 0.0])
    protein_feature = torch.zeros(6, 27)
    protein_feature[:, 1] = 1.0  # carbon element channel
    protein_feature[[0, 3], 6] = 1.0
    protein_feature[[1, 4], 7] = 1.0
    protein_feature[[2, 5], 8] = 1.0
    ligand_pos = torch.tensor([[0.2, 0.2, 0.5], [3.2, 0.2, 0.5]], dtype=torch.float32)
    return {
        "protein_pos": protein_pos,
        "apo_pos_ref": apo_pos_ref,
        "protein_feature": protein_feature,
        "atom_to_residue_global": torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        "residue_type": torch.tensor([0, 1], dtype=torch.long),
        "frame_valid": torch.tensor([True, True]),
        "batch_protein": torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        "batch_residue": torch.tensor([0, 1], dtype=torch.long),
        "ligand_pos": ligand_pos,
        "ligand_v": torch.tensor([1, 3], dtype=torch.long),
        "batch_ligand": torch.tensor([0, 1], dtype=torch.long),
        "targetdiff_t": torch.tensor([199, 100], dtype=torch.long),
        "pocket_k": torch.tensor([0, 5], dtype=torch.long),
        "protein_atom_name": ["N", "CA", "C", "N", "CA", "C"],
    }


def _forward(model, inputs):
    kwargs = dict(inputs)
    names = kwargs.pop("protein_atom_name")
    return model(**kwargs, protein_atom_name=names)


def test_mvp_forward_contract_and_zero_initial_motion():
    inputs = _batch_inputs()
    model = PocketDiffModel(encoder_layers=1, knn=4)
    output = _forward(model, inputs)
    assert output.remaining_translation_local.shape == (2, 3)
    assert output.remaining_rotvec_local.shape == (2, 3)
    assert output.remaining_translation_local.dtype == torch.float32
    assert output.remaining_rotvec_local.dtype == torch.float32
    assert output.remaining_chi is None
    assert torch.equal(output.frame_valid, inputs["frame_valid"])
    assert torch.equal(output.remaining_translation_local, torch.zeros(2, 3))
    assert torch.equal(output.remaining_rotvec_local, torch.zeros(2, 3))
    assert torch.isfinite(output.remaining_translation_local).all()
    assert torch.isfinite(output.remaining_rotvec_local).all()


def test_forward_does_not_mutate_coordinates_features_or_ligand():
    inputs = _batch_inputs()
    before = {key: value.clone() for key, value in inputs.items() if isinstance(value, torch.Tensor)}
    names = list(inputs["protein_atom_name"])
    _forward(PocketDiffModel(encoder_layers=1, knn=4), inputs)
    for key, value in before.items():
        assert torch.equal(inputs[key], value), key
    assert inputs["protein_atom_name"] == names


def test_local_outputs_are_global_se3_invariant_after_head_unfrozen():
    inputs = _batch_inputs()
    model = PocketDiffModel(encoder_layers=1, knn=4)
    with torch.no_grad():
        model.motion_head.network[-1].weight.normal_(mean=0.0, std=0.02)
        model.motion_head.network[-1].bias.normal_(mean=0.0, std=0.01)
    model.eval()
    base = _forward(model, inputs)

    from pocketdiff.geometry.so3 import so3_exp

    rotation = so3_exp(torch.tensor([0.2, -0.1, 0.3]))
    shift = torch.tensor([5.0, -2.0, 1.0])
    transformed = dict(inputs)
    transformed["protein_pos"] = inputs["protein_pos"] @ rotation.transpose(0, 1) + shift
    transformed["apo_pos_ref"] = inputs["apo_pos_ref"] @ rotation.transpose(0, 1) + shift
    transformed["ligand_pos"] = inputs["ligand_pos"] @ rotation.transpose(0, 1) + shift
    moved = _forward(model, transformed)
    assert torch.allclose(base.remaining_translation_local, moved.remaining_translation_local, atol=2e-4)
    assert torch.allclose(base.remaining_rotvec_local, moved.remaining_rotvec_local, atol=2e-4)


def test_mvp_has_finite_nonzero_gradients_when_motion_head_is_trainable():
    inputs = _batch_inputs()
    model = PocketDiffModel(encoder_layers=1, knn=4)
    with torch.no_grad():
        model.motion_head.network[-1].weight.fill_(0.01)
    output = _forward(model, inputs)
    loss = output.remaining_translation_local.square().sum() + output.remaining_rotvec_local.square().sum()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(float(gradient.abs().sum()) > 0.0 for gradient in gradients)


def test_zero_initialized_rotvec_path_keeps_a_finite_learning_gradient():
    head = ResidueMotionHead()
    descriptor = torch.randn(2, 283, requires_grad=True)
    _, rotvec = head(descriptor)
    target = torch.tensor([[0.2, -0.1, 0.3], [-0.1, 0.2, 0.1]])
    loss = (rotvec - target).square().sum()
    loss.backward()
    assert head.network[-1].weight.grad is not None
    assert torch.isfinite(head.network[-1].weight.grad).all()
    assert float(head.network[-1].weight.grad.abs().sum()) > 0.0


def test_real_apo2mol_sample_completes_mvp_forward():
    root = Path("Apo2Mol-main/Apo2MOl-dataset/data_folder")
    split_path = root.parent / "split_druglike_dict.pkl"
    if not root.is_dir() or not split_path.is_file():
        pytest.skip("Apo2Mol raw dataset is not available")
    with split_path.open("rb") as handle:
        record = pickle.load(handle)["train"][0]
    complex_value = Apo2MolAdapter(root).convert_record(record)
    output = PocketDiffModel(encoder_layers=1, knn=8).forward_complex(complex_value)
    assert output.remaining_translation_local.shape == (complex_value.num_residues, 3)
    assert output.remaining_rotvec_local.shape == (complex_value.num_residues, 3)
    assert bool(torch.isfinite(output.remaining_translation_local).all())
    assert bool(torch.isfinite(output.remaining_rotvec_local).all())
    assert bool(output.frame_valid.all())
