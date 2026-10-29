from pathlib import Path

import torch
import yaml

from models.layerwise_cross import LayerwiseCrossInteraction
from models.molopt_score_model import ScorePosNet3D


ROOT = Path(__file__).resolve().parents[1]


def _config():
    with (ROOT / "configs" / "internal_cross_md300.yml").open() as handle:
        payload = yaml.safe_load(handle)
    from easydict import EasyDict

    return EasyDict(payload["model"])


def _inputs():
    torch.manual_seed(11)
    hidden = torch.randn(8, 16)
    positions = torch.randn(8, 3)
    mask_ligand = torch.tensor(
        [False, False, False, False, True, True, True, True]
    )
    batch = torch.zeros(8, dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [4, 5, 6, 7, 0, 1, 2, 3],
        ],
        dtype=torch.long,
    )
    edge_attr = torch.eye(4).repeat(2, 1)
    return hidden, positions, mask_ligand, batch, edge_attr, edge_index


def test_zero_initialized_internal_cross_is_noop():
    module = LayerwiseCrossInteraction(
        hidden_dim=16,
        edge_feat_dim=4,
        num_r_gaussian=8,
    )
    output_h, output_x = module(
        *_inputs(),
        time_step=torch.tensor([333]),
        num_timesteps=1000,
    )
    assert torch.equal(output_h, torch.zeros_like(output_h))
    assert torch.equal(output_x, torch.zeros_like(output_x))


def test_internal_cross_coordinate_update_is_equivariant():
    module = LayerwiseCrossInteraction(
        hidden_dim=16,
        edge_feat_dim=4,
        num_r_gaussian=8,
    )
    with torch.no_grad():
        module.coord_gate.fill_(0.5)
        module.coord_message.net[-1].weight.normal_(std=0.1)
    values = _inputs()
    output_h, output_x = module(
        *values,
        time_step=torch.tensor([333]),
        num_timesteps=1000,
    )
    rotation = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = list(values)
    rotated[1] = rotated[1] @ rotation.T
    rotated_h, rotated_x = module(
        *rotated,
        time_step=torch.tensor([333]),
        num_timesteps=1000,
    )
    assert torch.allclose(rotated_h, output_h, atol=1e-5)
    assert torch.allclose(rotated_x, output_x @ rotation.T, atol=1e-5)


def test_conservative_checkpoint_loads_with_only_internal_branch_missing():
    checkpoint = ROOT / "runs" / "conservative_adapter_md300_20261002" / "adapter_best.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ScorePosNet3D(
        _config(),
        protein_atom_feature_dim=27,
        ligand_atom_feature_dim=13,
    )
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    assert not unexpected
    assert missing
    assert all(key.startswith("refine_net.internal_cross.") for key in missing)


def test_new_model_matches_conservative_checkpoint_at_zero_gate():
    checkpoint = ROOT / "runs" / "conservative_adapter_md300_20261002" / "adapter_best.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    from easydict import EasyDict

    old_payload = yaml.safe_load(
        (ROOT / "configs" / "conservative_adapter_md300.yml").read_text()
    )
    old_model = ScorePosNet3D(
        EasyDict(old_payload["model"]),
        protein_atom_feature_dim=27,
        ligand_atom_feature_dim=13,
    )
    old_model.load_state_dict(payload["model"], strict=True)

    new_model = ScorePosNet3D(
        _config(),
        protein_atom_feature_dim=27,
        ligand_atom_feature_dim=13,
    )
    missing, unexpected = new_model.load_state_dict(
        payload["model"], strict=False
    )
    assert not unexpected
    assert missing
    old_model.eval()
    new_model.eval()

    sample = torch.load(
        ROOT / "data" / "md300" / "prepared_targetdiff_md300_10frames.pt",
        map_location="cpu",
        weights_only=False,
    )["samples"][0]
    protein_pos = sample["protein_pos"].float()
    ligand_pos = sample["ligand_pos"].float()
    batch_protein = torch.zeros(protein_pos.size(0), dtype=torch.long)
    batch_ligand = torch.zeros(ligand_pos.size(0), dtype=torch.long)
    args = {
        "protein_pos": protein_pos,
        "protein_v": sample["protein_atom_feature"].float(),
        "batch_protein": batch_protein,
        "init_ligand_pos": ligand_pos,
        "init_ligand_v": sample["ligand_atom_feature_full"].long(),
        "batch_ligand": batch_ligand,
        "time_step": torch.tensor([777]),
    }
    with torch.no_grad():
        old_output = old_model(**args)
        new_output = new_model(**args)
    assert torch.equal(
        old_output["pred_ligand_pos"],
        new_output["pred_ligand_pos"],
    )
    assert torch.equal(
        old_output["pred_ligand_v"],
        new_output["pred_ligand_v"],
    )
