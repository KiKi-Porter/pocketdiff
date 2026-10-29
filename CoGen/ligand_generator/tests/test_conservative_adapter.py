from pathlib import Path

import torch
import yaml

from models.conservative_adapter import ConservativeCrossAdapter
from models.molopt_score_model import ScorePosNet3D


ROOT = Path(__file__).resolve().parents[1]


def _load_student_config():
    with (ROOT / "configs" / "conservative_adapter_md300.yml").open() as handle:
        return yaml.safe_load(handle)


def test_adapter_zero_initialized():
    adapter = ConservativeCrossAdapter(
        hidden_dim=16,
        num_classes=13,
        k=4,
        num_r_gaussian=8,
    )
    hidden = torch.randn(6, 16)
    positions = torch.randn(6, 3)
    mask_ligand = torch.tensor([False, False, False, True, True, True])
    batch = torch.zeros(6, dtype=torch.long)
    delta_pos, delta_type = adapter(
        hidden,
        positions,
        mask_ligand,
        batch,
        time_step=torch.tensor([333]),
        num_timesteps=1000,
    )
    assert torch.equal(delta_pos, torch.zeros_like(delta_pos))
    assert torch.equal(delta_type, torch.zeros_like(delta_type))


def test_adapter_position_is_rotation_equivariant():
    torch.manual_seed(7)
    adapter = ConservativeCrossAdapter(
        hidden_dim=16,
        num_classes=13,
        k=4,
        num_r_gaussian=8,
    )
    nn = adapter.direction_head
    nn.weight.data.normal_(std=0.1)
    nn.bias.data.normal_(std=0.1)
    hidden = torch.randn(6, 16)
    positions = torch.randn(6, 3)
    mask_ligand = torch.tensor([False, False, False, True, True, True])
    batch = torch.zeros(6, dtype=torch.long)
    rotation = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    delta_pos, _ = adapter(
        hidden,
        positions,
        mask_ligand,
        batch,
        time_step=torch.tensor([333]),
        num_timesteps=1000,
    )
    delta_rot, _ = adapter(
        hidden,
        positions @ rotation.T,
        mask_ligand,
        batch,
        time_step=torch.tensor([333]),
        num_timesteps=1000,
    )
    assert torch.allclose(delta_rot, delta_pos @ rotation.T, atol=1e-5)


def test_original_checkpoint_is_exact_noop_with_new_config():
    checkpoint_path = ROOT / "pretrained_models" / "pretrained_diffusion.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    student_config = _load_student_config()
    from easydict import EasyDict

    student = ScorePosNet3D(
        EasyDict(student_config["model"]),
        protein_atom_feature_dim=27,
        ligand_atom_feature_dim=13,
    )
    missing, unexpected = student.load_state_dict(
        payload["model"],
        strict=False,
    )
    assert not unexpected
    assert missing
    assert all(key.startswith("conservative_adapter.") for key in missing)

    sample = torch.load(
        ROOT / "data" / "md300" / "prepared_targetdiff_md300_10frames.pt",
        map_location="cpu",
        weights_only=False,
    )["samples"][0]
    protein_pos = sample["protein_pos"].float()
    protein_v = sample["protein_atom_feature"].float()
    ligand_pos = sample["ligand_pos"].float()
    ligand_v = sample["ligand_atom_feature_full"].long()
    batch_protein = torch.zeros(protein_pos.size(0), dtype=torch.long)
    batch_ligand = torch.zeros(ligand_pos.size(0), dtype=torch.long)
    with torch.no_grad():
        output = student(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            init_ligand_pos=ligand_pos,
            init_ligand_v=ligand_v,
            batch_ligand=batch_ligand,
            time_step=torch.tensor([333]),
        )
    assert torch.allclose(
        output["pred_ligand_pos"],
        output["base_pred_ligand_pos"],
        atol=1e-7,
    )
    assert torch.allclose(
        output["pred_ligand_v"],
        output["base_pred_ligand_v"],
        atol=1e-7,
    )
