import torch

from pocketdiff.data.schema import PocketComplex
from pocketdiff.training.diffusion import (
    DiffusionTrainConfig,
    load_diffusion_checkpoint,
    sample_checkpoint_on_complex,
    save_diffusion_checkpoint,
    train_diffusion_model,
)


def _synthetic_complex():
    apo = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    holo = apo + torch.tensor([0.05, -0.02, 0.03], dtype=torch.float32)
    return PocketComplex(
        sample_id="synthetic",
        protein_pos_apo=apo,
        protein_pos_holo=holo,
        protein_feature=torch.zeros((4, 27), dtype=torch.float32),
        protein_element=torch.tensor([7, 6, 6, 8], dtype=torch.long),
        protein_atom_name=["N", "CA", "C", "O"],
        protein_residue_name=["ALA", "ALA", "ALA", "ALA"],
        atom_to_residue=torch.zeros(4, dtype=torch.long),
        residue_type=torch.tensor([0], dtype=torch.long),
        residue_chain_id=["A"],
        residue_sequence_id=["1"],
        frame_valid=torch.ones(1, dtype=torch.bool),
        chi_apo=torch.zeros((1, 5), dtype=torch.float32),
        chi_holo=torch.zeros((1, 5), dtype=torch.float32),
        chi_mask=torch.zeros((1, 5), dtype=torch.bool),
        ligand_pos_ref=torch.tensor([[0.5, 0.5, 2.0]], dtype=torch.float32),
        ligand_type_ref=torch.tensor([1], dtype=torch.long),
        center_offset=torch.zeros(3, dtype=torch.float32),
    )


def test_diffusion_training_components_save_reloadable_checkpoint(tmp_path):
    value = _synthetic_complex()
    config = DiffusionTrainConfig(
        train_count=1,
        holdout_count=0,
        updates=1,
        times=(0.5,),
        sampler_steps=(1,),
        diffusion_seed=66,
    )
    model, optimizer, rows = train_diffusion_model([value], config)
    assert len(rows) == 1
    assert rows[0]["finite"] is True
    report = {"checkpoint_reload_exact": True, "training_rows": rows}
    checkpoint = tmp_path / "checkpoint.pt"
    save_diffusion_checkpoint(checkpoint, model, optimizer, config=config, report=report)
    loaded = load_diffusion_checkpoint(checkpoint)
    assert loaded.payload["format"] == "pocketdiff-diffusion-v1"
    trajectory = sample_checkpoint_on_complex(checkpoint, value, steps=2)
    assert len(trajectory.states) == 3
    assert trajectory.times == (1.0, 0.5, 0.0)
    assert all(torch.isfinite(state).all() for state in trajectory.states)
