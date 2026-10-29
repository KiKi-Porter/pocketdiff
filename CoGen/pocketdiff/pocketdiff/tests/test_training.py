import math

import torch

from pocketdiff.data.schema import PocketComplex
from pocketdiff.models import PocketDiffModel
from pocketdiff.training import (
    collate_clean_examples,
    make_clean_example,
    masked_remaining_motion_loss,
    save_checkpoint,
    train_clean_batch,
)


def _complex(sample_id: str, angle: float, shift):
    apo = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    c, s = math.cos(angle), math.sin(angle)
    rotation = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    holo = apo @ rotation.transpose(0, 1) + torch.tensor(shift, dtype=torch.float32)
    feature = torch.zeros(4, 27)
    feature[:, 1] = 1.0
    feature[:, 6] = 1.0
    return PocketComplex(
        sample_id=sample_id,
        protein_pos_apo=apo,
        protein_pos_holo=holo,
        protein_feature=feature,
        protein_element=torch.tensor([6, 6, 6, 6]),
        protein_atom_name=["N", "CA", "C", "CB"],
        protein_residue_name=["ALA"] * 4,
        atom_to_residue=torch.zeros(4, dtype=torch.long),
        residue_type=torch.tensor([0]),
        residue_chain_id=["A"],
        residue_sequence_id=["1"],
        frame_valid=torch.tensor([True]),
        chi_apo=torch.zeros(1, 5),
        chi_holo=torch.zeros(1, 5),
        chi_mask=torch.zeros(1, 5, dtype=torch.bool),
        ligand_pos_ref=torch.tensor([[0.5, 0.5, 0.5]]),
        ligand_type_ref=torch.tensor([1]),
        center_offset=torch.zeros(3),
    )


def _batch():
    return collate_clean_examples(
        [
            make_clean_example(_complex("a", 0.2, [0.2, 0.1, 0.0])),
            make_clean_example(_complex("b", -0.3, [-0.1, 0.4, 0.0])),
        ]
    )


def test_clean_batch_and_masked_loss_contract():
    batch = _batch()
    assert batch.protein_pos.shape == (8, 3)
    assert batch.atom_to_residue_global.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert batch.batch_residue.tolist() == [0, 1]
    model = PocketDiffModel(encoder_layers=1, knn=4)
    prediction = model(**batch.model_kwargs())
    motion_loss = masked_remaining_motion_loss(
        prediction,
        batch.target_translation_local,
        batch.target_rotvec_local,
        batch.frame_valid,
    )
    assert motion_loss.valid_residue_count == 2
    assert torch.isfinite(motion_loss.loss)


def test_clean_trainer_reduces_loss_and_checkpoint_reloads(tmp_path):
    torch.manual_seed(7)
    batch = _batch()
    model = PocketDiffModel(encoder_layers=1, knn=4)
    initial = masked_remaining_motion_loss(
        model(**batch.model_kwargs()),
        batch.target_translation_local,
        batch.target_rotvec_local,
        batch.frame_valid,
    ).loss.item()
    records = train_clean_batch(model, batch, steps=40, learning_rate=3e-3, log_every=20)
    assert records[-1].loss < initial
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        model,
        config={"encoder_layers": 1, "knn": 4},
        sample_ids=batch.sample_ids,
        final_record=records[-1],
    )
    payload = torch.load(checkpoint, map_location="cpu")
    restored = PocketDiffModel(encoder_layers=1, knn=4)
    restored.load_state_dict(payload["model_state_dict"])
    model.eval()
    restored.eval()
    with torch.no_grad():
        original = model(**batch.model_kwargs())
        reloaded = restored(**batch.model_kwargs())
    assert torch.allclose(original.remaining_translation_local, reloaded.remaining_translation_local)
    assert torch.allclose(original.remaining_rotvec_local, reloaded.remaining_rotvec_local)
