import torch

from pocketdiff.data.schema import PocketDiffPrediction, ResidueMetadata
from pocketdiff.geometry.bridge import apply_fractional_update
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.geometry.so3 import so3_exp
from pocketdiff.sampling import PocketStepSolver, pocket_step
from pocketdiff.targetdiff import initialize_targetdiff_state


def _state_and_metadata():
    protein_pos = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    feature = torch.zeros(3, 27)
    feature[:, 1] = 1.0
    state = initialize_targetdiff_state(
        protein_pos=protein_pos,
        protein_v=feature,
        batch_protein=torch.zeros(3, dtype=torch.long),
        ligand_pos=torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32),
        ligand_v=torch.tensor([1], dtype=torch.long),
        batch_ligand=torch.zeros(1, dtype=torch.long),
        apo_pos_ref=protein_pos,
    )
    metadata = ResidueMetadata(
        protein_atom_name=[["N", "CA", "C"]],
        protein_residue_name=[["ALA", "ALA", "ALA"]],
        residue_type=torch.tensor([0], dtype=torch.long),
        atom_to_residue_global=torch.zeros(3, dtype=torch.long),
        batch_residue=torch.zeros(1, dtype=torch.long),
        frame_valid_reference=torch.tensor([True]),
        chi_mask=torch.zeros(1, 5, dtype=torch.bool),
        chain_id=[["A"]],
        residue_sequence_id=[["1"]],
    )
    return state, metadata


def _batched_state_and_metadata():
    one_state, _ = _state_and_metadata()
    protein = torch.cat((one_state.protein_pos + torch.tensor([0.0, 0.0, 0.0]), one_state.protein_pos + torch.tensor([4.0, 1.0, 0.0])), dim=0)
    apo = protein.clone()
    feature = torch.zeros(6, 27)
    feature[:, 1] = 1.0
    state = initialize_targetdiff_state(
        protein_pos=protein,
        protein_v=feature,
        batch_protein=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        ligand_pos=torch.tensor([[0.5, 0.5, 0.5], [4.5, 1.5, 0.5]], dtype=torch.float32),
        ligand_v=torch.tensor([1, 1], dtype=torch.long),
        batch_ligand=torch.tensor([0, 1], dtype=torch.long),
        apo_pos_ref=apo,
    )
    metadata = ResidueMetadata(
        protein_atom_name=[["N", "CA", "C"], ["N", "CA", "C"]],
        protein_residue_name=[["ALA"] * 3, ["ALA"] * 3],
        residue_type=torch.tensor([0, 0], dtype=torch.long),
        atom_to_residue_global=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        batch_residue=torch.tensor([0, 1], dtype=torch.long),
        frame_valid_reference=torch.tensor([True, True]),
        chi_mask=torch.zeros(2, 5, dtype=torch.bool),
        chain_id=[["A"], ["B"]],
        residue_sequence_id=[["1"], ["1"]],
    )
    return state, metadata


class _FixedMotionModel(torch.nn.Module):
    def __init__(self, translation, rotvec=None):
        super().__init__()
        self.translation = torch.nn.Parameter(torch.tensor(translation, dtype=torch.float32).reshape(1, 3))
        self.rotvec = torch.nn.Parameter(
            torch.zeros(1, 3) if rotvec is None else torch.tensor(rotvec, dtype=torch.float32).reshape(1, 3)
        )

    def forward(self, **kwargs):
        num_residues = kwargs["residue_type"].shape[0]
        valid = kwargs["frame_valid"]
        return PocketDiffPrediction(
            remaining_translation_local=self.translation.expand(num_residues, -1),
            remaining_rotvec_local=self.rotvec.expand(num_residues, -1),
            remaining_chi=None,
            frame_valid=valid,
            diagnostics={},
        )


def test_zero_prediction_is_bitwise_unchanged_and_preserves_state():
    state, metadata = _state_and_metadata()
    model = _FixedMotionModel([0.0, 0.0, 0.0])
    before = {name: getattr(state, name).clone() for name in (
        "protein_pos", "protein_v", "batch_protein", "ligand_pos", "ligand_v",
        "batch_ligand", "apo_pos_ref", "center_offset"
    )}
    output = pocket_step(model, state, metadata, k=0)
    assert torch.equal(output.protein_pos_next, state.protein_pos)
    assert torch.equal(output.applied_translation_local, torch.zeros(1, 3))
    assert torch.equal(output.applied_rotvec_local, torch.zeros(1, 3))
    for name, value in before.items():
        assert torch.equal(getattr(state, name), value), name


def test_remaining_steps_fractionates_translation_and_one_step_matches_geometry():
    state, metadata = _state_and_metadata()
    model = _FixedMotionModel([0.4, 0.0, 0.0])
    output = pocket_step(model, state, metadata, k=0)
    assert torch.allclose(output.applied_translation_local, torch.tensor([[0.02, 0.0, 0.0]]))
    frames = build_residue_frames(
        state.protein_pos,
        metadata.atom_to_residue_global,
        ["N", "CA", "C"],
        num_residues=1,
    )
    expected = apply_fractional_update(
        state.protein_pos,
        metadata.atom_to_residue_global,
        frames.origins,
        frames.frames,
        torch.tensor([[0.4, 0.0, 0.0]]),
        torch.zeros(1, 3),
        remaining_steps=20,
        frame_valid=torch.tensor([True]),
    )
    assert torch.allclose(output.protein_pos_next, expected, atol=1e-6)
    full = PocketStepSolver(model).step(state, metadata, k=19)
    expected_full = apply_fractional_update(
        state.protein_pos,
        metadata.atom_to_residue_global,
        frames.origins,
        frames.frames,
        torch.tensor([[0.4, 0.0, 0.0]]),
        torch.zeros(1, 3),
        remaining_steps=1,
        frame_valid=torch.tensor([True]),
    )
    assert torch.allclose(full.protein_pos_next, expected_full, atol=1e-6)


def test_batched_graphs_use_independent_remaining_step_counts():
    state, metadata = _batched_state_and_metadata()
    model = _FixedMotionModel([0.4, 0.0, 0.0])
    output = pocket_step(model, state, metadata, k=torch.tensor([0, 19], dtype=torch.long))
    assert torch.allclose(
        output.applied_translation_local,
        torch.tensor([[0.02, 0.0, 0.0], [0.4, 0.0, 0.0]]),
    )


def test_solver_is_global_se3_equivariant():
    state, metadata = _state_and_metadata()
    model = _FixedMotionModel([0.2, -0.1, 0.0], [0.0, 0.0, 0.15])
    base = pocket_step(model, state, metadata, k=3)
    rotation = so3_exp(torch.tensor([0.2, -0.1, 0.3]))
    shift = torch.tensor([5.0, -2.0, 1.0])
    transformed = state.replace(
        protein_pos=state.protein_pos @ rotation.transpose(0, 1) + shift,
        apo_pos_ref=state.apo_pos_ref @ rotation.transpose(0, 1) + shift,
        ligand_pos=state.ligand_pos @ rotation.transpose(0, 1) + shift,
    )
    moved = pocket_step(model, transformed, metadata, k=3)
    expected = base.protein_pos_next @ rotation.transpose(0, 1) + shift
    assert torch.allclose(moved.protein_pos_next, expected, atol=2e-4)


def test_solver_keeps_gradient_through_predicted_update():
    state, metadata = _state_and_metadata()
    model = _FixedMotionModel([0.1, 0.0, 0.0], [0.0, 0.0, 0.1])
    output = pocket_step(model, state, metadata, k=0)
    loss = output.protein_pos_next.square().sum()
    loss.backward()
    assert model.translation.grad is not None
    assert model.rotvec.grad is not None
    assert torch.isfinite(model.translation.grad).all()
    assert torch.isfinite(model.rotvec.grad).all()
    assert float(model.translation.grad.abs().sum()) > 0.0
