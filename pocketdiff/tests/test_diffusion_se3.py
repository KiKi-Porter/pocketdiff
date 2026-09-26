import pickle
from pathlib import Path

import pytest
import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.diffusion.se3 import apply_local_se3_update
from pocketdiff.diffusion.state import sample_diffusion_state
from pocketdiff.geometry.chi import apply_chi_updates
from pocketdiff.geometry.current_state import build_current_chi_state
from pocketdiff.geometry.bridge import apply_fractional_update
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.geometry.so3 import so3_geodesic_angle


def _synthetic():
    positions = torch.tensor(
        [
            [0.0, 1.0, 0.0],  # N
            [0.0, 0.0, 0.0],  # CA
            [1.0, 0.0, 0.0],  # C
            [1.0, 1.0, 0.0],  # CB
        ],
        dtype=torch.float32,
    )
    names = ["N", "CA", "C", "CB"]
    atom_to_residue = torch.zeros(4, dtype=torch.long)
    frame = build_residue_frames(positions, atom_to_residue, names, num_residues=1)
    return positions, names, atom_to_residue, frame


def test_shared_se3_matches_historical_bridge_update():
    positions, _, atom_to_residue, frame = _synthetic()
    translation = torch.tensor([[0.4, -0.2, 0.1]])
    rotvec = torch.tensor([[0.1, 0.2, -0.15]])
    expected = apply_fractional_update(
        positions,
        atom_to_residue,
        frame.origins,
        frame.frames,
        translation,
        rotvec,
        remaining_steps=4,
        frame_valid=frame.valid,
    )
    actual = apply_local_se3_update(
        positions,
        atom_to_residue,
        frame.origins,
        frame.frames,
        translation,
        rotvec,
        fraction=0.25,
        frame_valid=frame.valid,
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_shared_se3_zero_update_is_bitwise_stationary():
    positions, _, atom_to_residue, frame = _synthetic()
    zero = torch.zeros(1, 3)
    actual = apply_local_se3_update(
        positions,
        atom_to_residue,
        frame.origins,
        frame.frames,
        zero,
        zero,
        frame_valid=frame.valid,
    )
    assert torch.equal(actual, positions)


@pytest.mark.parametrize("time_value", [0.25, 0.5, 0.75, 1.0])
def test_diffusion_target_reconstructs_holo_on_real_sample(time_value):
    data_root = Path("Apo2Mol-main/Apo2MOl-dataset/data_folder")
    split_path = data_root.parent / "split_druglike_dict.pkl"
    if not data_root.is_dir() or not split_path.is_file():
        pytest.skip("Apo2Mol raw dataset unavailable")
    with split_path.open("rb") as handle:
        record = pickle.load(handle)["train"][0]
    value = Apo2MolAdapter(data_root).convert_record(record)
    state = sample_diffusion_state(
        value,
        time_value,
        generator=torch.Generator().manual_seed(500 + int(time_value * 100)),
    )
    current_frames = build_residue_frames(
        state.model_input.protein_pos,
        state.model_input.atom_to_residue,
        state.model_input.protein_atom_name,
        num_residues=value.num_residues,
    )
    reconstructed = apply_local_se3_update(
        state.model_input.protein_pos,
        state.model_input.atom_to_residue,
        current_frames.origins,
        current_frames.frames,
        state.target.translation_target_local,
        state.target.rotation_target_local,
        frame_valid=state.model_input.frame_valid & current_frames.valid,
    )
    current_chi = build_current_chi_state(
        state.model_input.protein_pos,
        state.model_input.protein_atom_name,
        state.model_input.atom_to_residue,
        state.model_input.protein_residue_name,
    )
    chi_valid = state.model_input.chi_mask & current_chi.geometry_rotatable_mask
    reconstructed = apply_chi_updates(
        reconstructed,
        current_chi.axis_start,
        current_chi.axis_end,
        current_chi.downstream_atom_mask,
        state.target.chi_target,
        valid=chi_valid,
    ).positions
    reconstructed_frames = build_residue_frames(
        reconstructed,
        state.model_input.atom_to_residue,
        state.model_input.protein_atom_name,
        num_residues=value.num_residues,
    )
    holo_frames = build_residue_frames(
        value.protein_pos_holo,
        value.atom_to_residue,
        value.protein_atom_name,
        num_residues=value.num_residues,
    )
    valid_residues = (
        state.model_input.frame_valid
        & reconstructed_frames.valid
        & holo_frames.valid
    )
    origin_error = (
        reconstructed_frames.origins - holo_frames.origins
    ).norm(dim=-1)[valid_residues]
    rotation_error = so3_geodesic_angle(
        reconstructed_frames.frames[valid_residues],
        holo_frames.frames[valid_residues],
    )
    assert float(origin_error.max()) < 2.0e-4
    assert float(rotation_error.max()) < 2.0e-4
    holo_chi = build_current_chi_state(
        value.protein_pos_holo,
        value.protein_atom_name,
        value.atom_to_residue,
        state.model_input.protein_residue_name,
    )
    chi_valid = (
        chi_valid
        & holo_chi.geometry_rotatable_mask
        & valid_residues[:, None]
    )
    chi_error = torch.atan2(
        torch.sin(
            build_current_chi_state(
                reconstructed,
                state.model_input.protein_atom_name,
                state.model_input.atom_to_residue,
                state.model_input.protein_residue_name,
            ).angles
            - holo_chi.angles
        ),
        torch.cos(
            build_current_chi_state(
                reconstructed,
                state.model_input.protein_atom_name,
                state.model_input.atom_to_residue,
                state.model_input.protein_residue_name,
            ).angles
            - holo_chi.angles
        ),
    )
    assert float(chi_error[chi_valid].abs().max()) < 2.0e-4
