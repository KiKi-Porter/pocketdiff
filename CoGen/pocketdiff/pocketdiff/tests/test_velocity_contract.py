import torch

from pocketdiff.data.schema import PocketComplex
from pocketdiff.diffusion import (
    DiffusionMotionPrediction,
    sample_diffusion_state,
    sample_reverse_trajectory,
)
from pocketdiff.geometry.bridge import remaining_transform_current_to_holo
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.training.diffusion import _build_motion_buckets


def _complex(angle: float = 0.35, shift=(0.4, -0.2, 0.3)) -> PocketComplex:
    apo = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    rotation = torch.stack(
        (
            torch.stack((c, -s, torch.tensor(0.0))),
            torch.stack((s, c, torch.tensor(0.0))),
            torch.tensor((0.0, 0.0, 1.0)),
        )
    )
    holo = apo @ rotation.T + torch.tensor(shift, dtype=torch.float32)
    return PocketComplex(
        sample_id=f"synthetic-{angle}",
        protein_pos_apo=apo,
        protein_pos_holo=holo,
        protein_feature=torch.zeros(4, 27),
        protein_element=torch.tensor([7, 6, 6, 8]),
        protein_atom_name=["N", "CA", "C", "O"],
        protein_residue_name=["ALA"] * 4,
        atom_to_residue=torch.zeros(4, dtype=torch.long),
        residue_type=torch.tensor([0]),
        residue_chain_id=["A"],
        residue_sequence_id=["1"],
        frame_valid=torch.ones(1, dtype=torch.bool),
        chi_apo=torch.zeros(1, 5),
        chi_holo=torch.zeros(1, 5),
        chi_mask=torch.zeros(1, 5, dtype=torch.bool),
        ligand_pos_ref=torch.tensor([[0.5, 0.5, 2.0]]),
        ligand_type_ref=torch.tensor([1]),
        center_offset=torch.zeros(3),
    )


class _OracleVelocity(torch.nn.Module):
    def __init__(self, holo: torch.Tensor):
        super().__init__()
        self.holo = holo

    def forward(self, model_input):
        current = build_residue_frames(
            model_input.protein_pos,
            model_input.atom_to_residue,
            model_input.protein_atom_name,
            num_residues=1,
        )
        holo = build_residue_frames(
            self.holo,
            model_input.atom_to_residue,
            model_input.protein_atom_name,
            num_residues=1,
        )
        remaining = remaining_transform_current_to_holo(
            current.origins,
            current.frames,
            holo.origins,
            holo.frames,
            frame_valid=current.valid & holo.valid,
        )
        time = model_input.diffusion_time.reshape(-1)[0].clamp_min(0.02)
        return DiffusionMotionPrediction(
            remaining.translation_local / time,
            remaining.rotvec_local / time,
            torch.zeros((1, 5), dtype=model_input.protein_pos.dtype),
        )


def test_oracle_velocity_reaches_holo_in_twenty_steps():
    value = _complex()
    state = sample_diffusion_state(
        value,
        1.0,
        translation_noise_scale=0.0,
        rotation_noise_scale=0.0,
        chi_noise_scale=0.0,
    )
    trajectory = sample_reverse_trajectory(
        _OracleVelocity(value.protein_pos_holo),
        state,
        steps=20,
        prediction_type="velocity",
    )
    assert len(trajectory.states) == 21
    assert trajectory.times[0] == 1.0
    assert trajectory.times[-1] == 0.0
    torch.testing.assert_close(
        trajectory.states[-1],
        value.protein_pos_holo,
        atol=3e-5,
        rtol=0.0,
    )


def test_motion_buckets_are_balanced_by_sorted_motion_quantiles():
    values = [
        _complex(angle=0.05 * index, shift=(0.1 * index, 0.0, 0.0))
        for index in range(8)
    ]
    buckets, scores = _build_motion_buckets(values, 4)
    assert len(buckets) == 4
    assert [len(bucket) for bucket in buckets] == [2, 2, 2, 2]
    assert len(scores) == len(values)
    assert all(
        scores[buckets[index][-1]] <= scores[buckets[index + 1][0]]
        for index in range(len(buckets) - 1)
    )
