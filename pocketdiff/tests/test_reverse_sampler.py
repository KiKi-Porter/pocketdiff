import pickle
from pathlib import Path
import pytest, torch
from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.diffusion import (
    DiffusionMotionAdapter,
    DiffusionMotionPrediction,
    sample_diffusion_state,
    sample_reverse_trajectory,
)
from pocketdiff.geometry.bridge import remaining_transform_current_to_holo
from pocketdiff.geometry.chi import periodic_chi_delta
from pocketdiff.geometry.current_state import build_current_chi_state
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.geometry.so3 import so3_geodesic_angle
ROOT=Path('Apo2Mol-main/Apo2MOl-dataset/data_folder'); SPLIT=ROOT.parent/'split_druglike_dict.pkl'

def _state():
 if not ROOT.is_dir() or not SPLIT.is_file(): pytest.skip('dataset unavailable')
 with SPLIT.open('rb') as f:r=pickle.load(f)['train'][0]
 return sample_diffusion_state(Apo2MolAdapter(ROOT).convert_record(r),1.0,generator=torch.Generator().manual_seed(47))

def test_zero_output_sampler_is_stationary():
 state=_state(); model=DiffusionMotionAdapter(dropout=0.0); traj=sample_reverse_trajectory(model,state,steps=4)
 assert len(traj.states)==5 and len(traj.times)==5
 assert all(torch.equal(x,traj.states[0]) for x in traj.states)
 assert traj.times == (1.0,0.75,0.5,0.25,0.0)

def test_sampler_conditions_on_current_state_time():
 state=_state()
 calls=[]
 class RecordingZeroModel:
  def __call__(self, model_input):
   calls.append(float(model_input.diffusion_time.reshape(-1)[0]))
   return DiffusionMotionPrediction(
    translation_local=torch.zeros_like(model_input.chi_current[:, :3]),
    rotation_local=torch.zeros_like(model_input.chi_current[:, :3]),
    chi=torch.zeros_like(model_input.chi_current),
   )
 sample_reverse_trajectory(RecordingZeroModel(),state,steps=4)
 assert calls == [1.0,0.75,0.5,0.25]

def test_sampler_supports_configurable_steps_and_finite_states():
 state=_state(); model=DiffusionMotionAdapter(dropout=0.0)
 for steps in (1,4,8):
  traj=sample_reverse_trajectory(model,state,steps=steps)
  assert len(traj.states)==steps+1
  assert all(torch.isfinite(x).all() for x in traj.states)


def test_oracle_remaining_transform_reaches_holo_endpoint():
    if not ROOT.is_dir() or not SPLIT.is_file():
        pytest.skip('Apo2Mol raw dataset unavailable')
    with SPLIT.open('rb') as f:
        record = pickle.load(f)['train'][0]
    value = Apo2MolAdapter(ROOT).convert_record(record)
    state = sample_diffusion_state(
        value, 1.0, generator=torch.Generator().manual_seed(4701)
    )
    holo_frames = build_residue_frames(
        value.protein_pos_holo,
        value.atom_to_residue,
        value.protein_atom_name,
        num_residues=value.num_residues,
    )

    class OracleModel:
        def __call__(self, model_input):
            current_frames = build_residue_frames(
                model_input.protein_pos,
                model_input.atom_to_residue,
                model_input.protein_atom_name,
                num_residues=value.num_residues,
            )
            valid = model_input.frame_valid & current_frames.valid & holo_frames.valid
            target = remaining_transform_current_to_holo(
                current_frames.origins,
                current_frames.frames,
                holo_frames.origins,
                holo_frames.frames,
                frame_valid=valid,
            )
            current_chi = build_current_chi_state(
                model_input.protein_pos,
                model_input.protein_atom_name,
                model_input.atom_to_residue,
                model_input.protein_residue_name,
            )
            chi_mask = model_input.chi_mask & current_chi.geometry_rotatable_mask
            holo_chi = build_current_chi_state(
                value.protein_pos_holo,
                value.protein_atom_name,
                value.atom_to_residue,
                model_input.protein_residue_name,
            )
            return DiffusionMotionPrediction(
                translation_local=target.translation_local,
                rotation_local=target.rotvec_local,
                chi=periodic_chi_delta(
                    current_chi.angles, holo_chi.angles, chi_mask
                ),
            )

    trajectory = sample_reverse_trajectory(OracleModel(), state, steps=4)
    final_frames = build_residue_frames(
        trajectory.states[-1],
        state.model_input.atom_to_residue,
        state.model_input.protein_atom_name,
        num_residues=value.num_residues,
    )
    valid_residues = state.model_input.frame_valid & final_frames.valid & holo_frames.valid
    origin_error = (final_frames.origins - holo_frames.origins).norm(dim=-1)[valid_residues]
    rotation_error = so3_geodesic_angle(
        final_frames.frames[valid_residues], holo_frames.frames[valid_residues]
    )
    assert float(origin_error.max()) < 2.0e-4
    assert float(rotation_error.max()) < 2.0e-4
    final_chi = build_current_chi_state(
        trajectory.states[-1],
        state.model_input.protein_atom_name,
        state.model_input.atom_to_residue,
        state.model_input.protein_residue_name,
    )
    holo_chi = build_current_chi_state(
        value.protein_pos_holo,
        value.protein_atom_name,
        value.atom_to_residue,
        state.model_input.protein_residue_name,
    )
    chi_valid = (
        state.model_input.chi_mask
        & final_chi.geometry_rotatable_mask
        & holo_chi.geometry_rotatable_mask
        & valid_residues[:, None]
    )
    chi_error = torch.atan2(
        torch.sin(final_chi.angles - holo_chi.angles),
        torch.cos(final_chi.angles - holo_chi.angles),
    )
    assert float(chi_error[chi_valid].abs().max()) < 2.0e-4
