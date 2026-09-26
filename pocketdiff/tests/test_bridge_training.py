"""Independent rigid-motion checks for the teacher-forced training contract."""

from dataclasses import fields, replace
import math

import pytest
import torch

from pocketdiff.data.schema import PocketComplex
from pocketdiff.geometry.bridge import apply_fractional_update
from pocketdiff.geometry.frames import build_residue_frames
from pocketdiff.geometry.so3 import so3_exp
from pocketdiff.models import PocketDiffModel
from pocketdiff.training import (
    build_bridge_batch, collate_clean_examples, make_clean_example,
    masked_remaining_motion_loss,
)


def _rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _complex(sample_id, motions):
    apo, holo, names, ids = [], [], [], []
    for r, (angle, shift) in enumerate(motions):
        local = torch.tensor([[0., 1., 0.], [0., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
        count = 4 if r == 0 else 3
        origin = torch.tensor([r * 3., 0., 0.])
        apo.append(local[:count] + origin)
        holo.append(local[:count] @ _rz(angle).T + origin + torch.tensor(shift))
        names.extend(['N', 'CA', 'C', 'CB'][:count])
        ids.extend([r] * count)
    apo, holo = torch.cat(apo), torch.cat(holo)
    nr, na = len(motions), len(names)
    return PocketComplex(
        sample_id=sample_id, protein_pos_apo=apo, protein_pos_holo=holo,
        protein_feature=torch.zeros(na, 27), protein_element=torch.full((na,), 6),
        protein_atom_name=names, protein_residue_name=['ALA'] * na,
        atom_to_residue=torch.tensor(ids), residue_type=torch.zeros(nr, dtype=torch.long),
        residue_chain_id=['A'] * nr, residue_sequence_id=[str(r) for r in range(nr)],
        frame_valid=torch.ones(nr, dtype=torch.bool), chi_apo=torch.zeros(nr, 5),
        chi_holo=torch.zeros(nr, 5), chi_mask=torch.zeros(nr, 5, dtype=torch.bool),
        ligand_pos_ref=torch.tensor([[0.5, 0.5, 1.]]), ligand_type_ref=torch.tensor([1]),
        center_offset=torch.zeros(3),
    )


MOTIONS = [(0.6, [0.4, -0.3, 0.2]), (-0.9, [-0.2, 0.7, 0.1]),
           (1.2, [0.3, 0.2, -0.4])]


@pytest.fixture
def clean():
    values = [_complex('a', MOTIONS[:2]), _complex('b', MOTIONS[2:])]
    return collate_clean_examples([make_clean_example(v) for v in values])


def _oracle_step(batch):
    frames = build_residue_frames(batch.protein_pos, batch.atom_to_residue_global,
                                  batch.protein_atom_name, num_residues=len(batch.residue_type))
    return apply_fractional_update(
        batch.protein_pos, batch.atom_to_residue_global, frames.origins, frames.frames,
        batch.target_translation_local, batch.target_rotvec_local,
        remaining_steps=batch.remaining_steps[batch.batch_residue], frame_valid=batch.frame_valid,
    )


@pytest.mark.parametrize('k', range(20))
def test_all_times_match_analytic_rigid_motion_with_mixed_graph_progress(clean, k):
    times = torch.tensor([k, 19 - k])
    batch = build_bridge_batch(clean, times)
    assert batch.targetdiff_t.tolist() == [199 - 10 * k, 9 + 10 * k]
    assert batch.remaining_steps.tolist() == [20 - k, 1 + k]
    for r, (angle, shift) in enumerate(MOTIONS):
        graph = int(clean.batch_residue[r])
        s = float(times[graph]) / 20
        origin = torch.tensor([3. if r == 1 else 0., 0., 0.])
        mask = clean.atom_to_residue_global == r
        expected = (clean.apo_pos_ref[mask] - origin) @ _rz(s * angle).T + origin + s * torch.tensor(shift)
        expected_next = ((clean.apo_pos_ref[mask] - origin) @ _rz((s + .05) * angle).T
                         + origin + (s + .05) * torch.tensor(shift))
        torch.testing.assert_close(batch.protein_pos[mask], expected, atol=1e-5, rtol=0)
        torch.testing.assert_close(batch.protein_pos_next_target[mask], expected_next, atol=1e-5, rtol=0)
        torch.testing.assert_close(batch.target_translation_local[r],
                                  (1 - s) * torch.tensor(shift) @ _rz(s * angle), atol=1e-5, rtol=0)
        torch.testing.assert_close(batch.target_rotvec_local[r],
                                  torch.tensor([0., 0., (1 - s) * angle]), atol=1e-5, rtol=0)
    assert float((_oracle_step(batch) - batch.protein_pos_next_target).abs().max()) < 1e-4


def test_k_zero_is_clean_compatible_and_supervision_does_not_leak_to_model(clean):
    batch = build_bridge_batch(clean, torch.zeros(2, dtype=torch.long))
    for field in fields(clean):
        expected, actual = getattr(clean, field.name), getattr(batch, field.name)
        if isinstance(expected, torch.Tensor):
            assert torch.equal(actual, expected), field.name
        else:
            assert actual == expected
    assert batch.model_kwargs().keys() == clean.model_kwargs().keys()
    assert not any(name.startswith('target_') or name == 'protein_pos_holo' for name in batch.model_kwargs())


def test_rng_is_replayable_and_inputs_and_explicit_k_are_not_mutated(clean):
    snapshot = {f.name: getattr(clean, f.name).clone() for f in fields(clean)
                if isinstance(getattr(clean, f.name), torch.Tensor)}
    rng = torch.get_rng_state().clone()
    first = build_bridge_batch(clean, generator=torch.Generator().manual_seed(73))
    second = build_bridge_batch(clean, generator=torch.Generator().manual_seed(73))
    expected_k = torch.randint(20, (2,), generator=torch.Generator().manual_seed(73))
    assert torch.equal(first.pocket_k, expected_k)
    assert torch.equal(first.protein_pos, second.protein_pos)
    assert torch.equal(torch.get_rng_state(), rng)
    explicit = build_bridge_batch(clean, expected_k)
    expected_k.fill_(0)
    assert torch.equal(explicit.pocket_k, first.pocket_k)
    for name, value in snapshot.items():
        assert torch.equal(getattr(clean, name), value), name
    for name in ('apo_pos_ref', 'protein_pos_holo', 'ligand_pos', 'ligand_v', 'protein_feature',
                 'batch_protein', 'batch_residue', 'batch_ligand', 'atom_to_residue_global'):
        assert torch.equal(getattr(first, name), snapshot[name]), name


def test_invalid_frame_is_stationary_and_excluded_from_labels(clean):
    names = list(clean.protein_atom_name)
    names[0] = 'O'  # First residue has no N atom, despite supplied mask being True.
    invalid = replace(clean, protein_atom_name=names)
    batch = build_bridge_batch(invalid, torch.tensor([10, 19]))
    mask = batch.atom_to_residue_global == 0
    assert batch.frame_valid.tolist() == [False, True, True]
    assert torch.equal(batch.protein_pos[mask], clean.apo_pos_ref[mask])
    assert torch.equal(batch.protein_pos_next_target[mask], clean.apo_pos_ref[mask])
    assert torch.equal(batch.target_translation_local[0], torch.zeros(3))
    assert torch.equal(batch.target_rotvec_local[0], torch.zeros(3))
    assert torch.equal(_oracle_step(batch)[mask], clean.apo_pos_ref[mask])


def test_next_target_is_rigid_bridge_not_unrepresentable_holo_sidechain(clean):
    holo = clean.protein_pos_holo.clone()
    holo[3, 2] += 2.0  # Move CB internally, leaving N/CA/C endpoint frame unchanged.
    changed = replace(clean, protein_pos_holo=holo)
    batch = build_bridge_batch(changed, torch.tensor([19, 19]))
    baseline = build_bridge_batch(clean, torch.tensor([19, 19]))
    assert torch.equal(batch.protein_pos_next_target, baseline.protein_pos_next_target)
    assert float((batch.protein_pos_next_target[3] - holo[3]).norm()) > 1.99
    assert float((_oracle_step(batch) - batch.protein_pos_next_target).abs().max()) < 1e-4


def test_bridge_labels_and_coordinates_obey_global_se3(clean):
    rotation = so3_exp(torch.tensor([0.3, -0.2, 0.1]))
    shift = torch.tensor([4., -3., 2.])
    moved = replace(clean, **{name: getattr(clean, name) @ rotation.T + shift
                             for name in ('protein_pos', 'apo_pos_ref', 'protein_pos_holo', 'ligand_pos')})
    original = build_bridge_batch(clean, torch.tensor([7, 16]))
    transformed = build_bridge_batch(moved, torch.tensor([7, 16]))
    for name in ('target_translation_local', 'target_rotvec_local'):
        torch.testing.assert_close(getattr(original, name), getattr(transformed, name), atol=1e-4, rtol=0)
    for name in ('protein_pos', 'protein_pos_next_target'):
        torch.testing.assert_close(getattr(transformed, name), getattr(original, name) @ rotation.T + shift,
                                  atol=1e-4, rtol=0)


def test_labels_detached_but_model_remaining_loss_has_finite_gradients(clean):
    clean.apo_pos_ref.requires_grad_(True)
    clean.protein_pos_holo.requires_grad_(True)
    batch = build_bridge_batch(clean, torch.tensor([5, 19]))
    for name in ('protein_pos', 'protein_pos_next_target', 'target_translation_local', 'target_rotvec_local'):
        assert not getattr(batch, name).requires_grad, name
    model = PocketDiffModel(encoder_layers=1, knn=4)
    prediction = model(**batch.model_kwargs())
    loss = masked_remaining_motion_loss(prediction, batch.target_translation_local,
                                        batch.target_rotvec_local, batch.frame_valid).loss
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    head_grad = model.motion_head.network[-1].weight.grad
    assert float(head_grad[:3].abs().sum()) > 0
    assert float(head_grad[3:].abs().sum()) > 0
    assert clean.protein_pos_holo.grad is None


@pytest.mark.parametrize('k', [torch.tensor([-1, 0]), torch.tensor([0, 20]), torch.tensor([1]),
                              torch.tensor([1., 2.]), torch.tensor([[1, 2]]), [1, 2]])
def test_invalid_graph_time_rejected(clean, k):
    with pytest.raises(ValueError, match='pocket_k'):
        build_bridge_batch(clean, k)


def test_rejects_ambiguous_randomness_and_nonendpoint_inputs(clean):
    with pytest.raises(ValueError, match='either'):
        build_bridge_batch(clean, torch.tensor([0, 1]), generator=torch.Generator())
    with pytest.raises(TypeError, match='endpoint CleanBatch'):
        build_bridge_batch(build_bridge_batch(clean, torch.tensor([0, 1])))
    with pytest.raises(ValueError, match='apo endpoints'):
        build_bridge_batch(replace(clean, protein_pos=clean.protein_pos + 0.1))
