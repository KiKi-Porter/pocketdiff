from dataclasses import replace

import pytest
import torch

from pocketdiff.data.schema import PocketDiffPrediction
from pocketdiff.evaluation.clean_rollout import score_clean_rollout
from pocketdiff.models import PocketDiffModel
from pocketdiff.sampling.clean_rollout import INFERENCE_FIELDS, run_clean_rollout
from pocketdiff.tests.test_training import _batch


def inference_inputs(batch):
    return {name: getattr(batch, name) for name in INFERENCE_FIELDS}


class RecurrenceModel(torch.nn.Module):
    """Analytic x_next = x + .01*(1+x) in the fixture's unrotated frames."""
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, **kwargs):
        self.seen.append({n: v.clone() if isinstance(v, torch.Tensor) else list(v)
                          for n, v in kwargs.items()})
        nr = kwargs['residue_type'].numel()
        ca = kwargs['protein_pos'][[i for i, n in enumerate(kwargs['protein_atom_name']) if n == 'CA']]
        tr = torch.zeros(nr, 3)
        tr[:, 0] = (20-kwargs['pocket_k'][kwargs['batch_residue']])*.01*(1+ca[:, 0])
        return PocketDiffPrediction(tr, torch.zeros_like(tr), None, kwargs['frame_valid'], {})


def test_all_steps_feed_previous_prediction_and_follow_schedule():
    inputs = inference_inputs(_batch())
    model = RecurrenceModel()
    trace = run_clean_rollout(model, inputs)
    assert trace.positions.shape == (21, inputs['apo_pos_ref'].shape[0], 3)
    assert len(model.seen) == 20
    for k, kwargs in enumerate(model.seen):
        assert torch.equal(kwargs['protein_pos'], trace.positions[k])
        assert kwargs['pocket_k'].tolist() == [k, k]
        assert kwargs['targetdiff_t'].tolist() == [199-10*k]*2
        assert 'protein_pos_holo' not in kwargs
        for name in INFERENCE_FIELDS:
            if isinstance(inputs[name], torch.Tensor):
                assert torch.equal(kwargs[name], inputs[name]), name
    # Both fixture residues start with CA x=0; cumulative displacement follows
    # this independently derived recurrence, not 20 resets to the apo input.
    expected = 1.01**20-1
    assert torch.allclose(trace.positions[-1, :, 0]-trace.positions[0, :, 0],
                          torch.full_like(trace.positions[0, :, 0], expected), atol=3e-6)


@pytest.mark.parametrize('mode', ['remaining', 'bridge_rate'])
def test_zero_update_is_exact_preserves_rng_mode_and_caller_inputs(mode):
    inputs = inference_inputs(_batch())
    before = {n: v.clone() if isinstance(v, torch.Tensor) else list(v) for n, v in inputs.items()}
    model = PocketDiffModel(encoder_layers=1, knn=4, motion_parameterization=mode).train()
    rng = torch.get_rng_state().clone()
    trace = run_clean_rollout(model, inputs)
    assert model.training
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(trace.positions, inputs['apo_pos_ref'].expand(21, -1, -1))
    assert not trace.positions.requires_grad
    for n, v in inputs.items():
        assert torch.equal(v, before[n]) if isinstance(v, torch.Tensor) else v == before[n]


def test_holo_and_label_changes_only_change_scores_not_trajectory():
    batch = _batch()
    modified = replace(batch, protein_pos_holo=batch.protein_pos_holo+10,
                       target_translation_local=batch.target_translation_local*100,
                       frame_valid=~batch.frame_valid)
    model = RecurrenceModel()
    first = run_clean_rollout(model, inference_inputs(batch))
    second = run_clean_rollout(model, inference_inputs(modified))
    assert torch.equal(first.positions, second.positions)
    score1 = score_clean_rollout(first, inference_inputs(batch), batch.protein_pos_holo, batch.sample_ids)
    score2 = score_clean_rollout(second, inference_inputs(modified), modified.protein_pos_holo, batch.sample_ids)
    assert score1['per_step'][-1]['holo_rmsd'] != score2['per_step'][-1]['holo_rmsd']
    for row in score1['per_graph']:
        graph = batch.sample_ids.index(row['sample_id'])
        mask = batch.batch_protein == graph
        expected = (first.positions[row['step'], mask]-batch.protein_pos_holo[mask]).square().sum(-1).mean().sqrt()
        assert row['holo_rmsd'] == float(expected)


def test_score_mask_stays_fixed_when_a_later_frame_is_invalid():
    batch = _batch()
    inputs = inference_inputs(batch)
    trace = run_clean_rollout(RecurrenceModel(), inputs)
    invalid = trace.frame_valid.clone()
    invalid[-1, 0] = False
    modified = replace(trace, frame_valid=invalid)
    reference = score_clean_rollout(trace, inputs, batch.protein_pos_holo, batch.sample_ids)
    actual = score_clean_rollout(modified, inputs, batch.protein_pos_holo, batch.sample_ids)
    assert actual['per_step'][-1]['lost_apo_frame_count'] == 1
    assert actual['per_step'][-1]['holo_rmsd'] == reference['per_step'][-1]['holo_rmsd']


def test_inplace_mutation_raises_and_restores_mode_without_mutating_caller():
    class BadModel(RecurrenceModel):
        def forward(self, **kwargs):
            kwargs['ligand_pos'].add_(1)
            return super().forward(**kwargs)
    inputs = inference_inputs(_batch())
    before = inputs['ligand_pos'].clone()
    model = BadModel().train()
    with pytest.raises(RuntimeError, match='mutated fixed input'):
        run_clean_rollout(model, inputs)
    assert model.training
    assert torch.equal(before, inputs['ligand_pos'])


@pytest.mark.parametrize('extra', ['protein_pos_holo', 'protein_pos', 'frame_valid'])
def test_rejects_targets_current_state_and_externally_supplied_mask(extra):
    inputs = inference_inputs(_batch())
    inputs[extra] = torch.zeros(1)
    with pytest.raises(ValueError, match='exactly INFERENCE_FIELDS'):
        run_clean_rollout(RecurrenceModel(), inputs)


def test_nonfinite_prediction_fails_visibly_and_restores_mode():
    class BadModel(RecurrenceModel):
        def forward(self, **kwargs):
            pred = super().forward(**kwargs)
            pred.remaining_translation_local[0, 0] = float('nan')
            return pred
    model = BadModel().train()
    with pytest.raises(FloatingPointError, match='non-finite prediction'):
        run_clean_rollout(model, inference_inputs(_batch()))
    assert model.training
