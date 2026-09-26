import torch

from pocketdiff.diffusion import (
    DiffusionMotionAdapter,
    build_diffusion_state_from_current,
    collate_diffusion_states,
    sample_diffusion_state,
)


def test_anchor_and_batch_forward_match_independent_graph_calls():
    from pocketdiff.tests.test_diffusion_pipeline import _synthetic_complex

    first = _synthetic_complex()
    second = _synthetic_complex()
    second.sample_id = "synthetic-2"
    second.protein_pos_apo = second.protein_pos_apo + torch.tensor([4.0, 0.0, 0.0])
    second.protein_pos_holo = second.protein_pos_holo + torch.tensor([4.0, 0.0, 0.0])
    second.ligand_pos_ref = second.ligand_pos_ref + torch.tensor([4.0, 0.0, 0.0])
    a = sample_diffusion_state(first, 0.5, translation_noise_scale=0.0,
                               rotation_noise_scale=0.0, chi_noise_scale=0.0)
    b = sample_diffusion_state(second, 0.5, translation_noise_scale=0.0,
                               rotation_noise_scale=0.0, chi_noise_scale=0.0)
    inputs, _ = collate_diffusion_states([a, b])
    model = DiffusionMotionAdapter(dropout=0.0).eval()
    with torch.no_grad():
        batched = model(inputs)
        first_out = model(a.model_input)
        second_out = model(b.model_input)
    torch.testing.assert_close(batched.translation_local[:1], first_out.translation_local)
    torch.testing.assert_close(batched.translation_local[1:], second_out.translation_local)
    assert torch.isfinite(batched.rotation_local).all()


def test_anchor_construction_does_not_require_holo_in_model_input():
    from pocketdiff.tests.test_diffusion_pipeline import _synthetic_complex

    value = _synthetic_complex()
    current = value.protein_pos_apo.detach().clone()
    state = build_diffusion_state_from_current(value, current, 1.0)
    assert not hasattr(state.model_input, "protein_pos_holo")
    assert torch.equal(state.model_input.apo_pos_ref, value.protein_pos_apo)
