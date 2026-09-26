import torch

from pocketdiff.targetdiff import (
    TargetDiffAdapter,
    TargetDiffLigandConditionProvider,
    initialize_targetdiff_state,
)


class _ScheduleModel(torch.nn.Module):
    num_classes = 13
    num_timesteps = 5

    def __init__(self):
        super().__init__()
        self.register_buffer("alphas_cumprod", torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]))

    def q_v_pred(self, log_v0, t, batch):
        a = self.alphas_cumprod[t][batch, None]
        return (a * log_v0.exp() + (1.0 - a) / 13.0).clamp_min(1e-30).log()

    def forward(self, **kwargs):
        raise AssertionError("condition provider must not call the score model")


def _complex():
    from pocketdiff.tests.test_diffusion_pipeline import _synthetic_complex

    return _synthetic_complex()


def test_provider_maps_pocket_time_and_preserves_reference():
    adapter = TargetDiffAdapter(_ScheduleModel())
    provider = TargetDiffLigandConditionProvider(adapter)
    value = _complex()
    before_pos = value.ligand_pos_ref.clone()
    before_type = value.ligand_type_ref.clone()
    condition = provider.condition(
        value,
        0.5,
        generator=torch.Generator().manual_seed(9),
    )
    assert condition.targetdiff_t == 2
    assert condition.ligand_pos.shape == value.ligand_pos_ref.shape
    assert condition.ligand_type.shape == value.ligand_type_ref.shape
    assert torch.equal(value.ligand_pos_ref, before_pos)
    assert torch.equal(value.ligand_type_ref, before_type)


def test_provider_replays_independent_forward_noise_from_same_clean_reference():
    adapter = TargetDiffAdapter(_ScheduleModel())
    provider = TargetDiffLigandConditionProvider(adapter)
    value = _complex()
    first = provider.condition(value, 0.75, generator=torch.Generator().manual_seed(11))
    second = provider.condition(value, 0.75, generator=torch.Generator().manual_seed(11))
    assert torch.equal(first.ligand_pos, second.ligand_pos)
    assert torch.equal(first.ligand_type, second.ligand_type)
    assert first.ligand_pos.data_ptr() != value.ligand_pos_ref.data_ptr()


def test_provider_uses_clean_reference_each_call_not_previous_noisy_state():
    adapter = TargetDiffAdapter(_ScheduleModel())
    provider = TargetDiffLigandConditionProvider(adapter)
    value = _complex()
    low = provider.condition(value, 0.0, generator=torch.Generator().manual_seed(3))
    high = provider.condition(value, 1.0, generator=torch.Generator().manual_seed(3))
    assert low.targetdiff_t == 0
    assert high.targetdiff_t == 4
    assert not torch.equal(low.ligand_pos, high.ligand_pos)


def test_provider_can_recondition_a_label_free_diffusion_input_from_clean_refs():
    adapter = TargetDiffAdapter(_ScheduleModel())
    provider = TargetDiffLigandConditionProvider(adapter)
    value = _complex()
    from pocketdiff.diffusion import sample_diffusion_state

    state = sample_diffusion_state(value, 0.5)
    condition = provider.condition_from_input(
        state.model_input,
        0.25,
        generator=torch.Generator().manual_seed(13),
    )
    assert condition.targetdiff_t == 1
    assert condition.ligand_pos.shape == state.model_input.ligand_pos.shape
    assert condition.ligand_type.shape == state.model_input.ligand_type.shape


def test_condition_adapter_normalizes_cuda_alias_when_available():
    if not torch.cuda.is_available():
        return
    adapter = TargetDiffAdapter(_ScheduleModel().cuda(), device="cuda")
    assert adapter.device == torch.device("cuda", torch.cuda.current_device())
