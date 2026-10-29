from __future__ import annotations

import copy

import torch
import torch.nn as nn

from models.aplc import APLCBlock
from models.molopt_score_model import ScorePosNet3D
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral
from utils import misc


def _toy_context(hidden_dim: int = 16):
    torch.manual_seed(7)
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.1, 0.0],
            [0.1, 1.5, 0.2],
            [0.2, 0.2, 1.6],
            [2.2, 0.5, 0.1],
            [2.7, 1.1, 0.3],
            [10.0, 0.0, 0.0],
            [11.2, 0.2, 0.0],
            [10.1, 1.4, 0.1],
            [10.2, 0.4, 1.5],
            [12.0, 0.5, 0.1],
            [12.5, 1.0, 0.4],
        ],
        dtype=torch.float32,
    )
    hidden = torch.randn(positions.size(0), hidden_dim)
    mask_ligand = torch.tensor(
        [False, False, False, False, True, True] * 2,
        dtype=torch.bool,
    )
    batch = torch.tensor([0] * 6 + [1] * 6, dtype=torch.long)
    return hidden, positions, mask_ligand, batch


def _small_transformer(num_layers: int = 2) -> UniTransformerO2TwoUpdateGeneral:
    return UniTransformerO2TwoUpdateGeneral(
        num_blocks=1,
        num_layers=num_layers,
        hidden_dim=16,
        n_heads=4,
        k=4,
        num_r_gaussian=20,
        edge_feat_dim=4,
        num_node_types=8,
        act_fn="relu",
        norm=True,
        cutoff_mode="knn",
        ew_net_type="r",
        num_x2h=1,
        num_h2x=1,
        r_max=10.0,
        x2h_out_fc=False,
        sync_twoup=False,
    )


def test_aplc_zero_gate_is_identity_and_finite():
    hidden, positions, mask_ligand, batch = _toy_context()
    block = APLCBlock(hidden_dim=hidden.size(-1), k=3, num_r_gaussian=8)

    output = block(hidden, positions, mask_ligand, batch)

    assert torch.isfinite(output).all()
    assert torch.allclose(output, hidden, atol=0.0, rtol=0.0)

    loss = output.square().mean()
    loss.backward()
    assert block.residual_gate.grad is not None
    assert torch.isfinite(block.residual_gate.grad)


def test_aplc_has_gradients_after_gate_is_opened():
    hidden, positions, mask_ligand, batch = _toy_context()
    block = APLCBlock(hidden_dim=hidden.size(-1), k=3, num_r_gaussian=8)
    with torch.no_grad():
        block.residual_gate.fill_(0.25)

    output = block(hidden, positions, mask_ligand, batch)
    output.square().mean().backward()

    trainable_grads = [
        parameter.grad
        for name, parameter in block.named_parameters()
        if name != "residual_gate" and parameter.requires_grad
    ]
    assert any(
        gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        for gradient in trainable_grads
    )


def test_aplc_edges_never_cross_graphs():
    hidden, positions, mask_ligand, batch = _toy_context()
    block = APLCBlock(hidden_dim=hidden.size(-1), k=3, num_r_gaussian=8)
    for edge_index in block._build_edges(positions, mask_ligand, batch):
        if edge_index.numel() == 0:
            continue
        source, destination = edge_index
        assert torch.equal(batch[source], batch[destination])


def test_aplc_scalar_state_is_invariant_to_global_rigid_transform():
    hidden, positions, mask_ligand, batch = _toy_context()
    block = APLCBlock(hidden_dim=hidden.size(-1), k=3, num_r_gaussian=8)
    with torch.no_grad():
        block.residual_gate.fill_(0.2)

    angle = torch.tensor(0.73)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = torch.tensor([3.0, -2.0, 1.5])

    original = block(hidden, positions, mask_ligand, batch)
    transformed = block(
        hidden,
        positions @ rotation + translation,
        mask_ligand,
        batch,
    )

    assert torch.allclose(original, transformed, atol=2e-5, rtol=2e-5)


def test_aplc_time_conditioning_changes_opened_residual():
    hidden, positions, mask_ligand, batch = _toy_context()
    block = APLCBlock(
        hidden_dim=hidden.size(-1),
        k=3,
        num_r_gaussian=8,
        num_layers=3,
        layerwise_layers=3,
    )
    with torch.no_grad():
        block.residual_gate.fill_(0.2)
        block.time_mlp[0].weight.fill_(0.1)
        block.time_mlp[0].bias.zero_()
        block.time_mlp[2].weight.fill_(0.1)
        block.time_mlp[2].bias.zero_()
        block.time_scale.weight.fill_(0.1)
        block.time_scale.bias.zero_()

    early = block(
        hidden,
        positions,
        mask_ligand,
        batch,
        time_step=torch.zeros(2, dtype=torch.long),
        num_timesteps=1000,
        layer_index=2,
    )
    late = block(
        hidden,
        positions,
        mask_ligand,
        batch,
        time_step=torch.full((2,), 999, dtype=torch.long),
        num_timesteps=1000,
        layer_index=2,
    )

    assert torch.isfinite(early).all()
    assert torch.isfinite(late).all()
    assert not torch.allclose(early, late, atol=1e-7, rtol=1e-7)


def test_aplc_layerwise_gate_can_open_nonfinal_layer():
    hidden, positions, mask_ligand, batch = _toy_context()
    block = APLCBlock(
        hidden_dim=hidden.size(-1),
        k=3,
        num_r_gaussian=8,
        num_layers=3,
        layerwise_layers=3,
    )
    with torch.no_grad():
        block.residual_gate.fill_(0.2)

    closed = block(hidden, positions, mask_ligand, batch, layer_index=0)
    with torch.no_grad():
        block.layer_gates[0] = 0.5
    opened = block(hidden, positions, mask_ligand, batch, layer_index=0)

    assert torch.allclose(closed, hidden, atol=1e-7, rtol=1e-7)
    assert not torch.allclose(opened, hidden, atol=1e-7, rtol=1e-7)


def test_transformer_calls_aplc_only_in_configured_tail_layers():
    class TrackingAPLC(nn.Module):
        def __init__(self, layerwise_layers):
            super().__init__()
            self.layerwise_layers = layerwise_layers
            self.called_layers = []

        def forward(
            self,
            hidden,
            positions,
            mask_ligand,
            batch,
            time_step=None,
            num_timesteps=None,
            layer_index=0,
        ):
            self.called_layers.append(layer_index)
            return hidden

    transformer = _small_transformer(num_layers=4)
    tracker = TrackingAPLC(layerwise_layers=2)
    transformer.aplc = tracker
    hidden, positions, mask_ligand, batch = _toy_context(hidden_dim=16)

    output = transformer(
        hidden,
        positions,
        mask_ligand,
        batch,
        time_step=torch.tensor([10, 20]),
        num_timesteps=1000,
    )

    assert tracker.called_layers == [2, 3]
    assert torch.isfinite(output["x"]).all()
    assert torch.isfinite(output["h"]).all()


def test_zero_gate_preserves_transformer_output():
    torch.manual_seed(11)
    baseline = _small_transformer()
    with_aplc = _small_transformer()
    with_aplc.load_state_dict(baseline.state_dict(), strict=True)
    with_aplc.enable_aplc(k=3, num_r_gaussian=8)

    hidden, positions, mask_ligand, batch = _toy_context(hidden_dim=16)
    baseline_output = baseline(hidden, positions, mask_ligand, batch)
    aplc_output = with_aplc(hidden, positions, mask_ligand, batch)

    assert torch.allclose(baseline_output["x"], aplc_output["x"], atol=1e-7, rtol=1e-7)
    assert torch.allclose(baseline_output["h"], aplc_output["h"], atol=1e-7, rtol=1e-7)


def test_opened_aplc_changes_ligand_coordinate_branch():
    torch.manual_seed(13)
    baseline = _small_transformer()
    with_aplc = _small_transformer()
    with_aplc.load_state_dict(baseline.state_dict(), strict=True)
    with_aplc.enable_aplc(k=3, num_r_gaussian=8)
    with torch.no_grad():
        with_aplc.aplc.residual_gate.fill_(0.25)

    hidden, positions, mask_ligand, batch = _toy_context(hidden_dim=16)
    baseline_output = baseline(hidden, positions, mask_ligand, batch)
    aplc_output = with_aplc(hidden, positions, mask_ligand, batch)

    assert torch.isfinite(aplc_output["x"]).all()
    assert torch.isfinite(aplc_output["h"]).all()
    assert not torch.allclose(
        baseline_output["x"][mask_ligand],
        aplc_output["x"][mask_ligand],
        atol=1e-7,
        rtol=1e-7,
    )


def test_original_checkpoint_state_is_compatible_with_aplc_model():
    config = misc.load_config("configs/training.yml")
    original = ScorePosNet3D(config.model, 27, 13)

    aplc_config = copy.deepcopy(config.model)
    aplc_config.aplc_enabled = True
    aplc_config.aplc_k = 3
    aplc_config.aplc_num_r_gaussian = 8
    adapted = ScorePosNet3D(aplc_config, 27, 13)

    missing, unexpected = adapted.load_state_dict(
        original.state_dict(),
        strict=False,
    )

    assert not unexpected
    assert missing
    assert all(key.startswith("refine_net.aplc.") for key in missing)
