"""Minimal, read-only single-step adapter around official TargetDiff."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .state import TargetDiffState, initialize_targetdiff_state


@dataclass(frozen=True)
class TargetDiffStepRandomness:
    """Explicit random tensors consumed by one reverse diffusion step."""

    position_noise: torch.Tensor
    categorical_uniform: torch.Tensor

    def __post_init__(self) -> None:
        if self.position_noise.ndim != 2 or self.position_noise.shape[-1] != 3:
            raise ValueError("position_noise must have shape [Nl, 3]")
        if self.categorical_uniform.ndim != 2 or self.categorical_uniform.shape[-1] != 13:
            raise ValueError("categorical_uniform must have shape [Nl, 13]")
        if self.categorical_uniform.shape[0] != self.position_noise.shape[0]:
            raise ValueError("position and categorical randomness have different atom counts")
        if not torch.isfinite(self.position_noise).all() or not torch.isfinite(self.categorical_uniform).all():
            raise ValueError("explicit TargetDiff randomness must be finite")
        if bool((self.categorical_uniform < 0).any()) or bool((self.categorical_uniform >= 1).any()):
            raise ValueError("categorical_uniform must lie in [0, 1)")


@dataclass(frozen=True)
class TargetDiffRNGTrace:
    """Per-timestep randomness for an auditable, replayable reverse run."""

    steps: Mapping[int, TargetDiffStepRandomness]

    def __post_init__(self) -> None:
        if not isinstance(self.steps, Mapping) or not self.steps:
            raise ValueError("TargetDiffRNGTrace must contain at least one timestep")
        for timestep, randomness in self.steps.items():
            if not isinstance(timestep, int) or isinstance(timestep, bool) or timestep < 0:
                raise ValueError("RNG trace timesteps must be non-negative integers")
            if not isinstance(randomness, TargetDiffStepRandomness):
                raise TypeError("RNG trace values must be TargetDiffStepRandomness")

    def for_step(self, timestep: int) -> TargetDiffStepRandomness:
        try:
            return self.steps[timestep]
        except KeyError as exc:
            raise KeyError(f"RNG trace has no randomness for timestep {timestep}") from exc


@dataclass(frozen=True)
class TargetDiffStepAux:
    """Diagnostics needed by the later soft-condition bridge."""

    pred_x0: torch.Tensor
    pred_v0_prob: torch.Tensor
    posterior_v_prev_prob: torch.Tensor

    def __post_init__(self) -> None:
        if self.pred_x0.ndim != 2 or self.pred_x0.shape[-1] != 3:
            raise ValueError("pred_x0 must have shape [Nl, 3]")
        if self.pred_v0_prob.ndim != 2 or self.pred_v0_prob.shape[-1] != 13:
            raise ValueError("pred_v0_prob must have shape [Nl, 13]")
        if self.posterior_v_prev_prob.shape != self.pred_v0_prob.shape:
            raise ValueError("posterior_v_prev_prob and pred_v0_prob shapes differ")
        for name, value in (
            ("pred_x0", self.pred_x0),
            ("pred_v0_prob", self.pred_v0_prob),
            ("posterior_v_prev_prob", self.posterior_v_prev_prob),
        ):
            if value.numel() and not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        if self.pred_v0_prob.numel() and not torch.allclose(
            self.pred_v0_prob.sum(dim=-1), torch.ones(self.pred_v0_prob.shape[0], device=self.pred_v0_prob.device), atol=1e-5
        ):
            raise ValueError("pred_v0_prob rows must sum to one")
        if self.posterior_v_prev_prob.numel() and not torch.allclose(
            self.posterior_v_prev_prob.sum(dim=-1), torch.ones(self.posterior_v_prev_prob.shape[0], device=self.posterior_v_prev_prob.device), atol=1e-5
        ):
            raise ValueError("posterior_v_prev_prob rows must sum to one")


def _default_targetdiff_root() -> Path:
    return Path(__file__).resolve().parents[2] / "targetdiff-main" / "targetdiff-main"


def _official_score_model_class(targetdiff_root: Union[str, Path]):
    root = str(Path(targetdiff_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        module = importlib.import_module("models.molopt_score_model")
    except Exception as exc:
        raise RuntimeError(f"cannot import official TargetDiff from {root}: {exc}") from exc
    try:
        return module.ScorePosNet3D
    except AttributeError as exc:
        raise RuntimeError("official TargetDiff module has no ScorePosNet3D") from exc


def _as_easy_dict(value):
    try:
        from easydict import EasyDict
    except ImportError as exc:  # pragma: no cover - targetdiff env supplies easydict
        raise RuntimeError("easydict is required to load the official TargetDiff config") from exc
    if isinstance(value, dict):
        return EasyDict({key: _as_easy_dict(item) for key, item in value.items()})
    return value


def _randn_like(value: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    if generator is None:
        return torch.randn_like(value)
    return torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)


def _rand_like(value: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    if generator is None:
        return torch.rand_like(value)
    return torch.rand(value.shape, dtype=value.dtype, device=value.device, generator=generator)


def _index_to_log_onehot(index: torch.Tensor, num_classes: int) -> torch.Tensor:
    if index.dtype != torch.long or index.ndim != 1:
        raise ValueError("ligand type state must be LongTensor [Nl]")
    if index.numel() and (int(index.min()) < 0 or int(index.max()) >= num_classes):
        raise ValueError("ligand type state is out of range")
    return torch.log(F.one_hot(index, num_classes=num_classes).to(dtype=torch.float32).clamp_min(1e-30))


def _sample_log_categorical(
    log_probs: torch.Tensor,
    generator: Optional[torch.Generator],
    uniform: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    uniform = _rand_like(log_probs, generator) if uniform is None else uniform
    # Keep the same numerical expression as TargetDiff's log_sample_categorical;
    # the additive eps also defines the zero endpoint for deterministic traces.
    gumbel = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    return (gumbel + log_probs).argmax(dim=-1)


class TargetDiffAdapter:
    """Wrap the official score model without changing its source or checkpoint."""

    def __init__(self, model, *, checkpoint_config=None, device: Optional[Union[str, torch.device]] = None):
        self.model = model
        self.checkpoint_config = checkpoint_config
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.model.to(self.device)
        self.model.eval()
        self.num_classes = int(getattr(model, "num_classes", 13))
        self.num_timesteps = int(getattr(model, "num_timesteps", 1000))
        if self.num_classes != 13:
            raise ValueError(f"TargetDiff adapter requires 13 ligand classes, got {self.num_classes}")
        if self.num_timesteps <= 0:
            raise ValueError(f"TargetDiff adapter requires positive timesteps, got {self.num_timesteps}")

    @torch.no_grad()
    def initialize_sampling_state(
        self,
        *,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_protein: torch.Tensor,
        batch_ligand: torch.Tensor,
        apo_pos_ref: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        center_mode: str = "protein",
    ) -> TargetDiffState:
        """Create the official-style initial ligand state.

        The caller supplies ``batch_ligand`` so atom-count policy remains
        explicit.  Randomness is consumed in the same order as the official
        sampler: position Gaussian first, then uniform categorical sampling.
        Coordinates are centered exactly once using the apo protein mean.
        """

        if center_mode != "protein":
            raise ValueError("initialize_sampling_state requires center_mode='protein'")
        if protein_pos.device != self.device or protein_v.device != self.device:
            raise ValueError(f"protein tensors must be on adapter device {self.device}")
        if batch_protein.device != self.device or batch_ligand.device != self.device:
            raise ValueError(f"batch tensors must be on adapter device {self.device}")
        if protein_pos.ndim != 2 or protein_pos.shape[-1] != 3 or not protein_pos.is_floating_point():
            raise ValueError("protein_pos must be floating Tensor [Np, 3]")
        if protein_v.ndim != 2 or protein_v.shape != (protein_pos.shape[0], 27) or not protein_v.is_floating_point():
            raise ValueError("protein_v must be floating Tensor [Np, 27]")
        if batch_protein.dtype != torch.long or batch_protein.ndim != 1 or batch_protein.shape[0] != protein_pos.shape[0] or not batch_protein.numel():
            raise ValueError("batch_protein must be non-empty LongTensor [Np]")
        if batch_ligand.dtype != torch.long or batch_ligand.ndim != 1 or not batch_ligand.numel():
            raise ValueError("batch_ligand must be non-empty LongTensor [Nl]")
        if int(batch_protein.min()) < 0 or int(batch_ligand.min()) < 0:
            raise ValueError("batch ids cannot be negative")
        num_graphs = int(batch_protein.max().item()) + 1
        if int(batch_ligand.max()) >= num_graphs:
            raise ValueError("batch_ligand contains an out-of-range graph id")
        if apo_pos_ref is None:
            apo_pos_ref = protein_pos
        if apo_pos_ref.device != self.device or apo_pos_ref.shape != protein_pos.shape or not apo_pos_ref.is_floating_point():
            raise ValueError("apo_pos_ref must be floating Tensor with protein_pos shape on adapter device")

        center = torch.zeros((num_graphs, 3), dtype=protein_pos.dtype, device=self.device)
        center.index_add_(0, batch_protein, protein_pos)
        counts = torch.bincount(batch_protein, minlength=num_graphs).to(dtype=protein_pos.dtype)
        center = center / counts.clamp_min(1.0)[:, None]
        batch_center = center[batch_ligand]

        # Keep this order aligned with targetdiff-main/scripts/sample_diffusion.py.
        position_noise = _randn_like(batch_center, generator)
        initial_ligand_pos = batch_center + position_noise
        uniform_logits = torch.zeros(
            (batch_ligand.shape[0], self.num_classes),
            dtype=torch.float32,
            device=self.device,
        )
        initial_ligand_v = _sample_log_categorical(uniform_logits, generator)
        return initialize_targetdiff_state(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            ligand_pos=initial_ligand_pos,
            ligand_v=initial_ligand_v,
            batch_ligand=batch_ligand,
            apo_pos_ref=apo_pos_ref,
            center_mode="protein",
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        *,
        targetdiff_root: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> "TargetDiffAdapter":
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        payload = torch.load(checkpoint, map_location=device)
        if not isinstance(payload, dict) or "config" not in payload or "model" not in payload:
            raise ValueError("TargetDiff checkpoint must contain config and model")
        config = _as_easy_dict(payload["config"])
        if config.model.get("model_mean_type") != "C0":
            raise ValueError("minimal adapter only supports TargetDiff model_mean_type=C0")
        if config.model.get("center_pos_mode") != "protein":
            raise ValueError("minimal adapter requires TargetDiff center_pos_mode=protein")
        if int(config.model.get("num_diffusion_timesteps", -1)) != 1000:
            raise ValueError("minimal adapter requires the official 1000-step TargetDiff schedule")
        score_model_class = _official_score_model_class(targetdiff_root or _default_targetdiff_root())
        model = score_model_class(
            config.model,
            protein_atom_feature_dim=27,
            ligand_atom_feature_dim=13,
        ).to(device)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        return cls(model, checkpoint_config=config, device=device)

    def _validate_state(self, state: TargetDiffState) -> None:
        for name in (
            "protein_pos", "protein_v", "batch_protein", "ligand_pos", "ligand_v",
            "batch_ligand", "apo_pos_ref", "center_offset",
        ):
            if getattr(state, name).device != self.device:
                raise ValueError(f"state tensor {name} must be on adapter device {self.device}")
        if state.protein_v.dtype not in (torch.float16, torch.float32, torch.float64):
            raise TypeError("protein_v must be floating point")
        if state.protein_pos.dtype not in (torch.float16, torch.float32, torch.float64):
            raise TypeError("coordinates must be floating point")

    @torch.no_grad()
    def sample_step(
        self,
        state: TargetDiffState,
        t: int,
        generator: Optional[torch.Generator] = None,
        rng_trace: Optional[TargetDiffStepRandomness] = None,
    ) -> Tuple[TargetDiffState, TargetDiffStepAux]:
        """Apply one official reverse step, with input ``L_t`` and output ``L_{t-1}``."""

        self._validate_state(state)
        if generator is not None and rng_trace is not None:
            raise ValueError("provide either generator or rng_trace, not both")
        if not isinstance(t, int) or isinstance(t, bool) or t < 0 or t >= self.num_timesteps:
            raise ValueError(f"t must be an integer in [0, {self.num_timesteps - 1}]")
        t_graph = torch.full(
            (state.num_graphs,), t, dtype=torch.long, device=state.protein_pos.device
        )
        predictions = self.model(
            protein_pos=state.protein_pos,
            protein_v=state.protein_v,
            batch_protein=state.batch_protein,
            init_ligand_pos=state.ligand_pos,
            init_ligand_v=state.ligand_v,
            batch_ligand=state.batch_ligand,
            time_step=t_graph,
        )
        pred_x0 = predictions["pred_ligand_pos"]
        pred_v0_logits = predictions["pred_ligand_v"]
        pred_v0_prob = F.softmax(pred_v0_logits, dim=-1)

        pos_mean = self.model.q_pos_posterior(
            x0=pred_x0,
            xt=state.ligand_pos,
            t=t_graph,
            batch=state.batch_ligand,
        )
        pos_logvar = self.model.posterior_logvar[t_graph][state.batch_ligand].unsqueeze(-1)
        categorical_uniform = None
        if rng_trace is None:
            position_noise = _randn_like(state.ligand_pos, generator)
        else:
            if rng_trace.position_noise.shape != state.ligand_pos.shape:
                raise ValueError("RNG trace position_noise shape does not match state ligand_pos")
            if rng_trace.categorical_uniform.shape != (state.ligand_pos.shape[0], self.num_classes):
                raise ValueError("RNG trace categorical_uniform shape does not match state ligand_pos")
            position_noise = rng_trace.position_noise.to(
                device=state.ligand_pos.device, dtype=state.ligand_pos.dtype
            )
            categorical_uniform = rng_trace.categorical_uniform.to(
                device=state.ligand_pos.device, dtype=state.ligand_pos.dtype
            )
        nonzero = (t_graph != 0).to(dtype=state.ligand_pos.dtype)[state.batch_ligand, None]
        ligand_pos_prev = pos_mean + nonzero * (0.5 * pos_logvar).exp() * position_noise

        log_v0 = F.log_softmax(pred_v0_logits, dim=-1)
        log_vt = _index_to_log_onehot(state.ligand_v, self.num_classes)
        log_p_vprev = self.model.q_v_posterior(
            log_v0,
            log_vt,
            t_graph,
            state.batch_ligand,
        )
        posterior_v_prev_prob = log_p_vprev.exp()
        ligand_v_prev = _sample_log_categorical(log_p_vprev, generator, uniform=categorical_uniform)
        next_state = state.replace(ligand_pos=ligand_pos_prev, ligand_v=ligand_v_prev)
        aux = TargetDiffStepAux(
            pred_x0=pred_x0,
            pred_v0_prob=pred_v0_prob,
            posterior_v_prev_prob=posterior_v_prev_prob,
        )
        return next_state, aux

    @torch.no_grad()
    def run_steps(
        self,
        state: TargetDiffState,
        *,
        t_start: int,
        t_end_inclusive: int,
        generator: Optional[torch.Generator] = None,
        rng_trace: Optional[TargetDiffRNGTrace] = None,
    ) -> Tuple[TargetDiffState, Tuple[TargetDiffStepAux, ...]]:
        """Run descending steps ``t_start, ..., t_end_inclusive``."""

        if t_start < t_end_inclusive:
            raise ValueError("t_start must be >= t_end_inclusive")
        if generator is not None and rng_trace is not None:
            raise ValueError("provide either generator or rng_trace, not both")
        current = state
        auxiliaries = []
        for t in range(t_start, t_end_inclusive - 1, -1):
            step_trace = rng_trace.for_step(t) if rng_trace is not None else None
            current, aux = self.sample_step(
                current,
                t,
                generator=generator,
                rng_trace=step_trace,
            )
            auxiliaries.append(aux)
        return current, tuple(auxiliaries)


__all__ = [
    "TargetDiffAdapter",
    "TargetDiffRNGTrace",
    "TargetDiffStepAux",
    "TargetDiffStepRandomness",
]
