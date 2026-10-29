"""Compare the official TargetDiff sampling loop with the replayable adapter.

The official implementation is kept untouched.  Its two stochastic tensor
constructors are temporarily redirected to a fixed :class:`TargetDiffRNGTrace`
so that both implementations consume exactly the same reverse-diffusion noise.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.targetdiff import (
    TargetDiffAdapter,
    TargetDiffRNGTrace,
    TargetDiffStepRandomness,
    initialize_targetdiff_state,
)


def _find_record(records):
    for record in records:
        if str(record[0]).startswith("3txj") or str(record[1]).startswith("3txj"):
            return record
    raise RuntimeError("3txj record was not found in the Apo2Mol train split")


def _build_trace(state, timesteps, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return TargetDiffRNGTrace(
        {
            timestep: TargetDiffStepRandomness(
                position_noise=torch.randn(
                    state.ligand_pos.shape,
                    dtype=state.ligand_pos.dtype,
                    device=state.ligand_pos.device,
                    generator=generator,
                ),
                categorical_uniform=torch.rand(
                    (state.ligand_pos.shape[0], 13),
                    dtype=state.ligand_pos.dtype,
                    device=state.ligand_pos.device,
                    generator=generator,
                ),
            )
            for timestep in timesteps
        }
    )


def _max_abs(first, second):
    first = first.detach().cpu()
    second = second.detach().cpu()
    if first.numel() == 0:
        return 0.0
    return float((first - second).abs().max().item())


def _run_official_with_trace(model, state, trace, timesteps):
    """Run official sample_diffusion while replaying the supplied randomness."""

    original_randn_like = torch.randn_like
    original_rand_like = torch.rand_like
    position_index = 0
    categorical_index = 0

    def traced_randn_like(value, *args, **kwargs):
        del args, kwargs
        nonlocal position_index
        if position_index >= len(timesteps):
            raise AssertionError("official sampler requested too many position noise tensors")
        randomness = trace.for_step(timesteps[position_index])
        position_index += 1
        expected = randomness.position_noise.to(device=value.device, dtype=value.dtype)
        if expected.shape != value.shape:
            raise AssertionError(
                "official position noise shape differs from trace: "
                f"{tuple(value.shape)} vs {tuple(expected.shape)}"
            )
        return expected.clone()

    def traced_rand_like(value, *args, **kwargs):
        del args, kwargs
        nonlocal categorical_index
        if categorical_index >= len(timesteps):
            raise AssertionError("official sampler requested too many categorical uniform tensors")
        randomness = trace.for_step(timesteps[categorical_index])
        categorical_index += 1
        expected = randomness.categorical_uniform.to(device=value.device, dtype=value.dtype)
        if expected.shape != value.shape:
            raise AssertionError(
                "official categorical uniform shape differs from trace: "
                f"{tuple(value.shape)} vs {tuple(expected.shape)}"
            )
        return expected.clone()

    torch.randn_like = traced_randn_like
    torch.rand_like = traced_rand_like
    try:
        result = model.sample_diffusion(
            protein_pos=state.protein_pos,
            protein_v=state.protein_v,
            batch_protein=state.batch_protein,
            init_ligand_pos=state.ligand_pos,
            init_ligand_v=state.ligand_v,
            batch_ligand=state.batch_ligand,
            num_steps=len(timesteps),
            # The official implementation's ``none`` branch leaves offset as
            # a scalar and later indexes it, so the reference run uses its
            # normal protein-centering branch on raw coordinates.
            center_pos_mode="protein",
            pos_only=False,
            return_trajectory=True,
            show_progress=False,
        )
    finally:
        torch.randn_like = original_randn_like
        torch.rand_like = original_rand_like

    if position_index != len(timesteps) or categorical_index != len(timesteps):
        raise AssertionError(
            "official sampler did not consume one position and one categorical tensor per step: "
            f"position={position_index}, categorical={categorical_index}, expected={len(timesteps)}"
        )
    return result


def _make_state(value, center_mode):
    protein_batch = torch.zeros(value.num_protein_atoms, dtype=torch.long)
    ligand_batch = torch.zeros(value.num_ligand_atoms, dtype=torch.long)
    return initialize_targetdiff_state(
        protein_pos=value.protein_pos_apo.float(),
        protein_v=value.protein_feature.float(),
        batch_protein=protein_batch,
        ligand_pos=value.ligand_pos_ref.float(),
        ligand_v=value.ligand_type_ref.long(),
        batch_ligand=ligand_batch,
        apo_pos_ref=value.protein_pos_apo.float(),
        center_mode=center_mode,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("Apo2Mol-main/Apo2MOl-dataset/data_folder"),
    )
    parser.add_argument(
        "--split-pickle",
        type=Path,
        default=Path("Apo2Mol-main/Apo2MOl-dataset/split_druglike_dict.pkl"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("targetdiff-main/targetdiff-main/pretrained_models/pretrained_diffusion.pt"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--atol", type=float, default=2e-6)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.split_pickle.open("rb") as handle:
        record = _find_record(pickle.load(handle)["train"])
    value = Apo2MolAdapter(args.data_root).convert_record(record)
    initial_state = _make_state(value, center_mode="protein")
    official_state = _make_state(value, center_mode="none")
    before = {
        name: getattr(initial_state, name).clone()
        for name in (
            "protein_pos",
            "protein_v",
            "batch_protein",
            "batch_ligand",
            "apo_pos_ref",
            "center_offset",
            "ligand_pos",
            "ligand_v",
        )
    }

    adapter = TargetDiffAdapter.from_checkpoint(args.checkpoint, device="cpu")
    if adapter.num_timesteps != 1000:
        raise RuntimeError(f"expected official 1000-step schedule, got {adapter.num_timesteps}")
    timesteps = list(reversed(range(adapter.num_timesteps - 4, adapter.num_timesteps)))
    expected_timesteps = [999, 998, 997, 996]
    if timesteps != expected_timesteps:
        raise RuntimeError(f"unexpected official short-window time sequence: {timesteps}")
    trace = _build_trace(initial_state, timesteps, args.seed)

    official = _run_official_with_trace(adapter.model, official_state, trace, timesteps)
    adapter_snapshots = []
    auxiliaries = []
    current = initial_state
    for timestep in timesteps:
        current, auxiliary = adapter.sample_step(
            current, timestep, rng_trace=trace.for_step(timestep)
        )
        adapter_snapshots.append(current)
        auxiliaries.append(auxiliary)
    adapted = current

    official_offset = official_state.protein_pos.mean(dim=0, keepdim=True)
    per_step = []
    max_position_error = 0.0
    max_type_error = 0.0
    max_v0_log_error = 0.0
    max_posterior_error = 0.0
    for index, timestep in enumerate(timesteps):
        current = adapter_snapshots[index]
        official_position = official["pos_traj"][index] - official_offset[official_state.batch_ligand]
        official_type = official["v_traj"][index]
        current_position_error = _max_abs(current.ligand_pos, official_position)
        current_type_error = int((current.ligand_v.cpu() != official_type).sum().item())
        v0_log_error = _max_abs(torch.log(auxiliaries[index].pred_v0_prob.clamp_min(1e-30)), official["v0_traj"][index])
        posterior_error = _max_abs(auxiliaries[index].posterior_v_prev_prob, official["vt_traj"][index].exp())
        max_position_error = max(max_position_error, current_position_error)
        max_type_error = max(max_type_error, float(current_type_error))
        max_v0_log_error = max(max_v0_log_error, v0_log_error)
        max_posterior_error = max(max_posterior_error, posterior_error)
        per_step.append(
            {
                "timestep": timestep,
                "position_max_abs_error": current_position_error,
                "type_mismatch_count": current_type_error,
                "v0_log_max_abs_error": v0_log_error,
                "posterior_max_abs_error": posterior_error,
            }
        )

    official_final_position = official["pos"] - official_offset[official_state.batch_ligand]
    final_position_error = _max_abs(adapted.ligand_pos, official_final_position)
    final_type_mismatch = int((adapted.ligand_v.cpu() != official["v"]).sum().item())
    max_position_error = max(max_position_error, final_position_error)
    max_type_error = max(max_type_error, float(final_type_mismatch))

    invariant_checks = {
        "adapter_protein_unchanged": bool(torch.equal(adapted.protein_pos, before["protein_pos"])),
        "adapter_features_unchanged": bool(torch.equal(adapted.protein_v, before["protein_v"])),
        "adapter_batch_unchanged": bool(
            torch.equal(adapted.batch_protein, before["batch_protein"])
            and torch.equal(adapted.batch_ligand, before["batch_ligand"])
        ),
        "adapter_apo_reference_unchanged": bool(torch.equal(adapted.apo_pos_ref, before["apo_pos_ref"])),
        "adapter_center_offset_unchanged": bool(torch.equal(adapted.center_offset, before["center_offset"])),
        "input_state_not_mutated": bool(
            all(torch.equal(getattr(initial_state, name), value) for name, value in before.items())
        ),
        "atom_count_unchanged": bool(
            adapted.protein_pos.shape == initial_state.protein_pos.shape
            and adapted.ligand_pos.shape == initial_state.ligand_pos.shape
        ),
    }
    report = {
        "sample_id": value.sample_id,
        "seed": args.seed,
        "timesteps": timesteps,
        "num_protein_atoms": int(initial_state.protein_pos.shape[0]),
        "num_ligand_atoms": int(initial_state.ligand_pos.shape[0]),
        "max_position_max_abs_error": max_position_error,
        "max_type_mismatch_count": int(max_type_error),
        "final_position_max_abs_error": final_position_error,
        "final_type_mismatch_count": final_type_mismatch,
        "max_v0_log_max_abs_error": max_v0_log_error,
        "max_posterior_max_abs_error": max_posterior_error,
        "per_step": per_step,
        "invariant_checks": invariant_checks,
        "tolerance": args.atol,
        "official_center_mode": "protein",
        "official_center_offset": official_offset.tolist(),
    }
    report["passed"] = bool(
        max_position_error <= args.atol
        and max_type_error == 0
        and max_v0_log_error <= args.atol
        and max_posterior_error <= args.atol
        and all(invariant_checks.values())
        and torch.all(torch.isfinite(adapted.ligand_pos)).item()
        and torch.all((adapted.ligand_v >= 0) & (adapted.ligand_v < 13)).item()
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise RuntimeError(f"TargetDiff equivalence smoke failed: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
