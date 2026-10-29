"""Validate checkpoint reload on the device used by Phase 70 bounded runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pocketdiff.diffusion import sample_diffusion_state
from pocketdiff.training.diffusion import (
    DiffusionTrainConfig,
    _generator_for_device,
    _move_complex_to_device,
    load_apo2mol_slices,
    load_diffusion_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    config = DiffusionTrainConfig(
        device=args.device,
        train_count=8,
        valid_count=4,
        test_count=4,
        holdout_count=4,
        updates=1,
    )
    value = _move_complex_to_device(
        load_apo2mol_slices(config)["train"].values[0],
        device,
    )
    result = {}
    for checkpoint in sorted(args.root.glob("*/checkpoint.pt")):
        loaded = load_diffusion_checkpoint(checkpoint, map_location=str(device))
        state = sample_diffusion_state(
            value,
            0.5,
            generator=_generator_for_device(device, config.diffusion_seed + 999999),
        )
        loaded.model.eval()
        with torch.no_grad():
            first = loaded.model(state.model_input)
            second = loaded.model(state.model_input)
        max_delta = max(
            float((first.translation_local - second.translation_local).abs().max()),
            float((first.rotation_local - second.rotation_local).abs().max()),
            float((first.chi - second.chi).abs().max()),
        )
        result[checkpoint.parent.name] = {
            "max_repeat_delta": max_delta,
            "close_at_2e-6": bool(max_delta <= 2e-6),
        }
    output = args.root / "reload_validation.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
