from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from .batching import collate_complexes
from .cache import load_cache
from .model import PocketDiffV4Model
from .train import _rollout_loss, _save_checkpoint


def main():
    random.seed(317)
    torch.manual_seed(317)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cache = load_cache("pocketdiff_v4/data/residue_graphs_v41.pt")
    sample = cache["splits"]["train"][0]
    batch = collate_complexes([sample])

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value

    batch = move(batch)
    model = PocketDiffV4Model(hidden=128, vector_channels=12, layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    losses = []
    for update in range(60):
        random.seed(317)
        torch.manual_seed(317)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(317)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = _rollout_loss(
            model,
            batch,
            max_steps=4,
            oracle_rollout=True,
            disable_chi=True,
            direction_weight=0.1,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("smoke rollout produced non-finite loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("smoke rollout produced non-finite gradients")
        optimizer.step()
        losses.append(float(loss.detach()))

    output_dir = Path("pocketdiff_v4/runs/smoke")
    checkpoint_path = output_dir / "checkpoint.pt"
    module = model
    _save_checkpoint(checkpoint_path, {
        "format": "pocketdiff-v4-smoke-checkpoint",
        "model": module.state_dict(),
        "config": {"hidden": 128, "vector_channels": 12, "layers": 4},
    })
    reloaded = PocketDiffV4Model(hidden=128, vector_channels=12, layers=4).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    reloaded.load_state_dict(checkpoint["model"], strict=True)
    reloaded.eval()
    with torch.no_grad():
        inputs = batch["input"]
        prediction = reloaded(
            {"input": inputs}, inputs["apo_pos"], torch.tensor(0.5, device=device)
        )
    if not all(torch.isfinite(value).all() for value in prediction.values()):
        raise FloatingPointError("reloaded model produced non-finite output")

    summary = {
        "status": "passed",
        "device": str(device),
        "sample_id": sample["sample_id"],
        "updates": len(losses),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "last_10_mean_loss": sum(losses[-10:]) / 10.0,
        "strict_checkpoint_reload": True,
        "finite_multistep_rollout": True,
        "performance_evaluation": "not run",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
