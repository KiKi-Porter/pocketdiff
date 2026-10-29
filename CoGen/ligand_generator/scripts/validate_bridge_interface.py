from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from datasets.md_dataset import PreparedMDDataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D
from utils import misc


def _load_checkpoint(config, checkpoint_path: Path, device: torch.device):
    model = ScorePosNet3D(config.model, 27, 13).to(device)
    payload = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    missing, unexpected = model.load_state_dict(payload["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Dynamic checkpoint is not self-contained: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/md_dynamic.yml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/md_dynamic_300/partial_best.pt"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/md300/prepared_targetdiff_md300_10frames.pt"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.steps < 20:
        raise ValueError("Bridge protocol requires at least 20 TargetDiff steps")
    misc.seed_all(args.seed)
    device = torch.device(args.device)
    config = misc.load_config(str(args.config))
    model, checkpoint = _load_checkpoint(config, args.checkpoint, device)

    dataset = PreparedMDDataset(args.cache, split="valid")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=["ligand_nbh_list"],
    )
    batch = next(iter(loader)).to(device)
    protein_pos = batch.protein_pos
    protein_v = batch.protein_atom_feature.float()
    batch_protein = batch.protein_element_batch
    batch_ligand = batch.ligand_element_batch

    # This schedule stands in for the future PocketDiff output. It preserves
    # atom order and features while applying a small deterministic backbone
    # translation over the 20 TargetDiff calls.
    schedule = protein_pos.unsqueeze(0).repeat(args.steps, 1, 1)
    progress = torch.linspace(
        0.0,
        1.0,
        args.steps,
        device=device,
        dtype=protein_pos.dtype,
    )
    schedule = schedule + progress[:, None, None] * torch.tensor(
        [0.1, -0.05, 0.08],
        device=device,
        dtype=protein_pos.dtype,
    )

    ligand_center = protein_pos.mean(dim=0, keepdim=True)
    init_ligand_pos = ligand_center + torch.randn_like(batch.ligand_pos)
    init_ligand_v = batch.ligand_atom_feature_full

    with torch.no_grad():
        result = model.sample_diffusion(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            init_ligand_pos=init_ligand_pos,
            init_ligand_v=init_ligand_v,
            batch_ligand=batch_ligand,
            num_steps=args.steps,
            center_pos_mode="protein",
            pos_only=False,
            protein_pos_schedule=schedule,
            return_trajectory=True,
            show_progress=False,
        )

    if len(result["pos_traj"]) != args.steps:
        raise AssertionError(
            f"Expected {args.steps} ligand trajectory states, "
            f"got {len(result['pos_traj'])}"
        )
    if not torch.isfinite(result["pos"]).all():
        raise ValueError("Bridge interface produced non-finite ligand coordinates")
    if not torch.isfinite(result["v"].float()).all():
        raise ValueError("Bridge interface produced invalid ligand classes")
    for trajectory_state in result["pos_traj"]:
        if not torch.isfinite(trajectory_state).all():
            raise ValueError("Bridge trajectory contains non-finite coordinates")

    report = {
        "passed": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_stage": checkpoint.get("stage"),
        "steps": args.steps,
        "protein_atoms": int(protein_pos.shape[0]),
        "ligand_atoms": int(result["pos"].shape[0]),
        "protein_feature_dim": int(protein_v.shape[-1]),
        "ligand_class_dim": 13,
        "protein_schedule_shape": list(schedule.shape),
        "ligand_trajectory_length": len(result["pos_traj"]),
        "final_ligand_pos_shape": list(result["pos"].shape),
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
