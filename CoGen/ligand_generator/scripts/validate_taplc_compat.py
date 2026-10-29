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
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = misc.load_config(str(args.config))
    model = ScorePosNet3D(config.model, 27, 13)
    payload = torch.load(
        str(args.checkpoint),
        map_location="cpu",
        weights_only=False,
    )
    missing, unexpected = model.load_state_dict(
        payload["model"],
        strict=False,
    )
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    if not missing or not all(
        key.startswith("refine_net.aplc.") for key in missing
    ):
        raise RuntimeError(f"Unexpected missing checkpoint keys: {missing}")

    dataset = PreparedMDDataset(args.cache, split="valid")
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=["ligand_nbh_list"],
    )
    device = torch.device(args.device)
    batch = next(iter(loader)).to(device)
    model.to(device).eval()
    with torch.no_grad():
        output = model(
            protein_pos=batch.protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch.protein_element_batch,
            init_ligand_pos=batch.ligand_pos,
            init_ligand_v=batch.ligand_atom_feature_full,
            batch_ligand=batch.ligand_element_batch,
            time_step=torch.zeros(
                batch.num_graphs,
                dtype=torch.long,
                device=device,
            ),
        )
    finite = bool(
        torch.isfinite(output["pred_ligand_pos"]).all()
        and torch.isfinite(output["pred_ligand_v"]).all()
    )
    if not finite:
        raise FloatingPointError("T-APLC compatibility forward is non-finite")

    print(
        json.dumps(
            {
                "passed": True,
                "source_iteration": payload.get("iteration"),
                "missing_new_parameters": missing,
                "protein_features": list(batch.protein_atom_feature.shape),
                "ligand_classes": list(batch.ligand_atom_feature_full.shape),
                "pred_ligand_pos": list(output["pred_ligand_pos"].shape),
                "pred_ligand_v": list(output["pred_ligand_v"].shape),
                "finite": finite,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
