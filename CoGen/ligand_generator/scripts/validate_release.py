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
    parser.add_argument("--metadata", type=Path, default=Path("METADATA.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/md_dynamic.yml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("pretrained_models/pretrained_diffusion.pt"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/md300/prepared_targetdiff_md300_10frames.pt"),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text())
    required_files = [
        args.config,
        args.checkpoint,
        args.cache,
        Path("models/aplc.py"),
        Path("scripts/train_md_dynamic.py"),
        Path("tests/test_aplc.py"),
    ]
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing release files: {missing_files}")

    config = misc.load_config(str(args.config))
    cache_payload = torch.load(
        str(args.cache),
        map_location="cpu",
        weights_only=False,
    )
    cache_metadata = cache_payload["metadata"]
    feature_contract = cache_metadata["feature_contract"]
    if feature_contract["protein_atom_feature_dim"] != 27:
        raise ValueError("Release cache is not TargetDiff 27D protein input")
    if feature_contract["ligand_atom_class_dim"] != 13:
        raise ValueError("Release cache is not TargetDiff 13-class ligand input")

    model = ScorePosNet3D(config.model, 27, 13)
    checkpoint = torch.load(
        str(args.checkpoint),
        map_location="cpu",
        weights_only=False,
    )
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if unexpected or any(not key.startswith("refine_net.aplc.") for key in missing):
        raise RuntimeError(
            f"Unexpected original-checkpoint compatibility result: "
            f"missing={missing}, unexpected={unexpected}"
        )

    dataset = PreparedMDDataset(args.cache, split="valid")
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=["ligand_nbh_list"],
    )
    batch = next(iter(loader)).to(torch.device(args.device))
    model.to(torch.device(args.device))
    model.eval()
    with torch.no_grad():
        result = model(
            protein_pos=batch.protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch.protein_element_batch,
            init_ligand_pos=batch.ligand_pos,
            init_ligand_v=batch.ligand_atom_feature_full,
            batch_ligand=batch.ligand_element_batch,
            time_step=torch.zeros(
                batch.num_graphs,
                dtype=torch.long,
                device=batch.protein_pos.device,
            ),
        )
    if not torch.isfinite(result["pred_ligand_pos"]).all():
        raise ValueError("Release forward produced non-finite coordinates")
    if not torch.isfinite(result["pred_ligand_v"]).all():
        raise ValueError("Release forward produced non-finite atom logits")

    report = {
        "passed": True,
        "project": metadata.get("project"),
        "source_commit": metadata.get("source_commit"),
        "cache_samples": len(cache_payload["samples"]),
        "cache_split_counts": cache_metadata.get("split_counts", {}),
        "protein_feature_dim": 27,
        "ligand_class_dim": 13,
        "aplc_enabled": bool(config.model.aplc_enabled),
        "batch_protein_shape": list(batch.protein_atom_feature.shape),
        "batch_ligand_shape": list(batch.ligand_atom_feature_full.shape),
        "pred_ligand_pos_shape": list(result["pred_ligand_pos"].shape),
        "pred_ligand_v_shape": list(result["pred_ligand_v"].shape),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
