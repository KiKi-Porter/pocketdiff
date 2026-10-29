import json
from pathlib import Path

import torch

from models.molopt_score_model import ScorePosNet3D


def main():
    root = Path(__file__).resolve().parents[1]
    checkpoint_path = root / "pretrained_models" / "pretrained_diffusion.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = ScorePosNet3D(
        config=config.model,
        protein_atom_feature_dim=27,
        ligand_atom_feature_dim=13,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    generator = torch.Generator(device="cpu").manual_seed(20261001)
    n_protein, n_ligand = 32, 12
    protein_pos = torch.randn((n_protein, 3), generator=generator)
    ligand_pos = torch.randn((n_ligand, 3), generator=generator)
    protein_v = torch.zeros((n_protein, 27), dtype=torch.float32)
    protein_v[:, 0] = 1.0
    ligand_v = torch.randint(0, 13, (n_ligand,), generator=generator)
    batch_protein = torch.zeros(n_protein, dtype=torch.long)
    batch_ligand = torch.zeros(n_ligand, dtype=torch.long)

    with torch.no_grad():
        output = model(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            init_ligand_pos=ligand_pos,
            init_ligand_v=ligand_v,
            batch_ligand=batch_ligand,
            time_step=None,
            fix_x=True,
        )
        sample_output = model.sample_diffusion(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            init_ligand_pos=ligand_pos,
            init_ligand_v=ligand_v,
            batch_ligand=batch_ligand,
            num_steps=1,
            center_pos_mode=config.model.center_pos_mode,
            pos_only=False,
            return_trajectory=True,
            show_progress=False,
        )

    finite = all(torch.isfinite(value).all().item() for value in (
        output["pred_ligand_pos"],
        output["pred_ligand_v"],
        sample_output["pos"],
    ))
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_keys": len(checkpoint["model"]),
        "protein_feature_dim": 27,
        "ligand_feature_dim": 13,
        "pred_ligand_pos_shape": list(output["pred_ligand_pos"].shape),
        "pred_ligand_v_shape": list(output["pred_ligand_v"].shape),
        "sampled_ligand_pos_shape": list(sample_output["pos"].shape),
        "sample_trajectory_length": len(sample_output["pos_traj"]),
        "forward_finite": bool(finite),
        "passed": bool(finite),
    }
    report_path = root / "runs" / "original_smoke_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not finite:
        raise RuntimeError("Original TargetDiff smoke produced non-finite output.")


if __name__ == "__main__":
    main()
