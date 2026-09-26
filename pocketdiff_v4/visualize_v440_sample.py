from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .batching import collate_complexes
from .cache import load_cache
from .evaluate import _sample_metrics, _trajectory_metrics
from .model import PocketDiffV4Model
from .sampler import sample_complexes_with_trajectory


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _load_checkpoint(path, device):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["config"]
    model = PocketDiffV4Model(
        hidden=int(config["hidden"]),
        vector_channels=int(config["vector_channels"]),
        layers=int(config["layers"]),
        radial_count=int(config.get("radial_count", 24)),
        radial_cutoff=float(config.get("radial_cutoff", 16.0)),
        max_translation=float(config.get("max_translation", torch.pi)),
        max_rotation=float(config.get("max_rotation", torch.pi)),
        max_chi_step=float(config.get("max_chi_step", torch.pi)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def _pdb_name(name, fallback):
    name = str(name).strip()
    return (name[:4] if name else fallback).ljust(4)


def _write_pdb(path, record, positions):
    inputs = record["input"]
    atom_names = inputs.get("atom_names", [])
    residue_names = inputs.get("residue_names", [])
    atom_to_residue = inputs["atom_to_residue"].tolist()
    ligand_pos = inputs["ligand_pos"]
    ligand_type = inputs["ligand_type"].tolist()
    lines = []
    for atom_index, coordinate in enumerate(positions.tolist()):
        residue_index = atom_to_residue[atom_index]
        residue_name = (
            residue_names[residue_index]
            if residue_index < len(residue_names)
            else "UNK"
        )
        atom_name = (
            atom_names[atom_index]
            if atom_index < len(atom_names)
            else "X"
        )
        x, y, z = coordinate
        lines.append(
            f"ATOM  {atom_index + 1:5d} {_pdb_name(atom_name, ' X ')} "
            f"{str(residue_name)[:3]:>3s} A{residue_index + 1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    offset = len(lines)
    for ligand_index, (coordinate, atom_class) in enumerate(
        zip(ligand_pos.tolist(), ligand_type)
    ):
        x, y, z = coordinate
        lines.append(
            f"HETATM{offset + ligand_index + 1:5d}  L{int(atom_class):2d} LIG B"
            f"{ligand_index + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
            "  1.00  0.00           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def _backbone_indices(record):
    mask = record["input"]["backbone_mask"].bool()
    return torch.where(mask)[0]


def _plot_trajectory(path, record, trajectory, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inputs, targets = record["input"], record["target"]
    apo = inputs["apo_pos"].cpu()
    holo = targets["holo_pos"].cpu()
    ligand = inputs["ligand_pos"].cpu()
    backbone_indices = _backbone_indices(record)
    states = [state.cpu() for state in trajectory]
    display_states = states + [holo]
    labels = ["apo", "step 1", "step 2", "holo reference"]
    colors = {"N": "#2563eb", "CA": "#f59e0b", "C": "#16a34a", "O": "#dc2626"}
    frame_index = inputs["frame_index"]
    atom_names = inputs.get("atom_names", [])
    ca_indices = frame_index[:, 1]
    ca_indices = ca_indices[ca_indices >= 0]

    all_points = torch.cat(
        [state[backbone_indices] for state in display_states]
        + [ligand],
        dim=0,
    )
    lower = all_points.min(dim=0).values - 1.0
    upper = all_points.max(dim=0).values + 1.0

    figure = plt.figure(figsize=(16, 10), dpi=160)
    axes = [figure.add_subplot(2, 3, index + 1, projection="3d") for index in range(4)]
    for axis, state, label in zip(axes, display_states, labels):
        for atom_index in backbone_indices.tolist():
            name = str(atom_names[atom_index]).upper() if atom_index < len(atom_names) else "X"
            color = colors.get(name, "#64748b")
            point = state[atom_index]
            axis.scatter(
                [float(point[0])], [float(point[1])], [float(point[2])],
                s=18, color=color, depthshade=False,
            )
        if len(ca_indices) > 1:
            ca = state[ca_indices].numpy()
            axis.plot(ca[:, 0], ca[:, 1], ca[:, 2], color="#111827", linewidth=0.7)
        ligand_xyz = ligand.numpy()
        axis.scatter(
            ligand_xyz[:, 0], ligand_xyz[:, 1], ligand_xyz[:, 2],
            s=22, marker="x", color="#6b7280", alpha=0.8,
        )
        axis.set_title(label)
        axis.set_xlim(float(lower[0]), float(upper[0]))
        axis.set_ylim(float(lower[1]), float(upper[1]))
        axis.set_zlim(float(lower[2]), float(upper[2]))
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.view_init(elev=22, azim=-62)

    trajectory_rows = metrics["trajectory"]
    steps = [row["step"] for row in trajectory_rows]
    metric_axis = figure.add_subplot(2, 3, 5)
    metric_axis.plot(
        steps,
        [row["holo_backbone_rmsd"] for row in trajectory_rows],
        marker="o",
        label="to holo",
        color="#dc2626",
    )
    metric_axis.plot(
        steps,
        [row["apo_backbone_rmsd"] for row in trajectory_rows],
        marker="o",
        label="from apo",
        color="#2563eb",
    )
    metric_axis.set_xlabel("inference state")
    metric_axis.set_ylabel("backbone RMSD (A)")
    metric_axis.set_title("Backbone trajectory")
    metric_axis.grid(alpha=0.25)
    metric_axis.legend()

    text_axis = figure.add_subplot(2, 3, 6)
    final = trajectory_rows[-1]
    text = (
        f"sample: {record['sample_id']}\n"
        f"steps: {len(states) - 1}\n"
        f"final backbone RMSD: {final['holo_backbone_rmsd']:.3f} A\n"
        f"apo backbone RMSD: {final['apo_backbone_rmsd']:.3f} A\n"
        f"improvement: {metrics['improvement_backbone_rmsd']:.3f} A\n"
        f"direction cosine: {final['direction_cosine']:.3f}\n\n"
        "blue N, orange CA, green C, red O\n"
        "gray x: ligand atoms"
    )
    text_axis.axis("off")
    text_axis.text(0.02, 0.98, text, va="top", family="monospace", fontsize=11)
    figure.suptitle(
        "PocketDiff v4.4 apo + ligand conditioned backbone rollout",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="pocketdiff_v4/data/residue_graphs_v42_contract.pt",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "pocketdiff_v4/runs/v440_launch_probe_20260925/best.pt"
        ),
    )
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cache = load_cache(args.data)
    record = next(
        record
        for record in cache["splits"]["test"]
        if record["sample_id"] == args.sample_id
    )
    model = _load_checkpoint(args.checkpoint, device)
    batch = _move(collate_complexes([record]), device)
    predicted, trajectory = sample_complexes_with_trajectory(
        model,
        batch["input"],
        [args.sample_id],
        steps=args.steps,
        seed=args.seed,
        motion_scale=0.5,
        initial_noise_scale=0.0,
        disable_chi=True,
        schedule_type="remaining",
    )
    predicted = predicted.detach()
    trajectory = [state.detach() for state in trajectory]
    sampled_record = {
        "sample_id": record["sample_id"],
        "input": {
            key: (
                value.detach().cpu()
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in record["input"].items()
        },
        "target": {
            key: (
                value.detach().cpu()
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in record["target"].items()
        },
    }
    metrics = _sample_metrics(sampled_record, predicted.cpu())
    metrics["trajectory"] = _trajectory_metrics(
        sampled_record,
        [state.cpu() for state in trajectory],
        schedule_type="remaining",
        disable_chi=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, state in enumerate(trajectory):
        _write_pdb(
            output_dir / f"state_{index:02d}.pdb",
            sampled_record,
            state.cpu(),
        )
    _write_pdb(output_dir / "holo_reference.pdb", sampled_record, sampled_record["target"]["holo_pos"])
    torch.save(
        {
            "sample_id": args.sample_id,
            "trajectory": torch.stack([state.cpu() for state in trajectory]),
            "holo_pos": sampled_record["target"]["holo_pos"],
            "apo_pos": sampled_record["input"]["apo_pos"],
        },
        output_dir / "trajectory.pt",
    )
    _plot_trajectory(output_dir / "trajectory.png", sampled_record, trajectory, metrics)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
