"""Plot PocketDiff training and validation losses from a JSONL train log."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def _read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            # The DDP launcher appends a pretty-printed final report after the
            # JSONL progress rows. Only one-line update records belong here.
            continue
        if "update" in value:
            rows.append(value)
    if not rows:
        raise ValueError(f"no training rows found in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = _read_rows(args.log)
    updates = [int(row["update"]) for row in rows]
    train_loss = [float(row["loss"]) for row in rows]
    valid_rows = [
        row for row in rows
        if row.get("validation_loss") is not None
    ]

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(
        2, 1, figsize=(6.4, 5.2), sharex=True,
        gridspec_kw={"height_ratios": (2.0, 1.0)},
    )

    axes[0].plot(updates, train_loss, color="#2f5597", linewidth=0.8,
                 alpha=0.35, label="train loss")
    if len(train_loss) >= 25:
        window = min(101, max(5, len(train_loss) // 25 * 2 + 1))
        if window % 2 == 0:
            window += 1
        kernel = [1.0 / window] * window
        smooth = []
        padded = [train_loss[0]] * (window // 2) + train_loss + [train_loss[-1]] * (window // 2)
        for index in range(len(train_loss)):
            smooth.append(sum(
                padded[index + offset] * kernel[offset]
                for offset in range(window)
            ))
        axes[0].plot(updates, smooth, color="#1f3b73", linewidth=1.5,
                     label=f"train loss ({window}-update mean)")
    if valid_rows:
        valid_updates = [int(row["update"]) for row in valid_rows]
        valid_loss = [float(row["validation_loss"]) for row in valid_rows]
        axes[0].plot(valid_updates, valid_loss, "o-", color="#c0504d",
                     markersize=2.6, linewidth=1.0, label="validation loss")
        best_index = min(range(len(valid_loss)), key=valid_loss.__getitem__)
        axes[0].scatter(
            [valid_updates[best_index]], [valid_loss[best_index]],
            color="#b2182b", s=28, zorder=4,
            label=f"best validation ({valid_loss[best_index]:.4f})",
        )
    axes[0].set_ylabel("loss")
    axes[0].set_title("PocketDiff full-data training")
    axes[0].legend(loc="upper right", fontsize=7)
    axes[0].grid(axis="y", color="#d9d9d9", linewidth=0.5)

    if valid_rows:
        axes[1].plot(
            [int(row["update"]) for row in valid_rows],
            [float(row["validation_loss"]) for row in valid_rows],
            "o-", color="#c0504d", markersize=2.4, linewidth=1.0,
        )
        axes[1].set_ylabel("validation")
    axes[1].set_xlabel("update")
    axes[1].grid(axis="y", color="#d9d9d9", linewidth=0.5)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stem = args.output.with_suffix("")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "updates": len(rows),
        "first_update": updates[0],
        "last_update": updates[-1],
        "first_train_loss": train_loss[0],
        "last_train_loss": train_loss[-1],
        "validation_points": len(valid_rows),
        "best_validation_loss": (
            min(float(row["validation_loss"]) for row in valid_rows)
            if valid_rows else None
        ),
    }
    stem.with_name(stem.name + "_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
