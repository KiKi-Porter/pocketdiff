from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def _mean(values):
    return sum(values) / max(1, len(values))


def _paired_summary(v4_rows, v3_rows, key):
    v4 = {row["sample_id"]: row for row in v4_rows}
    v3 = {row["sample_id"]: row for row in v3_rows}
    if set(v4) != set(v3):
        raise ValueError("v3 and v4 sample identities do not match")
    pairs = sorted(v4)
    difference = [v4[sample][key] - v3[sample][key] for sample in pairs]
    rng = random.Random(20260923)
    bootstrap = []
    for _ in range(5000):
        bootstrap.append(
            _mean([difference[rng.randrange(len(difference))] for _ in difference])
        )
    bootstrap.sort()
    return {
        "count": len(pairs),
        "mean_v4_minus_v3_angstrom": _mean(difference),
        "median_v4_minus_v3_angstrom": sorted(difference)[len(difference) // 2],
        "fraction_v4_lower": sum(value < 0 for value in difference) / len(difference),
        "bootstrap_95pct_ci_angstrom": [
            bootstrap[int(0.025 * len(bootstrap))],
            bootstrap[min(len(bootstrap) - 1, int(0.975 * len(bootstrap)))],
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v4", required=True)
    parser.add_argument("--v3", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.v4).open("r") as stream:
        v4 = json.load(stream)
    with Path(args.v3).open("r") as stream:
        v3 = json.load(stream)
    result = {
        "format": "pocketdiff-v4-paired-comparison",
        "v4_checkpoint": v4["checkpoint"],
        "v3_checkpoint": v3["checkpoint"],
        "protocols": {
            "v4": {
                "steps": v4["steps"],
                "motion_scale": v4.get("motion_scale", 1.0),
                "initial_noise_scale": v4.get("initial_noise_scale", 1.0),
                "initialization": v4["initialization"],
                "holo_usage": v4["holo_usage"],
            },
            "v3": {
                "steps": v3["sampling_steps"],
                "initial_state": v3["initial_state"],
                "seed_mode": v3["seed_mode"],
            },
        },
        "paired": {},
    }
    for split in ("train", "valid", "test"):
        v4_rows, v3_rows = v4["rows"][split], v3["rows"][split]
        result["paired"][split] = {
            "all_protein_rmsd": _paired_summary(
                v4_rows, v3_rows, "sample_holo_rmsd"
            ),
            "pocket_rmsd": _paired_summary(
                v4_rows, v3_rows, "sample_holo_pocket_rmsd"
            ),
            "v4_apo_comparison": {
                "all_protein_improvement_mean_angstrom": v4["metrics"][split][
                    "improvement_mean"
                ],
                "all_protein_improved_fraction": v4["metrics"][split][
                    "improved_fraction"
                ],
                "pocket_improvement_mean_angstrom": v4["metrics"][split][
                    "pocket_improvement_mean"
                ],
                "pocket_improved_fraction": v4["metrics"][split][
                    "pocket_improved_fraction"
                ],
            },
        }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["paired"], indent=2), flush=True)


if __name__ == "__main__":
    main()
