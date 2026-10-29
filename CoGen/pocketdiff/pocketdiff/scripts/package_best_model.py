"""Package and validate a PocketDiff checkpoint as an engineering release."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.training.diffusion import (
    DEFAULT_DATA_ROOT,
    DEFAULT_SPLIT_PATH,
    load_diffusion_checkpoint,
    sample_checkpoint_on_complex,
)


RELEASE_FORMAT = "pocketdiff-model-release-v1"
RELEASE_NAME = "phase71-best-engineering-candidate"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _endpoint(report: Mapping[str, Any], split: str) -> Dict[str, float]:
    section = report.get(split)
    if not isinstance(section, Mapping):
        raise ValueError(f"source report has no {split!r} autonomous section")
    apo = float(section["mean_apo_rmsd"])
    final = float(section["mean_final_holo_rmsd"])
    return {
        "apo_rmsd": apo,
        "final_holo_rmsd": final,
        "delta_final_minus_apo": final - apo,
        "ratio_final_over_apo": final / apo if apo else float("inf"),
    }


def _selection_score(report: Mapping[str, Any]) -> float:
    valid = _endpoint(report, "valid")
    test = _endpoint(report, "test")
    return 0.5 * (
        valid["ratio_final_over_apo"] + test["ratio_final_over_apo"]
    )


def _load_source_report(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    with path.open() as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError("source report must contain a JSON object")
    return report


def _validate_copy(
    checkpoint: Path,
    *,
    data_root: Path,
    split_path: Path,
    split: str,
    index: int,
    steps: int,
) -> Dict[str, Any]:
    loaded = load_diffusion_checkpoint(checkpoint, map_location="cpu")
    payload = loaded.payload
    model_config = payload.get("model_config", {})
    if not isinstance(model_config, Mapping):
        raise ValueError("checkpoint model_config must be a mapping")
    with split_path.open("rb") as handle:
        import pickle

        split_data = pickle.load(handle)
    if split not in split_data:
        raise KeyError(f"split {split!r} not found in {split_path}")
    records = split_data[split]
    if index < 0 or index >= len(records):
        raise IndexError(f"index {index} is outside split {split!r}")
    value = Apo2MolAdapter(data_root).convert_record(records[index])
    trajectory = sample_checkpoint_on_complex(checkpoint, value, steps=steps)
    start_error = float((trajectory.states[0] - value.protein_pos_apo).abs().max())
    finite = all(bool(torch.isfinite(state).all()) for state in trajectory.states)
    return {
        "checkpoint_loadable_on_cpu": True,
        "model_config": _json_value(dict(model_config)),
        "sample_id": value.sample_id,
        "split": split,
        "index": index,
        "steps": steps,
        "state_count": len(trajectory.states),
        "start_matches_apo": bool(start_error <= 1.0e-6),
        "start_max_abs_error": start_error,
        "trajectory_finite": finite,
        "times": [float(item) for item in trajectory.times],
    }


def _write_model_card(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    checkpoint_path: Path,
) -> None:
    metrics = manifest["autonomous_metrics"]
    validation = manifest["validation"]
    lines = [
        "# PocketDiff Phase 71 Model Release",
        "",
        "This directory contains the current best engineering candidate under the",
        "Phase 70 CUDA v3 comparison protocol.",
        "",
        "## Checkpoint",
        "",
        f"- Release: `{manifest['release_name']}`",
        f"- File: `{manifest['checkpoint']}`",
        f"- SHA256: `{manifest['checkpoint_sha256']}`",
        f"- Backend: `{manifest['model_config']['encoder_backend']}`",
        f"- Hidden dimension: `{manifest['model_config']['hidden_dim']}`",
        "",
        "## Autonomous endpoint metrics (Angstrom)",
        "",
        "| Split | Apo | Final holo | Final - apo |",
        "| --- | ---: | ---: | ---: |",
    ]
    for split in ("train", "valid", "test"):
        item = metrics[split]
        lines.append(
            f"| {split} | {item['apo_rmsd']:.6f} | "
            f"{item['final_holo_rmsd']:.6f} | {item['delta_final_minus_apo']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- CPU checkpoint load: `{validation['checkpoint_loadable_on_cpu']}`",
            f"- Apo-start match: `{validation['start_matches_apo']}`",
            f"- Finite rollout: `{validation['trajectory_finite']}`",
            f"- Smoke sample: `{validation['sample_id']}`",
            f"- Smoke steps: `{validation['steps']}`",
            f"- CUDA reload close: `{manifest['cuda_reload_validation'].get('close_at_2e-6')}`",
            f"- CUDA max repeat delta: "
            f"`{manifest['cuda_reload_validation'].get('max_repeat_delta')}`",
            "",
            "## Scientific boundary",
            "",
            f"- `learning_goal_met`: `{manifest['scientific_status']['learning_goal_met']}`",
            "- The valid and test endpoints still regress versus their apo baselines.",
            "- Do not describe this artifact as a reliable generalized continuous",
            "  denoising field or as a solved apo-to-holo predictor.",
            "",
            "## Inference",
            "",
            "```bash",
            "PYTHONPATH=. conda run -n targetdiff python -m "
            "pocketdiff.scripts.sample_diffusion \\",
            f"  --checkpoint {checkpoint_path} \\",
            "  --output /tmp/pocketdiff_sample.json \\",
            "  --split valid --index 0 --steps 8",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def package_release(
    source_checkpoint: Path,
    source_report: Path,
    output_dir: Path,
    *,
    reload_validation: Optional[Path],
    data_root: Path,
    split_path: Path,
    split: str,
    index: int,
    steps: int,
) -> Dict[str, Any]:
    source_checkpoint = source_checkpoint.resolve()
    source_report = source_report.resolve()
    output_dir = output_dir.resolve()
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    if not source_report.is_file():
        raise FileNotFoundError(source_report)
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"release output directory is not empty: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True)

    report = _load_source_report(source_report)
    checkpoint = output_dir / "checkpoint.pt"
    shutil.copy2(source_checkpoint, checkpoint)
    copied_report = output_dir / "source_report.json"
    shutil.copy2(source_report, copied_report)
    reload_entry: Dict[str, Any] = {}
    if reload_validation is not None:
        with reload_validation.open() as handle:
            reload_all = json.load(handle)
        if not isinstance(reload_all, Mapping):
            raise ValueError("reload validation must contain a JSON object")
        reload_entry = dict(reload_all.get(source_checkpoint.parent.name, {}))
        if not reload_entry:
            raise KeyError(
                "reload validation has no entry for "
                f"{source_checkpoint.parent.name!r}"
            )
        shutil.copy2(reload_validation, output_dir / "reload_validation.json")
    validation = _validate_copy(
        checkpoint,
        data_root=data_root,
        split_path=split_path,
        split=split,
        index=index,
        steps=steps,
    )
    config = report.get("config", {})
    if not isinstance(config, Mapping):
        config = {}
    model_config = validation["model_config"]
    metrics = {
        split_name: _endpoint(report, split_name)
        for split_name in ("train", "valid", "test")
    }
    manifest: Dict[str, Any] = {
        "format": RELEASE_FORMAT,
        "release_name": RELEASE_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": "checkpoint.pt",
        "checkpoint_sha256": _sha256(checkpoint),
        "release_dir": str(output_dir),
        "source_checkpoint": str(source_checkpoint),
        "source_report": "source_report.json",
        "model_config": model_config,
        "training_config": _json_value(dict(config)),
        "autonomous_metrics": metrics,
        "selection_score_valid_test_ratio": _selection_score(report),
        "validation": validation,
        "engineering_validation": {
            "release_ready": bool(
                validation["checkpoint_loadable_on_cpu"]
                and validation["start_matches_apo"]
                and validation["trajectory_finite"]
                and (
                    not reload_entry
                    or bool(reload_entry.get("close_at_2e-6", False))
                )
            ),
            "cpu_checkpoint_load": bool(
                validation["checkpoint_loadable_on_cpu"]
            ),
            "apo_start_contract": bool(validation["start_matches_apo"]),
            "finite_rollout": bool(validation["trajectory_finite"]),
            "cuda_reload_close": (
                None
                if not reload_entry
                else bool(reload_entry.get("close_at_2e-6", False))
            ),
        },
        "scientific_status": {
            "learning_goal_met": bool(report.get("learning_goal_met", False)),
            "release_class": "engineering_release_candidate",
            "teacher_forced_loss_is_not_success_criterion": True,
            "valid_and_test_regress_against_apo": bool(
                metrics["valid"]["delta_final_minus_apo"] > 0.0
                and metrics["test"]["delta_final_minus_apo"] > 0.0
            ),
        },
        "source_report_status": {
            "legacy_report_passed": bool(report.get("passed", False)),
            "legacy_checkpoint_reload_exact": report.get("checkpoint_reload_exact"),
            "legacy_checkpoint_reload_close": report.get("checkpoint_reload_close"),
            "legacy_checkpoint_reload_tolerance": report.get(
                "checkpoint_reload_tolerance"
            ),
            "effective_cuda_reload_close": reload_entry.get("close_at_2e-6"),
        },
        "cuda_reload_validation": reload_entry,
    }
    manifest_path = output_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write_model_card(
        output_dir / "MODEL_CARD.md",
        manifest=manifest,
        checkpoint_path=output_dir / "checkpoint.pt",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--reload-validation", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = package_release(
        args.source_checkpoint,
        args.source_report,
        args.output_dir,
        reload_validation=args.reload_validation,
        data_root=args.data_root,
        split_path=args.split_path,
        split=args.split,
        index=args.index,
        steps=args.steps,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
