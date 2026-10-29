"""Fixed clean endpoint joint rigid/χ training comparison."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from pocketdiff.data.apo2mol_adapter import Apo2MolAdapter
from pocketdiff.models import PocketDiffModel
from pocketdiff.training import (
    JointTrainRecord,
    collate_clean_examples,
    evaluate_joint_clean,
    make_clean_example,
    masked_joint_clean_loss,
    save_joint_checkpoint,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(loss):
    return {
        "loss": float(loss.loss),
        "motion_loss": float(loss.motion_loss),
        "chi_loss": float(loss.chi_loss),
        "translation_loss": float(loss.translation_loss),
        "rotation_loss": float(loss.rotation_loss),
        "valid_residue_count": loss.valid_residue_count,
        "valid_chi_count": loss.valid_chi_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(".codex-tasks/pocketdiff-development")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "phase6b-clean-generalization/raw/manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    if args.limit < 2 or args.limit % 2 != 0:
        parser.error("limit must be a positive even number >= 2")
    if args.steps <= 0:
        parser.error("steps must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir()

    manifest = json.loads(args.manifest.read_text())
    entries = manifest["entries"][:args.limit]
    adapter = Apo2MolAdapter(".")
    examples = []
    for entry in entries:
        source = entry["source"]
        value, _diagnostics = adapter.convert_paths(
            source["holo_pocket"]["path"],
            source["apo_pocket"]["path"],
            source["ligand"]["path"],
            sample_id=entry["sample_id"],
        )
        examples.append(make_clean_example(value))
    midpoint = args.limit // 2
    train_batch = collate_clean_examples(examples[:midpoint])
    holdout_batch = collate_clean_examples(examples[midpoint:])

    seed = 17
    learning_rate = 1.0e-3
    chi_weight = 1.0
    max_grad_norm = 10.0
    torch.manual_seed(seed)
    model_config = {
        "encoder_layers": 1,
        "knn": 8,
        "sigma_translation": 1.0,
        "predict_chi": True,
    }
    model = PocketDiffModel(**model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    initial_train = evaluate_joint_clean(model, train_batch, chi_weight=chi_weight)
    initial_holdout = evaluate_joint_clean(model, holdout_batch, chi_weight=chi_weight)
    checkpoints = {}
    eval_steps = {0, 50, 100, 150, args.steps}
    history = []

    def record(step, train_loss, holdout_loss, gradient_norm):
        row = {
            "step": step,
            "train": _metrics(train_loss),
            "holdout": _metrics(holdout_loss),
            "gradient_norm_before_clip": gradient_norm,
        }
        history.append(row)
        if step == 0:
            final_record = JointTrainRecord(
                step=0,
                loss=row["train"]["loss"],
                motion_loss=row["train"]["motion_loss"],
                chi_loss=row["train"]["chi_loss"],
                translation_loss=row["train"]["translation_loss"],
                rotation_loss=row["train"]["rotation_loss"],
                valid_residue_count=row["train"]["valid_residue_count"],
                valid_chi_count=row["train"]["valid_chi_count"],
                gradient_norm_before_clip=0.0,
            )
        else:
            final_record = JointTrainRecord(
                step=step,
                loss=row["train"]["loss"],
                motion_loss=row["train"]["motion_loss"],
                chi_loss=row["train"]["chi_loss"],
                translation_loss=row["train"]["translation_loss"],
                rotation_loss=row["train"]["rotation_loss"],
                valid_residue_count=row["train"]["valid_residue_count"],
                valid_chi_count=row["train"]["valid_chi_count"],
                gradient_norm_before_clip=float(gradient_norm),
            )
        path = checkpoint_dir / ("checkpoint_%04d.pt" % step)
        save_joint_checkpoint(
            path,
            model,
            config=model_config,
            sample_ids=list(train_batch.sample_ids),
            final_record=final_record,
        )
        checkpoints[str(step)] = str(path)

    record(0, initial_train, initial_holdout, 0.0)
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(**train_batch.model_kwargs())
        loss = masked_joint_clean_loss(prediction, train_batch, chi_weight=chi_weight)
        loss.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm, error_if_nonfinite=True
        )
        optimizer.step()
        if step in eval_steps:
            train_eval = evaluate_joint_clean(model, train_batch, chi_weight=chi_weight)
            holdout_eval = evaluate_joint_clean(model, holdout_batch, chi_weight=chi_weight)
            record(step, train_eval, holdout_eval, float(gradient_norm))

    final = history[-1]
    final_checkpoint = Path(checkpoints[str(args.steps)])
    payload = torch.load(final_checkpoint, map_location="cpu")
    restored = PocketDiffModel(**payload["model_config"])
    restored.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    restored.eval()
    with torch.no_grad():
        original_prediction = model(**train_batch.model_kwargs())
        restored_prediction = restored(**train_batch.model_kwargs())
    reload_error = max(
        float((original_prediction.remaining_translation_local - restored_prediction.remaining_translation_local).abs().max()),
        float((original_prediction.remaining_rotvec_local - restored_prediction.remaining_rotvec_local).abs().max()),
        float((original_prediction.remaining_chi - restored_prediction.remaining_chi).abs().max()),
    )
    report = {
        "passed": bool(
            final["train"]["loss"] < history[0]["train"]["loss"]
            and all(
                value["loss"] == value["loss"]
                and value["chi_loss"] == value["chi_loss"]
                for value in (final["train"], final["holdout"])
            )
        ),
        "learning_goal_met": False,
        "mode": "phase29_joint_clean_endpoint_training",
        "seed": seed,
        "steps": args.steps,
        "model_config": model_config,
        "learning_rate": learning_rate,
        "chi_weight": chi_weight,
        "max_grad_norm": max_grad_norm,
        "train_sample_ids": list(train_batch.sample_ids),
        "holdout_sample_ids": list(holdout_batch.sample_ids),
        "train_valid_chi_count": int(train_batch.chi_mask.sum()),
        "holdout_valid_chi_count": int(holdout_batch.chi_mask.sum()),
        "checkpoint_reload_max_error": reload_error,
        "history": history,
        "checkpoints": checkpoints,
        "source_manifest_sha256": _sha(args.manifest),
        "scope": "clean endpoint teacher-forced training only; no autonomous rollout or side-chain coordinate update",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "passed": report["passed"],
        "learning_goal_met": report["learning_goal_met"],
        "initial_train_loss": history[0]["train"]["loss"],
        "final_train_loss": final["train"]["loss"],
        "initial_holdout_loss": history[0]["holdout"]["loss"],
        "final_holdout_loss": final["holdout"]["loss"],
        "train_valid_chi_count": report["train_valid_chi_count"],
        "holdout_valid_chi_count": report["holdout_valid_chi_count"],
        "checkpoint_reload_max_error": report["checkpoint_reload_max_error"],
    }), flush=True)
    if not report["passed"]:
        raise RuntimeError("Phase 29 joint clean training did not reduce train loss")


if __name__ == "__main__":
    main()
