from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

import utils.misc as misc
from datasets.md_dataset import PreparedMDDataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D


FEATURE_DIM_PROTEIN = 27
FEATURE_DIM_LIGAND = 13
COLLATE_EXCLUDE_KEYS = ["ligand_nbh_list"]


def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _is_main_process() -> bool:
    return _rank() == 0


class _DiffusionLossWrapper(nn.Module):
    def __init__(self, model: ScorePosNet3D):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        return self.model.get_diffusion_loss(**kwargs)


def _unwrap_model(model):
    if isinstance(model, DistributedDataParallel):
        model = model.module
    if isinstance(model, _DiffusionLossWrapper):
        return model.model
    return model


def _get_diffusion_loss(model, **kwargs):
    if isinstance(model, DistributedDataParallel):
        return model(**kwargs)
    return model.get_diffusion_loss(**kwargs)


def _load_yaml(path: Path):
    with path.open() as handle:
        return misc.load_config(str(path))


def _load_model(
    config,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ScorePosNet3D, dict]:
    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=FEATURE_DIM_PROTEIN,
        ligand_atom_feature_dim=FEATURE_DIM_LIGAND,
    )
    payload = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Checkpoint has no model state: {checkpoint_path}")
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    allowed_prefixes = (
        "refine_net.aplc.",
        "refine_net.internal_cross.",
        "conservative_adapter.",
    )
    invalid_missing = [
        key
        for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_prefixes)
    ]
    if unexpected or invalid_missing:
        raise RuntimeError(
            "Checkpoint compatibility failure: "
            f"unexpected={unexpected}, invalid_missing={invalid_missing}"
        )
    model.to(device)
    return model, payload


def _set_trainable_stage(model: ScorePosNet3D, stage: str, tail_layers: int) -> list[str]:
    model = _unwrap_model(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    has_aplc = getattr(model.refine_net, "aplc", None) is not None
    has_adapter = getattr(model, "conservative_adapter", None) is not None
    has_internal_cross = (
        getattr(model.refine_net, "internal_cross", None) is not None
    )
    if stage in ("aplc", "partial") and not has_aplc:
        raise RuntimeError("APLC must be enabled for APLC fine-tuning")
    if stage == "adapter" and not has_adapter:
        raise RuntimeError(
            "Conservative adapter must be enabled for adapter fine-tuning"
        )
    if stage == "internal_adapter" and not (
        has_adapter and has_internal_cross
    ):
        raise RuntimeError(
            "Both Conservative Adapter and internal cross branch must be enabled"
        )
    if has_aplc and stage in ("aplc", "partial"):
        for parameter in model.refine_net.aplc.parameters():
            parameter.requires_grad_(True)

    if stage == "partial":
        if tail_layers < 1 or tail_layers > len(model.refine_net.base_block):
            raise ValueError(
                f"tail_layers={tail_layers} is invalid for "
                f"{len(model.refine_net.base_block)} transformer layers"
            )
        for layer in model.refine_net.base_block[-tail_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
        for parameter in model.v_inference.parameters():
            parameter.requires_grad_(True)
    elif stage == "adapter":
        for parameter in model.conservative_adapter.parameters():
            parameter.requires_grad_(True)
    elif stage == "internal_adapter":
        for parameter in model.conservative_adapter.parameters():
            parameter.requires_grad_(True)
        for parameter in model.refine_net.internal_cross.parameters():
            parameter.requires_grad_(True)
    elif stage != "aplc":
        raise ValueError(f"Unknown stage: {stage}")

    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def _make_optimizer(model: ScorePosNet3D, lr: float, weight_decay: float):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable parameters in current stage")
    return torch.optim.Adam(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.95, 0.999),
    )


def _infinite_loader(loader):
    epoch = 0
    while True:
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def _subset_dataset(dataset, limit: int | None):
    if limit is None or limit >= len(dataset):
        return dataset
    return torch.utils.data.Subset(dataset, list(range(limit)))


def _make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler=None,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=COLLATE_EXCLUDE_KEYS,
        pin_memory=torch.cuda.is_available(),
    )


def _set_rng_seed(seed: int) -> None:
    misc.seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _adapter_regularization(result, config):
    anchor_pos_weight = float(
        getattr(config.train, "adapter_anchor_pos_weight", 0.25)
    )
    anchor_v_weight = float(
        getattr(config.train, "adapter_anchor_v_weight", 0.10)
    )
    delta_weight = float(
        getattr(config.train, "adapter_delta_weight", 0.01)
    )
    anchor_pos = (
        result["pred_ligand_pos"] - result["base_pred_ligand_pos"]
    ).pow(2).sum(dim=-1).mean()
    student_log_v = F.log_softmax(result["pred_ligand_v"], dim=-1)
    base_prob_v = F.softmax(
        result["base_pred_ligand_v"].detach(), dim=-1
    )
    anchor_v = F.kl_div(
        student_log_v,
        base_prob_v,
        reduction="batchmean",
    ).clamp_min(0.0)
    delta_regularizer = (
        result["adapter_delta_pos"].pow(2).mean()
        + result["adapter_delta_v"].pow(2).mean()
    )
    anchor_loss = (
        anchor_pos_weight * anchor_pos
        + anchor_v_weight * anchor_v
        + delta_weight * delta_regularizer
    )
    return anchor_loss, anchor_pos, anchor_v, delta_regularizer


@torch.no_grad()
def _validate(
    model: ScorePosNet3D,
    loader,
    device: torch.device,
    time_steps: Iterable[int],
    max_batches: int | None,
    stage: str = "partial",
    config=None,
) -> dict[str, float]:
    model.eval()
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    # Fixed validation randomness makes checkpoint comparisons reproducible.
    _set_rng_seed(1729)

    totals = {
        "loss": 0.0,
        "loss_pos": 0.0,
        "loss_v": 0.0,
        "objective": 0.0,
        "anchor_loss": 0.0,
        "graphs": 0,
    }
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = batch.to(device)
        batch_size = int(batch.num_graphs)
        for time_value in time_steps:
            time_step = torch.full(
                (batch_size,),
                int(time_value),
                dtype=torch.long,
                device=device,
            )
            result = _get_diffusion_loss(
                model,
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,
                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch,
                time_step=time_step,
            )
            totals["loss"] += float(result["loss"]) * batch_size
            totals["loss_pos"] += float(result["loss_pos"]) * batch_size
            totals["loss_v"] += float(result["loss_v"]) * batch_size
            objective = result["loss"]
            anchor_loss = result["loss"].new_zeros(())
            if stage in ("adapter", "internal_adapter"):
                if config is None:
                    raise ValueError("config is required for adapter validation")
                (
                    anchor_loss,
                    _anchor_pos,
                    _anchor_v,
                    _delta_regularizer,
                ) = _adapter_regularization(result, config)
                objective = objective + anchor_loss
            totals["objective"] += float(objective) * batch_size
            totals["anchor_loss"] += float(anchor_loss) * batch_size
            totals["graphs"] += batch_size

    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    torch.random.set_rng_state(cpu_rng)

    if _is_distributed():
        values = torch.tensor(
            [
                totals["loss"],
                totals["loss_pos"],
                totals["loss_v"],
                totals["objective"],
                totals["anchor_loss"],
                float(totals["graphs"]),
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        (
            totals["loss"],
            totals["loss_pos"],
            totals["loss_v"],
            totals["objective"],
            totals["anchor_loss"],
            totals["graphs"],
        ) = [float(value) for value in values.tolist()]

    if totals["graphs"] == 0:
        raise RuntimeError("Validation loader produced no graphs")
    return {
        "loss": totals["loss"] / totals["graphs"],
        "loss_pos": totals["loss_pos"] / totals["graphs"],
        "loss_v": totals["loss_v"] / totals["graphs"],
        "objective": totals["objective"] / totals["graphs"],
        "anchor_loss": totals["anchor_loss"] / totals["graphs"],
    }


def _save_checkpoint(
    path: Path,
    model: ScorePosNet3D,
    optimizer,
    stage: str,
    iteration: int,
    metrics: dict[str, float],
    config,
    source_checkpoint: Path,
    trainable_names: list[str],
) -> None:
    if not _is_main_process():
        return
    model = _unwrap_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stage": stage,
            "iteration": iteration,
            "metrics": metrics,
            "source_checkpoint": str(source_checkpoint),
            "trainable_parameter_names": trainable_names,
            "feature_contract": {
                "protein_atom_feature_dim": FEATURE_DIM_PROTEIN,
                "ligand_atom_feature_dim": FEATURE_DIM_LIGAND,
                "cache": str(config.data.cache),
            },
        },
        path,
    )


def _run_stage(
    model: ScorePosNet3D,
    stage: str,
    stage_config,
    train_loader,
    valid_loader,
    device: torch.device,
    output_dir: Path,
    config,
    source_checkpoint: Path,
    max_val_batches: int | None,
) -> dict:
    trainable_names = _set_trainable_stage(
        model,
        stage=stage,
        tail_layers=int(getattr(stage_config, "tail_layers", 2)),
    )
    optimizer = _make_optimizer(
        model,
        lr=float(stage_config.lr),
        weight_decay=float(stage_config.weight_decay),
    )
    train_iterator = _infinite_loader(train_loader)
    max_iters = int(stage_config.max_iters)
    val_every = int(config.train.val_every)
    time_steps = [int(value) for value in config.train.val_time_steps]

    history_path = output_dir / f"{stage}_history.jsonl"
    best_path = output_dir / f"{stage}_best.pt"
    best_loss = math.inf
    best_metrics = None
    model.train()

    history_handle = history_path.open("w") if _is_main_process() else None
    try:
        for iteration in range(1, max_iters + 1):
            batch = next(train_iterator).to(device)
            optimizer.zero_grad(set_to_none=True)
            protein_noise_std = float(config.train.pos_noise_std)
            if protein_noise_std:
                protein_pos = batch.protein_pos + torch.randn_like(
                    batch.protein_pos
                ) * protein_noise_std
            else:
                protein_pos = batch.protein_pos
            result = _get_diffusion_loss(
                model,
                protein_pos=protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,
                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch,
            )
            loss = result["loss"]
            if stage in ("adapter", "internal_adapter"):
                anchor_loss, anchor_pos, anchor_v, delta_regularizer = (
                    _adapter_regularization(result, config)
                )
                loss = loss + anchor_loss
                result["anchor_loss"] = anchor_loss
                result["anchor_pos"] = anchor_pos
                result["anchor_v"] = anchor_v
                result["delta_regularizer"] = delta_regularizer
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at {stage} iteration {iteration}"
                )
            loss.backward()
            grad_norm = clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(config.train.max_grad_norm),
            )
            optimizer.step()

            record = {
                "stage": stage,
                "iteration": iteration,
                "train_loss": float(loss.detach()),
                "train_loss_pos": float(result["loss_pos"].detach()),
                "train_loss_v": float(result["loss_v"].detach()),
                "grad_norm": float(grad_norm),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            if stage == "adapter":
                record.update({
                    "train_anchor_loss": float(result["anchor_loss"].detach()),
                    "train_anchor_pos": float(result["anchor_pos"].detach()),
                    "train_anchor_v": float(result["anchor_v"].detach()),
                    "train_delta_regularizer": float(
                        result["delta_regularizer"].detach()
                    ),
                })

            if iteration % val_every == 0 or iteration == max_iters:
                metrics = _validate(
                    model=model,
                    loader=valid_loader,
                    device=device,
                    time_steps=time_steps,
                    max_batches=max_val_batches,
                    stage=stage,
                    config=config,
                )
                record.update({f"valid_{key}": value for key, value in metrics.items()})
                selection_loss = metrics["objective"]
                if selection_loss < best_loss:
                    best_loss = selection_loss
                    best_metrics = dict(metrics)
                    _save_checkpoint(
                        best_path,
                        model=model,
                        optimizer=optimizer,
                        stage=stage,
                        iteration=iteration,
                        metrics=metrics,
                        config=config,
                        source_checkpoint=source_checkpoint,
                        trainable_names=trainable_names,
                    )
                    if _is_main_process():
                        shutil.copyfile(best_path, output_dir / "best.pt")

            if _is_main_process():
                history_handle.write(json.dumps(record) + "\n")
                history_handle.flush()
            if _is_main_process() and (
                iteration == 1 or iteration % max(1, val_every) == 0
            ):
                print(json.dumps(record), flush=True)
    finally:
        if history_handle is not None:
            history_handle.close()

    if best_metrics is None:
        raise RuntimeError(f"Stage {stage} did not run validation")
    return {
        "stage": stage,
        "best_loss": best_loss,
        "best_metrics": best_metrics,
        "best_checkpoint": str(best_path),
        "trainable_tensor_count": len(trainable_names),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/md_dynamic.yml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("pretrained_models/pretrained_diffusion.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/md_dynamic"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument(
        "--only-stage",
        choices=["aplc", "partial", "adapter", "internal_adapter"],
        default=None,
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--stage1-iters", type=int, default=None)
    parser.add_argument("--stage2-iters", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Set stage iterations to epochs * local train-loader length.",
    )
    args = parser.parse_args()

    config = _load_yaml(args.config)
    if args.num_workers is not None:
        config.train.num_workers = int(args.num_workers)
    _set_rng_seed(int(config.train.seed))
    distributed = _is_distributed() or args.distributed
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        device = torch.device(args.device)
    if _is_main_process():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.config, args.output_dir / args.config.name)
    if distributed:
        dist.barrier()

    train_dataset = PreparedMDDataset(config.data.cache, split=config.data.train_split)
    valid_dataset = PreparedMDDataset(config.data.cache, split=config.data.valid_split)
    train_dataset = _subset_dataset(train_dataset, args.max_train_samples)
    valid_dataset = _subset_dataset(valid_dataset, args.max_valid_samples)
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True)
        if distributed
        else None
    )
    valid_sampler = (
        DistributedSampler(valid_dataset, shuffle=False)
        if distributed
        else None
    )
    train_loader = _make_loader(
        train_dataset,
        batch_size=int(config.train.batch_size),
        shuffle=train_sampler is None,
        num_workers=int(config.train.num_workers),
        sampler=train_sampler,
    )
    valid_loader = _make_loader(
        valid_dataset,
        batch_size=int(config.train.batch_size),
        shuffle=False,
        num_workers=int(config.train.num_workers),
        sampler=valid_sampler,
    )
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise ValueError("Training and validation datasets must be non-empty")

    model, source_payload = _load_model(config, args.checkpoint, device)
    if distributed:
        model = DistributedDataParallel(
            _DiffusionLossWrapper(model),
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    if _is_main_process():
        print(
            json.dumps(
                {
                    "device": str(device),
                    "distributed": distributed,
                    "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                    "train_samples": len(train_dataset),
                    "valid_samples": len(valid_dataset),
                    "train_loader_batches_per_rank": len(train_loader),
                    "valid_loader_batches_per_rank": len(valid_loader),
                    "trainable_parameters_initial": misc.count_parameters(
                        _unwrap_model(model)
                    ),
                    "source_iteration": source_payload.get("iteration"),
                }
            ),
            flush=True,
        )

    stage_summaries = []
    stage1_value = getattr(config.train, "stage1", None)
    if stage1_value is None:
        stage1_value = getattr(config.train, "adapter", None)
    if stage1_value is None:
        stage1_value = getattr(config.train, "internal_adapter", None)
    if stage1_value is None:
        stage1_value = getattr(config.train, "stage2", None)
    if stage1_value is None:
        raise AttributeError("Training config has no stage configuration")
    stage1_config = copy.deepcopy(stage1_value)
    stage2_value = getattr(config.train, "stage2", None)
    if stage2_value is None:
        stage2_value = getattr(config.train, "adapter", None)
    if stage2_value is None:
        stage2_value = getattr(config.train, "internal_adapter", None)
    if stage2_value is None:
        stage2_value = stage1_value
    stage2_config = copy.deepcopy(stage2_value)
    if args.stage1_iters is not None:
        stage1_config.max_iters = args.stage1_iters
    if args.stage2_iters is not None:
        stage2_config.max_iters = args.stage2_iters

    if args.only_stage in (None, "aplc"):
        stage1_summary = _run_stage(
            model,
            "aplc",
            stage1_config,
            train_loader,
            valid_loader,
            device,
            args.output_dir,
            config,
            args.checkpoint,
            args.max_val_batches,
        )
        stage_summaries.append(stage1_summary)
        if args.only_stage is None:
            best_payload = torch.load(
                stage1_summary["best_checkpoint"],
                map_location="cpu",
                weights_only=False,
            )
            _unwrap_model(model).load_state_dict(
                best_payload["model"],
                strict=True,
            )

    if args.only_stage in (None, "partial", "adapter", "internal_adapter"):
        if args.only_stage in ("adapter", "internal_adapter"):
            stage_name = args.only_stage
        else:
            stage_name = "partial"
        stage_config = (
            copy.deepcopy(getattr(config.train, stage_name))
            if stage_name in ("adapter", "internal_adapter")
            else stage2_config
        )
        if args.stage2_iters is not None:
            stage_config.max_iters = args.stage2_iters
        if args.epochs is not None:
            stage_config.max_iters = len(train_loader) * int(args.epochs)
        stage_summaries.append(
            _run_stage(
                model,
                stage_name,
                stage_config,
                train_loader,
                valid_loader,
                device,
                args.output_dir,
                config,
                args.checkpoint,
                args.max_val_batches,
            )
        )

    summary = {
        "config": str(args.config),
        "source_checkpoint": str(args.checkpoint),
        "cache": str(config.data.cache),
        "stages": stage_summaries,
    }
    if _is_main_process():
        (args.output_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
