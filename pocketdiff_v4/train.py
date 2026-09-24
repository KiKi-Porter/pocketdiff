from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .batching import collate_complexes
from .cache import load_cache
from .constants import NUM_CHI
from .geometry import apply_motion, residue_frames
from .model import PocketDiffV4Model
from pocketdiff.geometry.bridge import remaining_transform_current_to_holo


class ComplexDataset(Dataset):
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


def _build_train_loader(
    dataset,
    batch_size: int,
    distributed: bool,
    rank: int,
    world: int,
    num_workers: int,
    pin_memory: bool,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    sampler = None
    if distributed:
        if world <= 0 or len(dataset) % world:
            raise ValueError(
                "distributed training requires dataset size divisible by world size"
            )
        sampler = DistributedSampler(
            dataset,
            num_replicas=world,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_complexes,
    )
    return sampler, loader


def _build_eval_loader(dataset, batch_size: int, distributed: bool, rank: int, world: int, device):
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_complexes,
    )
    return sampler, loader


def _rank_info():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    distributed = world > 1
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    return distributed, rank, local_rank, world, device


def _move_batch(batch, device):
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device=device, non_blocking=True)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return value
        return value
    return move(batch)


def _wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _project_vector_norm(vector: torch.Tensor, maximum: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    limit = maximum * 0.95
    scale = torch.clamp(limit / norm.clamp_min(1e-8), max=1.0)
    return vector * scale


def _graph_balanced_mean(
    values: torch.Tensor,
    graph_index: torch.Tensor,
    graph_count: int,
    mask: torch.Tensor = None,
    weights: torch.Tensor = None,
) -> torch.Tensor:
    if values.ndim != 1 or graph_index.shape != values.shape:
        raise ValueError("values and graph_index must be matching vectors")
    active = torch.ones_like(values) if mask is None else mask.to(values.dtype)
    weight = active if weights is None else active * weights.to(values.dtype)
    sums = values.new_zeros(graph_count)
    counts = values.new_zeros(graph_count)
    sums.index_add_(0, graph_index, values * weight)
    counts.index_add_(0, graph_index, weight)
    valid = counts > 0
    if not bool(valid.any()):
        return values.sum() * 0.0
    return (sums[valid] / counts[valid]).mean()


def _bridge_step_targets(
    model: PocketDiffV4Model,
    inputs: Dict[str, object],
    targets: Dict[str, torch.Tensor],
    current: torch.Tensor,
    cumulative_chi: torch.Tensor,
    remaining_steps: int,
):
    current_origin, current_frame, current_valid = residue_frames(
        current, inputs["frame_index"]
    )
    holo_origin, holo_frame, holo_valid = residue_frames(
        targets["holo_pos"].float(), inputs["frame_index"]
    )
    bridge = remaining_transform_current_to_holo(
        current_origin,
        current_frame,
        holo_origin,
        holo_frame,
        frame_valid=current_valid & holo_valid,
    )
    translation = _project_vector_norm(
        bridge.translation_local / remaining_steps, model.max_translation
    )
    rotation = _project_vector_norm(
        bridge.rotvec_local / remaining_steps, model.max_rotation
    )
    cumulative_chi = cumulative_chi[..., :NUM_CHI]
    chi_apo = inputs.get("chi_apo", targets.get("chi_apo"))
    if chi_apo is None:
        raise KeyError("chi_apo must be present in the apo-only model input")
    chi = _wrap_angle(
        targets["chi_holo"][..., :NUM_CHI]
        - chi_apo[..., :NUM_CHI]
        - cumulative_chi
    ) / remaining_steps
    chi = chi.clamp(
        -0.95 * model.max_chi_step, 0.95 * model.max_chi_step
    )
    chi_mask = targets["chi_supervision_mask"].bool()[..., :NUM_CHI]
    chi = torch.where(chi_mask, chi, torch.zeros_like(chi))
    return translation, rotation, chi, bridge.valid, chi_mask


def _rollout_loss(
    model,
    batch,
    max_steps: int,
    noise_scale_min: float = 0.0,
    noise_scale_max: float = 0.0,
    oracle_rollout: bool = False,
    disable_chi: bool = False,
    direction_weight: float = 0.0,
):
    inputs, targets = batch["input"], batch["target"]
    apo = inputs["apo_pos"].float()
    holo = targets["holo_pos"].float()
    num_residues = inputs["residue_type"].shape[0]
    if dist.is_available() and dist.is_initialized():
        step_count = torch.tensor(
            [random.randint(2, max_steps) if dist.get_rank() == 0 else 0],
            dtype=torch.long,
        )
        dist.broadcast(step_count, src=0)
        num_steps = int(step_count.item())
    else:
        num_steps = random.randint(2, max_steps)
    current = apo
    if noise_scale_min < 0.0 or noise_scale_max < noise_scale_min:
        raise ValueError("invalid training noise scale range")
    noise_scale = random.uniform(noise_scale_min, noise_scale_max)
    initial_translation = torch.randn(
        (num_residues, 3), device=apo.device, dtype=apo.dtype
    ) * noise_scale
    initial_rotation = torch.randn_like(initial_translation) * (noise_scale * 0.5)
    initial_chi = torch.zeros(
        (num_residues, NUM_CHI), device=apo.device, dtype=apo.dtype
    )
    current = apply_motion(
        inputs, current, initial_translation, initial_rotation, initial_chi
    ).detach()

    cumulative_chi = torch.zeros(
        (num_residues, NUM_CHI), device=apo.device, dtype=apo.dtype
    )
    total_loss = apo.new_zeros(())
    endpoint_loss = apo.new_zeros(())
    ligand_pos = inputs["ligand_pos"].float()
    atom_ptr = inputs["atom_ptr"]
    residue_ptr = inputs["residue_ptr"]
    ligand_ptr = inputs["ligand_ptr"]
    graph_count = atom_ptr.numel() - 1
    atom_graph = torch.repeat_interleave(
        torch.arange(graph_count, device=apo.device),
        atom_ptr[1:] - atom_ptr[:-1],
    )
    residue_graph = torch.repeat_interleave(
        torch.arange(graph_count, device=apo.device),
        residue_ptr[1:] - residue_ptr[:-1],
    )
    atom_weight = apo.new_ones(apo.shape[0])
    for graph in range(graph_count):
        atom_start, atom_end = int(atom_ptr[graph]), int(atom_ptr[graph + 1])
        ligand_start, ligand_end = int(ligand_ptr[graph]), int(ligand_ptr[graph + 1])
        nearest = torch.cdist(
            apo[atom_start:atom_end], ligand_pos[ligand_start:ligand_end]
        ).min(dim=-1).values
        atom_weight[atom_start:atom_end] = torch.where(
            nearest <= 12.0, 1.0, 0.15
        )
    chi_graph = residue_graph[:, None].expand(-1, NUM_CHI).reshape(-1)
    ambiguous_mask = inputs.get(
        "chi_ambiguous_mask",
        torch.zeros(
            (num_residues, NUM_CHI), dtype=torch.bool, device=apo.device
        ),
    )

    for step in range(num_steps):
        remaining_steps = num_steps - step
        prediction = model(
            {"input": inputs},
            current,
            apo.new_tensor(float(remaining_steps) / num_steps),
        )
        (
            target_translation,
            target_rotation,
            target_chi,
            rigid_mask,
            chi_mask,
        ) = _bridge_step_targets(
            model,
            inputs,
            targets,
            current,
            cumulative_chi,
            remaining_steps,
        )
        if disable_chi:
            target_chi = torch.zeros_like(target_chi)
            chi_mask = torch.zeros_like(chi_mask)
        translation_loss = F.smooth_l1_loss(
            prediction["translation_local"], target_translation, reduction="none"
        ).sum(-1)
        rotation_loss = F.smooth_l1_loss(
            prediction["rotation_local"], target_rotation, reduction="none"
        ).sum(-1)
        rigid_loss = _graph_balanced_mean(
            translation_loss + rotation_loss,
            residue_graph,
            graph_count,
            mask=rigid_mask,
        )
        chi_error = _wrap_angle(prediction["chi_delta"] - target_chi)
        chi_point_loss = torch.where(
            ambiguous_mask,
            1.0 - torch.abs(torch.cos(chi_error)),
            1.0 - torch.cos(chi_error),
        )
        chi_loss = _graph_balanced_mean(
            chi_point_loss.reshape(-1),
            chi_graph,
            graph_count,
            mask=chi_mask.reshape(-1),
        )

        oracle_next = apply_motion(
            inputs,
            current,
            target_translation.float(),
            target_rotation.float(),
            target_chi.float(),
        ).detach()
        predicted_chi = prediction["chi_delta"]
        if disable_chi:
            predicted_chi = torch.zeros_like(predicted_chi)
        updated = apply_motion(
            inputs,
            current,
            prediction["translation_local"].float(),
            prediction["rotation_local"].float(),
            predicted_chi.float(),
        )
        bridge_coordinate_error = F.smooth_l1_loss(
            updated, oracle_next, reduction="none"
        ).mean(-1)
        bridge_endpoint = _graph_balanced_mean(
            bridge_coordinate_error,
            atom_graph,
            graph_count,
            weights=atom_weight,
        )
        coordinate_error = F.smooth_l1_loss(
            updated, holo, reduction="none"
        ).mean(-1)
        endpoint = _graph_balanced_mean(
            coordinate_error,
            atom_graph,
            graph_count,
            weights=atom_weight,
        )
        step_loss = 0.2 * rigid_loss + 0.1 * chi_loss + 0.75 * bridge_endpoint
        if direction_weight > 0.0:
            target_delta = oracle_next - current
            predicted_delta = updated - current
            target_norm = torch.linalg.vector_norm(target_delta, dim=-1)
            predicted_norm = torch.linalg.vector_norm(predicted_delta, dim=-1)
            direction_mask = (target_norm > 1e-6) & (predicted_norm > 1e-6)
            direction_cosine = (target_delta * predicted_delta).sum(-1) / (
                target_norm * predicted_norm
            ).clamp_min(1e-8)
            direction_loss = _graph_balanced_mean(
                1.0 - direction_cosine,
                atom_graph,
                graph_count,
                mask=direction_mask,
                weights=atom_weight,
            )
            step_loss = step_loss + direction_weight * direction_loss
        if step + 1 == num_steps:
            step_loss = step_loss + endpoint
        total_loss = total_loss + step_loss / num_steps
        endpoint_loss = endpoint_loss + endpoint.detach() / num_steps
        if oracle_rollout:
            current = oracle_next
            cumulative_chi = cumulative_chi + target_chi.detach()
        else:
            current = updated.detach()
            cumulative_chi = cumulative_chi + predicted_chi.detach()

    return total_loss, {
        "endpoint_loss": endpoint_loss,
        "steps": apo.new_tensor(float(num_steps)),
    }


def _save_checkpoint(path: Path, state: Dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(state, temporary)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


@torch.no_grad()
def _validation_loss(
    model,
    loader,
    device,
    max_steps: int,
    oracle_rollout: bool = False,
    disable_chi: bool = False,
    direction_weight: float = 0.0,
):
    model.eval()
    values = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        loss, _ = _rollout_loss(
            model,
            batch,
            max_steps,
            noise_scale_min=0.0,
            noise_scale_max=0.0,
            oracle_rollout=oracle_rollout,
            disable_chi=disable_chi,
            direction_weight=direction_weight,
        )
        values.append(loss.detach())
    if not values:
        return torch.zeros((), device=device)
    return torch.stack(values).mean()


def _synchronize_gradients(model, world: int):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    flat = torch.cat([
        (parameter.grad.detach().reshape(-1) if parameter.grad is not None
         else torch.zeros(parameter.numel(), device=parameter.device, dtype=parameter.dtype))
        for parameter in parameters
    ]).to(device="cpu")
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world)
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        synchronized = flat[offset : offset + count].view_as(parameter).to(parameter.device)
        if parameter.grad is None:
            parameter.grad = synchronized
        else:
            parameter.grad.copy_(synchronized)
        offset += count


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="pocketdiff_v4/data/residue_graphs.pt")
    parser.add_argument("--output-dir", default="pocketdiff_v4/runs/v4_3000")
    parser.add_argument("--updates", type=int, default=0, help="override epoch-derived updates")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32, help="per-rank batch size")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--noise-scale-min", type=float, default=0.0)
    parser.add_argument("--noise-scale-max", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--valid-every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=73421)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--vector-channels", type=int, default=16)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--oracle-rollout",
        action="store_true",
        help="advance training states with oracle bridge targets instead of model rollouts",
    )
    parser.add_argument(
        "--disable-chi",
        action="store_true",
        help="train and apply rigid residue motion only",
    )
    parser.add_argument("--direction-weight", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = _parse_args()
    distributed, rank, local_rank, world, device = _rank_info()
    seed = args.seed + rank
    random.seed(seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)

    cache = load_cache(args.data)
    dataset = ComplexDataset(cache["splits"]["train"])
    valid_dataset = ComplexDataset(cache["splits"]["valid"])
    sampler, loader = _build_train_loader(
        dataset,
        batch_size=args.batch_size,
        distributed=distributed,
        rank=rank,
        world=world,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if not len(loader):
        raise ValueError("training loader has no complete batches")
    valid_sampler, valid_loader = _build_eval_loader(
        valid_dataset, args.batch_size, distributed, rank, world, device
    )
    updates = args.updates or (len(loader) * args.epochs)
    if updates <= 0:
        raise ValueError("updates or epochs must be positive")

    model = PocketDiffV4Model(
        hidden=args.hidden,
        vector_channels=args.vector_channels,
        layers=args.layers,
    ).to(device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=args.amp and device.type == "cuda"
    )
    output_dir = Path(args.output_dir)
    step = 0
    epoch = 0
    best_valid = float("inf")
    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint.get("format") != "pocketdiff-v4-checkpoint":
            raise ValueError("unsupported resume checkpoint")
        if checkpoint.get("cache_source", {}).get("sha256") != cache["source"]["sha256"]:
            raise ValueError("resume checkpoint was trained from a different cache")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        step = int(checkpoint["update"])
        epoch = int(checkpoint["epoch"])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(vars(args), indent=2, sort_keys=True) + "\n"
        )

    start = time.time()
    loss_window = []
    while step < updates:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for raw_batch in loader:
            if step >= updates:
                break
            model.train()
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            autocast_enabled = args.amp and device.type == "cuda"
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                loss, metrics = _rollout_loss(
                    model,
                    batch,
                    args.max_steps,
                    noise_scale_min=args.noise_scale_min,
                    noise_scale_max=args.noise_scale_max,
                    oracle_rollout=args.oracle_rollout,
                    disable_chi=args.disable_chi,
                    direction_weight=args.direction_weight,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss at update %d" % step)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if distributed:
                _synchronize_gradients(model, world)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("non-finite gradient norm at update %d" % step)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            loss_window.append(
                (float(loss.detach()), float(metrics["endpoint_loss"]), float(grad_norm))
            )

            if step % args.log_every == 0 or step == 1:
                average = [sum(row[i] for row in loss_window) / len(loss_window) for i in range(3)]
                peak_memory = (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                    if device.type == "cuda" else 0.0
                )
                if distributed:
                    aggregate = torch.tensor(
                        [
                            sum(row[0] for row in loss_window),
                            sum(row[1] for row in loss_window),
                            sum(row[2] for row in loss_window),
                            float(len(loss_window)),
                        ],
                        dtype=torch.float64,
                    )
                    max_memory = torch.tensor([peak_memory], dtype=torch.float64)
                    dist.all_reduce(aggregate, op=dist.ReduceOp.SUM)
                    dist.all_reduce(max_memory, op=dist.ReduceOp.MAX)
                    average = (aggregate[:3] / aggregate[3]).tolist()
                    peak_memory = float(max_memory.item())
                if rank == 0:
                    print(json.dumps({
                        "update": step,
                        "epoch": epoch,
                        "train_loss": average[0],
                        "rollout_endpoint_loss": average[1],
                        "grad_norm": average[2],
                        "peak_cuda_memory_mb": round(peak_memory, 1),
                        "elapsed_sec": round(time.time() - start, 1),
                    }), flush=True)
                loss_window.clear()

            if rank == 0 and (step % args.save_every == 0 or step == updates):
                _save_checkpoint(output_dir / "latest.pt", {
                    "format": "pocketdiff-v4-checkpoint",
                    "version": 3,
                    "architecture": "pocketdiff-v4.2-rigid-oracle-bridge",
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "update": step,
                    "epoch": epoch,
                    "config": vars(args),
                    "cache_source": cache["source"],
                })
        epoch += 1
        if args.valid_every > 0 and epoch % args.valid_every == 0:
            if valid_sampler is not None:
                valid_sampler.set_epoch(epoch)
            valid_value = _validation_loss(
                model,
                valid_loader,
                device,
                args.max_steps,
                oracle_rollout=args.oracle_rollout,
                disable_chi=args.disable_chi,
                direction_weight=args.direction_weight,
            )
            if distributed:
                dist.all_reduce(valid_value, op=dist.ReduceOp.SUM)
                valid_value /= world
            if rank == 0:
                current_valid = float(valid_value.item())
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "update": step,
                            "valid_loss": current_valid,
                        }
                    ),
                    flush=True,
                )
                if current_valid < best_valid:
                    best_valid = current_valid
                    _save_checkpoint(output_dir / "best.pt", {
                        "format": "pocketdiff-v4-checkpoint",
                        "version": 3,
                        "architecture": "pocketdiff-v4.2-rigid-oracle-bridge",
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "update": step,
                        "epoch": epoch,
                        "config": vars(args),
                        "cache_source": cache["source"],
                        "best_valid_loss": best_valid,
                    })

    if distributed:
        dist.barrier()
    if rank == 0:
        summary = {
            "status": "completed",
            "updates": step,
            "epochs": epoch,
            "elapsed_sec": round(time.time() - start, 1),
            "checkpoint": str(output_dir / "latest.pt"),
            "best_checkpoint": str(output_dir / "best.pt"),
            "best_valid_loss": best_valid,
            "performance_evaluation": "not run; reserved until full training completes",
        }
        (output_dir / "train_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(summary), flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
