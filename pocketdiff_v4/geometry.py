from __future__ import annotations

from typing import Dict, List

import torch


def residue_frames(pos: torch.Tensor, frame_index: torch.Tensor):
    """Return CA origins, orthonormal N-CA-C frames, and validity."""
    xyz = pos.float()
    indices = frame_index.clamp_min(0)
    n, ca, c = (xyz[indices[:, i]] for i in range(3))
    u = c - ca
    v = n - ca
    u_norm = torch.linalg.vector_norm(u, dim=-1)
    v_norm = torch.linalg.vector_norm(v, dim=-1)
    e1 = u / u_norm.clamp_min(1e-6)[:, None]
    vn = v / v_norm.clamp_min(1e-6)[:, None]
    cross = torch.cross(e1, vn, dim=-1)
    cross_norm = torch.linalg.vector_norm(cross, dim=-1)
    e3 = cross / cross_norm.clamp_min(1e-6)[:, None]
    e2 = torch.cross(e3, e1, dim=-1)
    frame = torch.stack((e1, e2, e3), dim=-1)
    valid = (frame_index >= 0).all(-1) & (u_norm > 1e-5) & (v_norm > 1e-5) & (cross_norm > 1e-5)
    eye = torch.eye(3, dtype=xyz.dtype, device=xyz.device).expand_as(frame)
    frame = torch.where(valid[:, None, None], frame, eye)
    origin = torch.where((frame_index[:, 1] >= 0)[:, None], ca, torch.zeros_like(ca))
    return origin, frame, valid


def axis_angle_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Stable Rodrigues exponential with a differentiable zero-angle limit."""
    theta2 = (rotvec * rotvec).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-16))
    x, y, z = rotvec.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(-1, 3, 3)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand_as(skew)
    a = torch.where(theta2 < 1e-8, 1 - theta2 / 6 + theta2.square() / 120, torch.sin(theta) / theta)
    b = torch.where(theta2 < 1e-8, 0.5 - theta2 / 24 + theta2.square() / 720, (1 - torch.cos(theta)) / theta2.clamp_min(1e-16))
    return eye + a[:, :, None] * skew + b[:, :, None] * (skew @ skew)


def apply_rigid(
    pos: torch.Tensor,
    atom_to_residue: torch.Tensor,
    origin: torch.Tensor,
    frame: torch.Tensor,
    translation_local: torch.Tensor,
    rotation_local: torch.Tensor,
) -> torch.Tensor:
    rotation = axis_angle_matrix(rotation_local)
    global_rotation = frame @ rotation @ frame.transpose(-1, -2)
    global_translation = (frame @ translation_local.unsqueeze(-1)).squeeze(-1)
    relative = pos - origin[atom_to_residue]
    return pos + torch.bmm(
        (global_rotation - torch.eye(3, dtype=pos.dtype, device=pos.device))[atom_to_residue],
        relative.unsqueeze(-1),
    ).squeeze(-1) + global_translation[atom_to_residue]


def apply_chi_sparse(
    pos: torch.Tensor,
    chi_delta: torch.Tensor,
    chi_axis: torch.Tensor,
    chi_ptr: torch.Tensor,
    chi_downstream: torch.Tensor,
    chi_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply residue chi rotations, vectorized across residues for each slot."""
    result = pos
    lengths = chi_ptr[1:] - chi_ptr[:-1]
    for slot in range(5):
        rows = torch.arange(chi_delta.shape[0], device=pos.device) * 5 + slot
        row_lengths = lengths[rows]
        active = chi_mask[:, slot] & (row_lengths > 0) & (chi_axis[:, slot] >= 0).all(-1)
        active_rows = rows[active]
        if active_rows.numel() == 0:
            continue

        counts = lengths[active_rows]
        segment_residue = torch.repeat_interleave(
            torch.div(active_rows, 5, rounding_mode="floor"), counts
        )
        row_starts = chi_ptr[active_rows]
        segment_starts = torch.cumsum(counts, dim=0) - counts
        local_offsets = (
            torch.arange(int(counts.sum()), device=pos.device)
            - torch.repeat_interleave(segment_starts, counts)
        )
        starts = chi_axis[segment_residue, slot, 0]
        ends = chi_axis[segment_residue, slot, 1]
        selected = chi_downstream[torch.repeat_interleave(row_starts, counts) + local_offsets]

        axes = result[ends] - result[starts]
        axes = axes / torch.linalg.vector_norm(axes, dim=-1, keepdim=True).clamp_min(1e-8)
        points = result[selected] - result[ends]
        angles = chi_delta[segment_residue, slot]
        cos_a, sin_a = torch.cos(angles), torch.sin(angles)
        rotated = (
            points * cos_a[:, None]
            + torch.cross(axes, points, dim=-1) * sin_a[:, None]
            + axes * (axes * points).sum(-1, keepdim=True) * (1 - cos_a[:, None])
        )
        result = result.clone()
        result[selected] = result[ends] + rotated
    return result


def apply_motion(
    sample: Dict[str, object],
    pos: torch.Tensor,
    translation_local: torch.Tensor,
    rotation_local: torch.Tensor,
    chi_delta: torch.Tensor,
    fraction: float = 1.0,
) -> torch.Tensor:
    device = pos.device
    atom_to_residue = sample["atom_to_residue"].to(device)
    frame_index = sample["frame_index"].to(device)
    origin, frame, valid = residue_frames(pos, frame_index)
    t = translation_local * fraction
    r = rotation_local * fraction
    t = torch.where(valid[:, None], t, torch.zeros_like(t))
    r = torch.where(valid[:, None], r, torch.zeros_like(r))
    moved = apply_rigid(pos, atom_to_residue, origin, frame, t, r)
    chi_mask = sample["chi_geometry_mask"].to(device) & valid[:, None]
    return apply_chi_sparse(
        moved,
        chi_delta * fraction,
        sample["chi_axis"].to(device),
        sample["chi_ptr"].to(device),
        sample["chi_downstream"].to(device),
        chi_mask,
    )


def pack_chi_sparse(
    atom_names: List[str], residue_names: List[str], atom_to_residue: torch.Tensor
):
    """Build compact chi-axis and downstream CSR arrays from atom topology."""
    from pocketdiff.geometry.chi import build_chi_update_metadata

    metadata = build_chi_update_metadata(atom_names, atom_to_residue, residue_names)
    nr, _, natoms = metadata.downstream_atom_mask.shape
    axes = torch.stack((metadata.axis_start, metadata.axis_end), dim=-1)
    ptr = [0]
    downstream = []
    for row in range(nr * 5):
        residue, slot = divmod(row, 5)
        values = torch.where(metadata.downstream_atom_mask[residue, slot])[0].tolist()
        downstream.extend(values)
        ptr.append(len(downstream))
    return (
        axes.long(),
        torch.tensor(ptr, dtype=torch.long),
        torch.tensor(downstream, dtype=torch.long),
        metadata.valid.bool(),
    )


def residue_level_names(atom_names: List[str], residue_names: List[str],
                         atom_to_residue: torch.Tensor, num_residues: int) -> List[str]:
    """Validate per-atom residue names and return one canonical name per residue."""
    if len(atom_names) != len(residue_names) or len(atom_names) != atom_to_residue.numel():
        raise ValueError("atom names, residue names, and residue mapping lengths differ")
    result = [None] * num_residues
    for atom_name, residue_name, residue in zip(
        atom_names, residue_names, atom_to_residue.detach().cpu().tolist()
    ):
        name = residue_name.strip().upper()
        previous = result[residue]
        if previous is not None and previous != name:
            raise ValueError("inconsistent residue names for residue %d" % residue)
        result[residue] = name
    if any(name is None for name in result):
        raise ValueError("atom_to_residue does not cover every residue")
    return result
