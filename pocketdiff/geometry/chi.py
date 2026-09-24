"""Standard protein side-chain chi-angle geometry."""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Optional, Sequence, Tuple, Union

import torch


AtomSpec = Union[str, Tuple[str, ...]]
ChiDefinition = Tuple[AtomSpec, AtomSpec, AtomSpec, AtomSpec]


@dataclass(frozen=True)
class ChiUpdateResult:
    """Result of applying explicit residue-local χ rotations."""

    positions: torch.Tensor
    applied_chi: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self) -> None:
        if self.positions.ndim != 2 or self.positions.shape[-1] != 3:
            raise ValueError("positions must have shape [N, 3]")
        if self.applied_chi.ndim != 2 or not 1 <= self.applied_chi.shape[-1] <= 5:
            raise ValueError("applied_chi must have shape [Nr, Nchi] with 1 <= Nchi <= 5")
        if self.valid.dtype != torch.bool or self.valid.shape != self.applied_chi.shape:
            raise ValueError("valid must be BoolTensor with shape [Nr, Nchi]")
        if not (torch.isfinite(self.positions).all() and torch.isfinite(self.applied_chi).all()):
            raise ValueError("χ update outputs must be finite")


@dataclass(frozen=True)
class ChiUpdateMetadata:
    """Explicit atom topology needed by :func:`apply_chi_updates`."""

    axis_start: torch.Tensor
    axis_end: torch.Tensor
    downstream_atom_mask: torch.Tensor
    valid: torch.Tensor
    quartet_indices: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.axis_start.dtype != torch.long or self.axis_end.dtype != torch.long:
            raise TypeError("axis indices must be LongTensor")
        if self.axis_start.ndim != 2 or not 1 <= self.axis_start.shape[-1] <= 5 or self.axis_end.shape != self.axis_start.shape:
            raise ValueError("axis indices must have shape [Nr, Nchi] with 1 <= Nchi <= 5")
        if self.valid.dtype != torch.bool or self.valid.shape != self.axis_start.shape:
            raise ValueError("valid must be BoolTensor with shape [Nr, 5]")
        if self.downstream_atom_mask.ndim != 3 or self.downstream_atom_mask.dtype != torch.bool or self.downstream_atom_mask.shape[:2] != self.axis_start.shape:
            raise ValueError("downstream_atom_mask must have shape [Nr, Nchi, N]")
        if self.axis_start.numel() and (
            bool((self.axis_start < -1).any()) or bool((self.axis_end < -1).any())
            or bool((self.axis_start >= self.downstream_atom_mask.shape[-1]).any())
            or bool((self.axis_end >= self.downstream_atom_mask.shape[-1]).any())
        ):
            raise ValueError("axis indices must be -1 or valid atom indices")
        if self.quartet_indices is not None:
            if self.quartet_indices.dtype != torch.long:
                raise TypeError("quartet_indices must be LongTensor")
            if self.quartet_indices.shape != self.axis_start.shape + (4,):
                raise ValueError("quartet_indices must have shape [Nr, Nchi, 4]")
            if self.quartet_indices.device != self.axis_start.device:
                raise ValueError("quartet_indices must share the metadata device")
            if self.quartet_indices.numel() and (
                bool((self.quartet_indices < -1).any())
                or bool((self.quartet_indices >= self.downstream_atom_mask.shape[-1]).any())
            ):
                raise ValueError("quartet_indices must be -1 or valid atom indices")

    @property
    def num_residues(self) -> int:
        return int(self.axis_start.shape[0])

    @property
    def num_atoms(self) -> int:
        return int(self.downstream_atom_mask.shape[-1])


def _aliases(*names):
    return names[0] if len(names) == 1 else tuple(names)


# The definitions follow the conventional N-CA-CB side-chain ordering.  When
# a terminal atom has equivalent alternatives, the first atom present in the
# pocket is selected deterministically.
CHI_DEFINITIONS: Mapping[str, Tuple[ChiDefinition, ...]] = {
    "ALA": (),
    "CYS": ((_aliases("N"), _aliases("CA"), _aliases("CB"), _aliases("SG")),),
    "ASP": (
        (_aliases("N"), _aliases("CA"), _aliases("CB"), _aliases("CG")),
        (_aliases("CA"), _aliases("CB"), _aliases("CG"), _aliases("OD1", "OD2")),
    ),
    "GLU": (
        (_aliases("N"), _aliases("CA"), _aliases("CB"), _aliases("CG")),
        (_aliases("CA"), _aliases("CB"), _aliases("CG"), _aliases("CD")),
        (_aliases("CB"), _aliases("CG"), _aliases("CD"), _aliases("OE1", "OE2")),
    ),
    "PHE": (
        (_aliases("N"), _aliases("CA"), _aliases("CB"), _aliases("CG")),
        (_aliases("CA"), _aliases("CB"), _aliases("CG"), _aliases("CD1", "CD2")),
    ),
    "HIS": (
        (_aliases("N"), _aliases("CA"), _aliases("CB"), _aliases("CG")),
        (_aliases("CA"), _aliases("CB"), _aliases("CG"), _aliases("ND1", "CD1")),
    ),
    "ILE": (
        (_aliases("N"), _aliases("CA"), _aliases("CB"), _aliases("CG1", "CG2")),
        (_aliases("CA"), _aliases("CB"), _aliases("CG1", "CG2"), _aliases("CD1")),
    ),
    "LYS": (
        (_aliases("N"), _aliases("CA"), _aliases("CB"), _aliases("CG")),
        (_aliases("CA"), _aliases("CB"), _aliases("CG"), _aliases("CD")),
        (_aliases("CB"), _aliases("CG"), _aliases("CD"), _aliases("CE")),
        (_aliases("CG"), _aliases("CD"), _aliases("CE"), _aliases("NZ")),
    ),
    "LEU": (
        (_aliases("N"), _aliases("CA"), _aliases("CB"), _aliases("CG")),
        (_aliases("CA"), _aliases("CB"), _aliases("CG"), _aliases("CD1", "CD2")),
    ),
    "MET": (
        (_aliases("N"), _aliases("CA"), _aliases("CB"), _aliases("CG")),
        (_aliases("CA"), _aliases("CB"), _aliases("CG"), _aliases("SD")),
        (_aliases("CB"), _aliases("CG"), _aliases("SD"), _aliases("CE")),
    ),
    "ASN": (
        (_aliases("N"), _aliases("CA"), _aliases("CB"), _aliases("CG")),
        (_aliases("CA"), _aliases("CB"), _aliases("CG"), _aliases("OD1", "ND2")),
    ),
    "PRO": (
        (_aliases("N"), _aliases("CA"), _aliases("CB"), _aliases("CG")),
        (_aliases("CA"), _aliases("CB"), _aliases("CG"), _aliases("CD")),
    ),
    "GLN": (
        (_aliases("N"), _aliases("CA"), _aliases("CB"), _aliases("CG")),
        (_aliases("CA"), _aliases("CB"), _aliases("CG"), _aliases("CD")),
        (_aliases("CB"), _aliases("CG"), _aliases("CD"), _aliases("OE1", "NE2")),
    ),
    "ARG": (
        (_aliases("N"), _aliases("CA"), _aliases("CB"), _aliases("CG")),
        (_aliases("CA"), _aliases("CB"), _aliases("CG"), _aliases("CD")),
        (_aliases("CB"), _aliases("CG"), _aliases("CD"), _aliases("NE")),
        (_aliases("CG"), _aliases("CD"), _aliases("NE"), _aliases("CZ")),
    ),
    "SER": ((_aliases("N"), _aliases("CA"), _aliases("CB"), _aliases("OG")),),
    "THR": ((_aliases("N"), _aliases("CA"), _aliases("CB"), _aliases("OG1", "OG")),),
    "VAL": ((_aliases("N"), _aliases("CA"), _aliases("CB"), _aliases("CG1", "CG2")),),
    "TRP": (
        (_aliases("N"), _aliases("CA"), _aliases("CB"), _aliases("CG")),
        (_aliases("CA"), _aliases("CB"), _aliases("CG"), _aliases("CD1", "CD2")),
    ),
    "TYR": (
        (_aliases("N"), _aliases("CA"), _aliases("CB"), _aliases("CG")),
        (_aliases("CA"), _aliases("CB"), _aliases("CG"), _aliases("CD1", "CD2")),
    ),
}


# Complete standard heavy-atom connectivity, independent of the χ quartets.
# Cross-checked against Apo2Mol-main/utils/data.py CHI_BOND_DICTS and the
# AlphaFold residue_constants rigid groups. No upstream code is imported.
SIDECHAIN_BONDS = {
    "ALA": "CA-CB", "GLY": "",
    "CYS": "CA-CB CB-SG",
    "ASP": "CA-CB CB-CG CG-OD1 CG-OD2",
    "GLU": "CA-CB CB-CG CG-CD CD-OE1 CD-OE2",
    "ASN": "CA-CB CB-CG CG-OD1 CG-ND2",
    "GLN": "CA-CB CB-CG CG-CD CD-OE1 CD-NE2",
    "ARG": "CA-CB CB-CG CG-CD CD-NE NE-CZ CZ-NH1 CZ-NH2",
    "LYS": "CA-CB CB-CG CG-CD CD-CE CE-NZ",
    "MET": "CA-CB CB-CG CG-SD SD-CE",
    "SER": "CA-CB CB-OG", "THR": "CA-CB CB-OG1 CB-CG2",
    "VAL": "CA-CB CB-CG1 CB-CG2", "ILE": "CA-CB CB-CG1 CB-CG2 CG1-CD1",
    "LEU": "CA-CB CB-CG CG-CD1 CG-CD2",
    "PHE": "CA-CB CB-CG CG-CD1 CG-CD2 CD1-CE1 CD2-CE2 CE1-CZ CE2-CZ",
    "TYR": "CA-CB CB-CG CG-CD1 CG-CD2 CD1-CE1 CD2-CE2 CE1-CZ CE2-CZ CZ-OH",
    "HIS": "CA-CB CB-CG CG-ND1 CG-CD2 ND1-CE1 CE1-NE2 NE2-CD2",
    "TRP": "CA-CB CB-CG CG-CD1 CG-CD2 CD1-NE1 NE1-CE2 CE2-CD2 CD2-CE3 CE3-CZ3 CZ3-CH2 CH2-CZ2 CZ2-CE2",
    "PRO": "CA-CB CB-CG CG-CD CD-N",
}


def build_chi_update_metadata(
    atom_names: Sequence[str],
    atom_to_residue: torch.Tensor,
    residue_names: Sequence[str],
    *,
    num_chi: int = 5,
) -> ChiUpdateMetadata:
    """Build heavy-atom rotations using complete residue bond templates.

    Canonical χ names are required; alternative atom names never substitute
    for a missing axis or quartet atom. Full-template connectivity determines
    the distal component, even in cropped pockets. PRO is frozen because its
    ring requires a closure solver. Unknown atom names (including hydrogens)
    are rejected rather than silently left behind. Cross-residue covalent
    links are outside this residue-local contract.
    """
    if not isinstance(num_chi, int) or isinstance(num_chi, bool) or not 1 <= num_chi <= 5:
        raise ValueError("num_chi must lie in [1, 5]")
    if atom_to_residue.dtype != torch.long or atom_to_residue.ndim != 1:
        raise TypeError("atom_to_residue must be LongTensor [N]")
    if len(atom_names) != atom_to_residue.shape[0]:
        raise ValueError("atom_names and atom_to_residue have different lengths")
    if not residue_names:
        raise ValueError("residue_names must be non-empty")
    residue_count = len(residue_names)
    if atom_to_residue.numel() and (
        int(atom_to_residue.min()) < 0 or int(atom_to_residue.max()) >= residue_count
    ):
        raise ValueError("atom_to_residue contains an invalid residue id")
    atom_ids = tuple(int(value) for value in atom_to_residue.detach().cpu().tolist())
    names = tuple(str(value) for value in atom_names)
    residue_key = tuple(str(value) for value in residue_names)
    axis_cpu, end_cpu, downstream_cpu, valid_cpu, quartet_cpu = _cached_chi_template(
        names, atom_ids, residue_key, num_chi
    )
    device = atom_to_residue.device
    return ChiUpdateMetadata(
        axis_cpu.to(device=device),
        end_cpu.to(device=device),
        downstream_cpu.to(device=device),
        valid_cpu.to(device=device),
        quartet_cpu.to(device=device),
    )


@lru_cache(maxsize=4096)
def _cached_chi_template(
    atom_names: Tuple[str, ...],
    atom_ids: Tuple[int, ...],
    residue_names: Tuple[str, ...],
    num_chi: int,
):
    """Build topology once on CPU; all coordinate-dependent work stays tensorized."""
    residue_count = len(residue_names)
    atom_count = len(atom_names)
    axis_start = torch.full((residue_count, num_chi), -1, dtype=torch.long)
    axis_end = torch.full_like(axis_start, -1)
    downstream = torch.zeros((residue_count, num_chi, atom_count), dtype=torch.bool)
    valid = torch.zeros_like(axis_start, dtype=torch.bool)
    quartets = torch.full((residue_count, num_chi, 4), -1, dtype=torch.long)
    for residue_id, residue_name in enumerate(residue_names):
        name = str(residue_name).upper()
        indices = [i for i, r in enumerate(atom_ids) if r == residue_id]
        by_name = {atom_names[i]: i for i in indices}
        if len(by_name) != len(indices):
            raise ValueError("duplicate atom names in residue %d" % residue_id)
        if name not in SIDECHAIN_BONDS:
            continue
        bonds = [tuple(pair.split('-')) for pair in
                 ("N-CA CA-C C-O C-OXT " + SIDECHAIN_BONDS[name]).split()]
        adjacency = {}
        for left, right in bonds:
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)
        unknown = set(by_name) - set(adjacency)
        if unknown:
            raise ValueError("unsupported atoms in %s: %s" % (name, sorted(unknown)))
        if name == "PRO":
            continue
        for chi_id, definition in enumerate(CHI_DEFINITIONS.get(name, ())[:num_chi]):
            quartet = [spec if isinstance(spec, str) else spec[0] for spec in definition]
            if any(atom not in by_name for atom in quartet):
                continue
            quartets[residue_id, chi_id] = torch.tensor(
                [by_name[atom] for atom in quartet], dtype=torch.long
            )
            start, end = quartet[1:3]
            visited, queue = {end}, [end]
            while queue:
                current = queue.pop()
                for neighbor in adjacency[current]:
                    if {current, neighbor} == {start, end}:
                        continue
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            if start in visited:  # The axis lies in a closed ring.
                continue
            axis_start[residue_id, chi_id] = by_name[start]
            axis_end[residue_id, chi_id] = by_name[end]
            valid[residue_id, chi_id] = True
            for atom in visited - {end}:
                if atom in by_name:
                    downstream[residue_id, chi_id, by_name[atom]] = True
    return axis_start, axis_end, downstream, valid, quartets


def _dihedral(points: torch.Tensor):
    b0 = points[1] - points[0]
    b1 = points[2] - points[1]
    b2 = points[3] - points[2]
    b1_norm = torch.linalg.vector_norm(b1)
    if float(b1_norm) <= 1e-8:
        return torch.tensor(0.0, dtype=points.dtype, device=points.device), False
    axis = b1 / b1_norm
    v = b0 - torch.dot(b0, axis) * axis
    w = b2 - torch.dot(b2, axis) * axis
    v_norm = torch.linalg.vector_norm(v)
    w_norm = torch.linalg.vector_norm(w)
    if float(v_norm) <= 1e-8 or float(w_norm) <= 1e-8:
        return torch.tensor(0.0, dtype=points.dtype, device=points.device), False
    x = torch.dot(v, w)
    y = torch.dot(torch.cross(axis, v), w)
    return torch.atan2(y, x), True


def _find_atom(atom_indices, spec):
    if isinstance(spec, str):
        return atom_indices.get(spec)
    for name in spec:
        if name in atom_indices:
            return atom_indices[name]
    return None


def extract_chi_angles(
    positions: torch.Tensor,
    atom_names: Sequence[str],
    atom_to_residue: torch.Tensor,
    residue_names: Sequence[str],
    *,
    num_chi: int = 5,
    metadata: Optional[ChiUpdateMetadata] = None,
):
    """Return angles and masks with shape [num_residues, num_chi]."""
    if positions.ndim != 2 or positions.shape[-1] != 3 or not positions.is_floating_point():
        raise ValueError("positions must be floating [N, 3]")
    if len(atom_names) != positions.shape[0]:
        raise ValueError("atom_names and positions have different lengths")
    if atom_to_residue.dtype != torch.long or atom_to_residue.shape != (positions.shape[0],):
        raise ValueError("atom_to_residue must be LongTensor [N]")
    residue_count = len(residue_names)
    if residue_count == 0 or num_chi <= 0:
        raise ValueError("residue_names and num_chi must be non-empty")
    if atom_to_residue.numel() and (
        int(atom_to_residue.min()) < 0 or int(atom_to_residue.max()) >= residue_count
    ):
        raise ValueError("atom_to_residue contains an invalid residue id")
    metadata = metadata or build_chi_update_metadata(
        atom_names, atom_to_residue, residue_names, num_chi=num_chi
    )
    if metadata.quartet_indices is None:
        raise ValueError("chi metadata does not contain quartet indices")
    quartet = metadata.quartet_indices
    safe = quartet.clamp_min(0)
    points = positions[safe]
    b0 = points[..., 1, :] - points[..., 0, :]
    b1 = points[..., 2, :] - points[..., 1, :]
    b2 = points[..., 3, :] - points[..., 2, :]
    b1_norm = torch.linalg.vector_norm(b1, dim=-1)
    axis = b1 / b1_norm.clamp_min(1.0e-8)[..., None]
    v = b0 - (b0 * axis).sum(dim=-1, keepdim=True) * axis
    w = b2 - (b2 * axis).sum(dim=-1, keepdim=True) * axis
    v_norm = torch.linalg.vector_norm(v, dim=-1)
    w_norm = torch.linalg.vector_norm(w, dim=-1)
    values = torch.atan2(
        (torch.cross(axis, v, dim=-1) * w).sum(dim=-1),
        (v * w).sum(dim=-1),
    )
    valid = (
        metadata.valid
        & (quartet >= 0).all(dim=-1)
        & torch.isfinite(values)
        & torch.isfinite(b1_norm)
        & torch.isfinite(v_norm)
        & torch.isfinite(w_norm)
        & (b1_norm > 1.0e-8)
        & (v_norm > 1.0e-8)
        & (w_norm > 1.0e-8)
    )
    return torch.where(valid, values, torch.zeros_like(values)), valid


def periodic_chi_delta(
    chi_apo: torch.Tensor,
    chi_holo: torch.Tensor,
    chi_mask: torch.Tensor,
):
    """Return holo-minus-apo angular deltas wrapped to [-pi, pi]."""
    if chi_apo.shape != chi_holo.shape or chi_apo.ndim != 2:
        raise ValueError("chi tensors must have identical shape [Nr, Nchi]")
    if chi_mask.dtype != torch.bool or chi_mask.shape != chi_apo.shape:
        raise ValueError("chi_mask must be BoolTensor with the chi shape")
    delta = torch.atan2(torch.sin(chi_holo - chi_apo),
                        torch.cos(chi_holo - chi_apo))
    return torch.where(chi_mask, delta, torch.zeros_like(delta))


def apply_chi_updates(
    positions: torch.Tensor,
    axis_start: torch.Tensor,
    axis_end: torch.Tensor,
    downstream_atom_mask: torch.Tensor,
    chi_delta: torch.Tensor,
    *,
    valid: Optional[torch.Tensor] = None,
    eps: float = 1.0e-8,
) -> ChiUpdateResult:
    """Apply explicit χ rotations to a protein coordinate tensor.

    ``axis_start``/``axis_end`` contain global atom indices for each residue
    and χ slot.  ``downstream_atom_mask[r, c]`` identifies the atoms rotated
    around that bond.  This deliberately requires topology to be supplied by
    the caller: guessing chemical connectivity from a pocket atom list is a
    later phase.  Rotations are applied in χ order, so a χ2 mask can consume
    the coordinates produced by χ1.
    """

    if positions.ndim != 2 or positions.shape[-1] != 3 or not positions.is_floating_point():
        raise ValueError("positions must be floating [N, 3]")
    if not torch.isfinite(positions).all():
        raise ValueError("positions must be finite")
    if axis_start.dtype != torch.long or axis_end.dtype != torch.long:
        raise TypeError("axis_start and axis_end must be LongTensor")
    if axis_start.ndim != 2 or not 1 <= axis_start.shape[-1] <= 5 or axis_end.shape != axis_start.shape:
        raise ValueError("axis indices must have shape [Nr, Nchi] with 1 <= Nchi <= 5")
    if chi_delta.ndim != 2 or chi_delta.shape != axis_start.shape or not chi_delta.is_floating_point():
        raise ValueError("chi_delta must be floating [Nr, Nchi]")
    if not torch.isfinite(chi_delta).all():
        raise ValueError("chi_delta must be finite")
    num_residues = int(axis_start.shape[0])
    if downstream_atom_mask.dtype != torch.bool or downstream_atom_mask.shape != (
        num_residues, axis_start.shape[1], positions.shape[0]
    ):
        raise ValueError("downstream_atom_mask must have shape [Nr, Nchi, N]")
    if axis_start.device != positions.device or axis_end.device != positions.device or downstream_atom_mask.device != positions.device or chi_delta.device != positions.device:
        raise ValueError("χ update tensors must be on the same device as positions")
    if valid is None:
        valid = (axis_start >= 0) & (axis_end >= 0)
    if valid.dtype != torch.bool or valid.shape != axis_start.shape:
        raise ValueError("valid must be BoolTensor with shape [Nr, Nchi]")
    if valid.device != positions.device:
        raise ValueError("valid must be on the positions device")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive")
    if axis_start.numel() and (
        bool((axis_start < -1).any()) or bool((axis_end < -1).any())
        or bool((axis_start >= positions.shape[0]).any())
        or bool((axis_end >= positions.shape[0]).any())
    ):
        raise ValueError("axis indices must be -1 or valid global atom indices")

    current = positions
    applied = torch.zeros_like(chi_delta)
    applied_valid = torch.zeros_like(chi_delta, dtype=torch.bool)
    # χ slots remain sequential because χ2 depends on χ1-updated coordinates.
    # Within one slot, operate only on downstream residue/atom pairs.  The
    # previous dense implementation materialized [Nr, N, 3] Rodrigues tensors;
    # the sparse gather/index_add path keeps peak memory proportional to the
    # number of actually rotated atoms.
    for chi_id in range(axis_start.shape[1]):
        start = axis_start[:, chi_id].clamp_min(0)
        end = axis_end[:, chi_id].clamp_min(0)
        requested = valid[:, chi_id] & (axis_start[:, chi_id] >= 0) & (axis_end[:, chi_id] >= 0)
        axis_vector = current[end] - current[start]
        axis_norm = torch.linalg.vector_norm(axis_vector, dim=-1)
        slot_valid = requested & torch.isfinite(axis_norm) & (axis_norm > eps)
        axis = axis_vector / axis_norm.clamp_min(eps)[:, None]
        angle = torch.atan2(
            torch.sin(chi_delta[:, chi_id]),
            torch.cos(chi_delta[:, chi_id]),
        )

        residue_ids, atom_ids = downstream_atom_mask[:, chi_id].nonzero(as_tuple=True)
        if atom_ids.numel():
            point = current[atom_ids]
            origin = current[start[residue_ids]]
            pair_axis = axis[residue_ids]
            pair_angle = angle[residue_ids]
            relative = point - origin
            parallel = (relative * pair_axis).sum(dim=-1, keepdim=True) * pair_axis
            perpendicular = relative - parallel
            rotated = (
                origin
                + parallel
                + torch.cos(pair_angle)[:, None] * perpendicular
                + torch.sin(pair_angle)[:, None] * torch.cross(
                    pair_axis, perpendicular, dim=-1
                )
            )
            delta = rotated - point
            active_pair = slot_valid[residue_ids]
            derivative_at_zero = torch.cross(pair_axis, perpendicular, dim=-1)
            zero_angle = pair_angle == 0
            zero_angle_straight_through = (
                (pair_angle - pair_angle.detach())[:, None] * derivative_at_zero
            )
            delta = torch.where(
                zero_angle[:, None],
                zero_angle_straight_through,
                delta,
            )
            delta = torch.where(active_pair[:, None], delta, torch.zeros_like(delta))
            current = current.index_add(0, atom_ids, delta)

        applied[:, chi_id] = torch.where(slot_valid, angle, torch.zeros_like(angle))
        applied_valid[:, chi_id] = slot_valid
    return ChiUpdateResult(positions=current, applied_chi=applied, valid=applied_valid)


__all__ = [
    "CHI_DEFINITIONS",
    "SIDECHAIN_BONDS",
    "ChiUpdateMetadata",
    "ChiUpdateResult",
    "apply_chi_updates",
    "build_chi_update_metadata",
    "extract_chi_angles",
    "periodic_chi_delta",
]
