"""Geometry primitives for the independent PocketDiff implementation."""

from .bridge import (
    BridgeState,
    OracleReconstructionMetrics,
    RemainingTransform,
    apply_fractional_update,
    build_bridge_state,
    oracle_reconstruction_metrics,
    remaining_transform_current_to_holo,
)
from .frames import FRAME_ATOMS, ResidueFrameResult, build_residue_frames, frame_orthogonality_error
from .residue_update import apply_residue_fractional_se3_update
from .so3 import so3_compose, so3_exp, so3_geodesic_angle, so3_inverse, so3_is_proper, so3_log
from .chi import (
    CHI_DEFINITIONS,
    ChiUpdateMetadata,
    ChiUpdateResult,
    apply_chi_updates,
    build_chi_update_metadata,
    extract_chi_angles,
    periodic_chi_delta,
)
from .current_state import CurrentChiState, build_current_chi_state
from .oracle import OracleReconstruction, oracle_rigid_chi_reconstruction

__all__ = [
    "BridgeState",
    "FRAME_ATOMS",
    "OracleReconstructionMetrics",
    "RemainingTransform",
    "ResidueFrameResult",
    "apply_fractional_update",
    "apply_residue_fractional_se3_update",
    "build_bridge_state",
    "build_residue_frames",
    "frame_orthogonality_error",
    "oracle_reconstruction_metrics",
    "remaining_transform_current_to_holo",
    "so3_compose",
    "so3_exp",
    "so3_geodesic_angle",
    "so3_inverse",
    "so3_is_proper",
    "so3_log",
    "CHI_DEFINITIONS",
    "ChiUpdateMetadata",
    "ChiUpdateResult",
    "apply_chi_updates",
    "build_chi_update_metadata",
    "extract_chi_angles",
    "periodic_chi_delta",
    "CurrentChiState",
    "build_current_chi_state",
    "OracleReconstruction",
    "oracle_rigid_chi_reconstruction",
]
