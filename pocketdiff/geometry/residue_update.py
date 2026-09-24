"""Explicitly named residue-local SE(3) update wrapper."""

from .bridge import apply_fractional_update


def apply_residue_fractional_se3_update(*args, **kwargs):
    """Alias kept separate so model code can depend on an update-only module."""

    return apply_fractional_update(*args, **kwargs)


__all__ = ["apply_fractional_update", "apply_residue_fractional_se3_update"]
