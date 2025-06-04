"""Utility helpers for Energy Transformer layers."""

from torch import Tensor

__all__ = ["_validate_scalar_energy"]


def _validate_scalar_energy(energy: Tensor, component_name: str) -> None:
    """Validate that ``energy`` is a scalar tensor."""
    if energy.dim() != 0:
        raise ValueError(
            f"{component_name} must return scalar energy tensor, got shape {energy.shape}"
        )
