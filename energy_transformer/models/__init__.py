from __future__ import annotations

"""High-level Energy Transformer models."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .base import EnergyTransformer

__all__ = ["EnergyTransformer"]


def __getattr__(name: str) -> object:
    if name == "EnergyTransformer":
        from .base import EnergyTransformer

        return EnergyTransformer
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
