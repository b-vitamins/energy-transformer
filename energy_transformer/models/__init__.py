from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .base import EnergyTransformer

__all__ = ["EnergyTransformer"]


def __getattr__(name: str) -> object:
    if name == "EnergyTransformer":
        from .base import EnergyTransformer

        return EnergyTransformer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
