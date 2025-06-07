from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .base import REALISER_REGISTRY, EnergyTransformer

__all__ = ["REALISER_REGISTRY", "EnergyTransformer"]


def __getattr__(name: str) -> object:
    if name == "EnergyTransformer":
        from .base import EnergyTransformer

        return EnergyTransformer
    if name == "REALISER_REGISTRY":
        from .base import REALISER_REGISTRY

        return REALISER_REGISTRY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
