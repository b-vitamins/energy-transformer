"""High-level Energy Transformer models."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .base import EnergyTransformer
    from .configs import ViETConfig, ViSETConfig, ViTConfig

__all__ = ["EnergyTransformer", "ViETConfig", "ViSETConfig", "ViTConfig"]


def __getattr__(name: str) -> object:
    """Lazily import the :class:`EnergyTransformer` model.

    Parameters
    ----------
    name : str
        Attribute being requested.

    Returns
    -------
    object
        The requested attribute.

    Raises
    ------
    AttributeError
        If ``name`` is not ``"EnergyTransformer"``.
    """
    if name == "EnergyTransformer":
        from .base import EnergyTransformer

        return EnergyTransformer
    if name in {"ViTConfig", "ViETConfig", "ViSETConfig"}:
        from .configs import ViETConfig, ViSETConfig, ViTConfig

        return {
            "ViTConfig": ViTConfig,
            "ViETConfig": ViETConfig,
            "ViSETConfig": ViSETConfig,
        }[name]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
