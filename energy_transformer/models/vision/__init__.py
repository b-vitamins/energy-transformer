"""Vision-specific Energy Transformer model variants."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from energy_transformer.models.base import EnergyTransformer

    from .viet import (
        VisionEnergyTransformer,
        viet_2l_cifar,
        viet_4l_cifar,
        viet_6l_cifar,
        viet_base,
        viet_large,
        viet_small,
        viet_small_cifar,
        viet_tiny,
        viet_tiny_cifar,
    )
    from .viset import (
        VisionSimplicialTransformer,
        viset_2l_cifar,
        viset_4l_cifar,
        viset_6l_cifar,
        viset_base,
        viset_large,
        viset_small,
        viset_small_cifar,
        viset_tiny,
        viset_tiny_cifar,
    )
    from .vit import (
        VisionTransformer,
        vit_base,
        vit_large,
        vit_small,
        vit_small_cifar,
        vit_tiny,
        vit_tiny_cifar,
    )

__all__ = [
    "EnergyTransformer",
    "VisionEnergyTransformer",
    "VisionSimplicialTransformer",
    "VisionTransformer",
    "viet_2l_cifar",
    "viet_4l_cifar",
    "viet_6l_cifar",
    "viet_base",
    "viet_large",
    "viet_small",
    "viet_small_cifar",
    "viet_tiny",
    "viet_tiny_cifar",
    "viset_2l_cifar",
    "viset_4l_cifar",
    "viset_6l_cifar",
    "viset_base",
    "viset_large",
    "viset_small",
    "viset_small_cifar",
    "viset_tiny",
    "viset_tiny_cifar",
    "vit_base",
    "vit_large",
    "vit_small",
    "vit_small_cifar",
    "vit_tiny",
    "vit_tiny_cifar",
]


def __getattr__(name: str) -> object:
    if name == "EnergyTransformer":
        from energy_transformer.models.base import (
            EnergyTransformer as _EnergyTransformer,
        )

        return _EnergyTransformer
    if name in {
        "VisionEnergyTransformer",
        "viet_2l_cifar",
        "viet_4l_cifar",
        "viet_6l_cifar",
        "viet_base",
        "viet_large",
        "viet_small",
        "viet_small_cifar",
        "viet_tiny",
        "viet_tiny_cifar",
    }:
        from . import viet

        return getattr(viet, name)
    if name in {
        "VisionSimplicialTransformer",
        "viset_2l_cifar",
        "viset_4l_cifar",
        "viset_6l_cifar",
        "viset_base",
        "viset_large",
        "viset_small",
        "viset_small_cifar",
        "viset_tiny",
        "viset_tiny_cifar",
    }:
        from . import viset

        return getattr(viset, name)
    if name in {
        "VisionTransformer",
        "vit_base",
        "vit_large",
        "vit_small",
        "vit_small_cifar",
        "vit_tiny",
        "vit_tiny_cifar",
    }:
        from . import vit

        return getattr(vit, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
