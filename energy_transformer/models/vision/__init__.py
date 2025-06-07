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

_ATTR_TO_MODULE = {
    "EnergyTransformer": "energy_transformer.models.base",
    "VisionEnergyTransformer": ".viet",
    "viet_2l_cifar": ".viet",
    "viet_4l_cifar": ".viet",
    "viet_6l_cifar": ".viet",
    "viet_base": ".viet",
    "viet_large": ".viet",
    "viet_small": ".viet",
    "viet_small_cifar": ".viet",
    "viet_tiny": ".viet",
    "viet_tiny_cifar": ".viet",
    "VisionSimplicialTransformer": ".viset",
    "viset_2l_cifar": ".viset",
    "viset_4l_cifar": ".viset",
    "viset_6l_cifar": ".viset",
    "viset_base": ".viset",
    "viset_large": ".viset",
    "viset_small": ".viset",
    "viset_small_cifar": ".viset",
    "viset_tiny": ".viset",
    "viset_tiny_cifar": ".viset",
    "VisionTransformer": ".vit",
    "vit_base": ".vit",
    "vit_large": ".vit",
    "vit_small": ".vit",
    "vit_small_cifar": ".vit",
    "vit_tiny": ".vit",
    "vit_tiny_cifar": ".vit",
}


def __getattr__(name: str) -> object:
    """Lazily resolve vision model attributes.

    Parameters
    ----------
    name : str
        Attribute name to import from a submodule.

    Returns
    -------
    object
        The requested attribute.

    Raises
    ------
    AttributeError
        If ``name`` is not recognized.
    """
    module_name = _ATTR_TO_MODULE.get(name)
    if module_name is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    from importlib import import_module

    module = import_module(
        f"{__name__}{module_name}"
        if module_name.startswith(".")
        else module_name
    )
    attr = getattr(module, name)
    globals()[name] = attr
    return attr
