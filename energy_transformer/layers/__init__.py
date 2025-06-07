from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .attention import MultiheadEnergyAttention
    from .embeddings import ConvPatchEmbed, PatchifyEmbed, PosEmbed2D
    from .heads import (
        ClassifierHead,
        LinearClassifierHead,
        NormLinearClassifierHead,
        NormMLPClassifierHead,
        ReLUMLPClassifierHead,
    )
    from .hopfield import HopfieldNetwork
    from .layer_norm import EnergyLayerNorm
    from .mlp import MLP
    from .simplicial import SimplicialHopfieldNetwork

__all__ = [
    "MLP",
    "ClassifierHead",
    "ConvPatchEmbed",
    "EnergyLayerNorm",
    "HopfieldNetwork",
    "LinearClassifierHead",
    "MultiheadEnergyAttention",
    "NormLinearClassifierHead",
    "NormMLPClassifierHead",
    "PatchifyEmbed",
    "PosEmbed2D",
    "ReLUMLPClassifierHead",
    "SimplicialHopfieldNetwork",
]

_ATTR_TO_MODULE = {
    "MLP": ".mlp",
    "ClassifierHead": ".heads",
    "ConvPatchEmbed": ".embeddings",
    "EnergyLayerNorm": ".layer_norm",
    "HopfieldNetwork": ".hopfield",
    "LinearClassifierHead": ".heads",
    "MultiheadEnergyAttention": ".attention",
    "NormLinearClassifierHead": ".heads",
    "NormMLPClassifierHead": ".heads",
    "PatchifyEmbed": ".embeddings",
    "PosEmbed2D": ".embeddings",
    "ReLUMLPClassifierHead": ".heads",
    "SimplicialHopfieldNetwork": ".simplicial",
}


def __getattr__(name: str) -> object:
    module_name = _ATTR_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(f"{__name__}{module_name}")
    attr = getattr(module, name)
    globals()[name] = attr
    return attr
