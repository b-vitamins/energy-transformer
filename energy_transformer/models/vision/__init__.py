"""Vision models for Energy Transformer.

This module provides three vision transformer variants:
1. VisionTransformer (ViT) - Standard vision transformer baseline
2. VisionEnergyTransformer (ViET) - Energy-based with regular Hopfield networks
3. VisionSimplicialEnergyTransformer (ViSET) - Energy-based with topology-aware simplicial networks
"""

from energy_transformer.models.base import REALISER_REGISTRY, EnergyTransformer

# Vision Energy Transformer
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

# Vision Simplicial Energy Transformer
from .viset import (
    VisionSimplicialEnergyTransformer,
    get_viset_name,
    viset_2l_e40_t40_tet20_cifar,
    viset_2l_e50_t50_cifar,
    viset_2l_e100_cifar,
    viset_2l_random_cifar,
    viset_2l_t100_cifar,
    viset_4l_e50_t50_cifar,
    viset_6l_e50_t50_cifar,
    viset_base,
    viset_small,
    viset_tiny,
)

# Standard Vision Transformer
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
    # Core models
    "EnergyTransformer",
    "REALISER_REGISTRY",
    # Vision Transformer (ViT)
    "VisionTransformer",
    "vit_tiny",
    "vit_small",
    "vit_base",
    "vit_large",
    "vit_tiny_cifar",
    "vit_small_cifar",
    # Vision Energy Transformer (ViET)
    "VisionEnergyTransformer",
    "viet_tiny",
    "viet_small",
    "viet_base",
    "viet_large",
    "viet_tiny_cifar",
    "viet_small_cifar",
    "viet_2l_cifar",
    "viet_4l_cifar",
    "viet_6l_cifar",
    # Vision Simplicial Energy Transformer (ViSET)
    "VisionSimplicialEnergyTransformer",
    "viset_tiny",
    "viset_small",
    "viset_base",
    "viset_2l_e50_t50_cifar",
    "viset_2l_e100_cifar",
    "viset_2l_t100_cifar",
    "viset_2l_random_cifar",
    "viset_4l_e50_t50_cifar",
    "viset_6l_e50_t50_cifar",
    "viset_2l_e40_t40_tet20_cifar",
    "get_viset_name",
]
