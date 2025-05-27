"""Vision models for Energy Transformer.

This module provides three vision transformer variants, each building on the previous:

1. **VisionTransformer (ViT)** - Standard vision transformer baseline
2. **VisionEnergyTransformer (ViET)** - Energy-based with regular Hopfield networks
3. **VisionSimplicialEnergyTransformer (ViSET)** - Energy-based with topology-aware simplicial networks

Model Naming Convention
-----------------------
- ViT: Standard Vision Transformer (baseline)
- ViET: Vision Energy Transformer
- ViSET: Vision Simplicial Energy Transformer

Configuration Sizes
-------------------
- Tiny: 192 dim, 3 heads (5.7M params)
- Small: 384 dim, 6 heads (22M params)
- Base: 768 dim, 12 heads (86M params)
- Large: 1024 dim, 16 heads (307M params)

CIFAR Configurations
--------------------
Special configurations optimized for 32x32 images with 4x4 patches:
- 2L, 4L, 6L: Shallow variants with fewer layers
- E50-T50: 50% edges, 50% triangles (ViSET only)
- E100: 100% edges from k-NN graph (ViSET only)
- T100: 100% triangles from Delaunay (ViSET only)
- Random: Random topology baseline (ViSET only)

Example
-------
>>> # Standard ViT for ImageNet
>>> model = vit_base(img_size=224, patch_size=16, num_classes=1000)
>>>
>>> # Energy Transformer for CIFAR-100
>>> model = viet_2l_cifar(num_classes=100)
>>>
>>> # Simplicial Energy Transformer with topology awareness
>>> model = viset_2l_e50_t50_cifar(num_classes=100)
"""

# Re-export base model and registry for convenience
from energy_transformer.models.base import REALISER_REGISTRY, EnergyTransformer

# Vision Energy Transformer
from .viet import (
    VisionEnergyTransformer,
    # CIFAR configurations
    viet_2l_cifar,
    viet_4l_cifar,
    viet_6l_cifar,
    # Standard configurations
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
    # Utility
    get_viset_name,
    viset_2l_e40_t40_tet20_cifar,
    viset_2l_e50_t50_cifar,
    # CIFAR configurations - topology variants
    viset_2l_e100_cifar,
    viset_2l_random_cifar,
    viset_2l_t100_cifar,
    viset_4l_e50_t50_cifar,
    viset_6l_e50_t50_cifar,
    # Standard configurations
    viset_base,
    viset_small,
    viset_tiny,
)

# Standard Vision Transformer (baseline)
from .vit import (
    VisionTransformer,
    # Standard configurations
    vit_base,
    vit_large,
    vit_small,
    # CIFAR configurations
    vit_small_cifar,
    vit_tiny,
    vit_tiny_cifar,
)

__all__ = [
    # Core models and registry
    "EnergyTransformer",
    "REALISER_REGISTRY",
    # Vision Transformer (ViT) - Baseline
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
