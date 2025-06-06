"""Vision models for Energy Transformer.

This module provides three vision transformer variants, each building on the previous:

1. **VisionTransformer (ViT)** - Standard vision transformer baseline
2. **VisionEnergyTransformer (ViET)** - Energy-based with regular Hopfield networks

Model Naming Convention
-----------------------
- ViT: Standard Vision Transformer (baseline)
- ViET: Vision Energy Transformer

Configuration Sizes
-------------------
- Tiny: 192 dim, 3 heads (5.7M params)
- Small: 384 dim, 6 heads (22M params)
- Base: 768 dim, 12 heads (86M params)
- Large: 1024 dim, 16 heads (307M params)

CIFAR Configurations
--------------------

Example
-------
>>> # Standard ViT for ImageNet
>>> model = vit_base(img_size=224, patch_size=16, num_classes=1000)
>>>
>>> # Energy Transformer for CIFAR-100
>>> model = viet_2l_cifar(num_classes=100)
>>>
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
    "REALISER_REGISTRY",
    # Core models and registry
    "EnergyTransformer",
    # Vision Energy Transformer (ViET)
    "VisionEnergyTransformer",
    # Vision Transformer (ViT) - Baseline
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
    "vit_base",
    "vit_large",
    "vit_small",
    "vit_small_cifar",
    "vit_tiny",
    "vit_tiny_cifar",
]
