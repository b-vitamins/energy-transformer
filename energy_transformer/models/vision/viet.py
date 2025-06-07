"""Vision Energy Transformer (ViET) implementation.

This module implements the Vision Energy Transformer, which replaces standard
transformer components with energy-based alternatives as described in
"Energy Transformer" (Hoover et al., 2023).

from __future__ import annotations

Key Differences from Standard ViT
---------------------------------
- Multi-Head Energy Attention instead of standard self-attention
- Hopfield Networks instead of MLP blocks
- Energy-based LayerNorm with learnable temperature
- Iterative refinement through gradient descent on energy landscape

Classes
-------
VisionEnergyTransformer
    Main ViET model that orchestrates energy-based components

Factory Functions
-----------------
viet_tiny, viet_small, viet_base, viet_large
    Standard configurations matching ViT sizes
viet_tiny_cifar, viet_small_cifar
    CIFAR-optimized configurations
viet_2l_cifar, viet_4l_cifar, viet_6l_cifar
    Shallow variants for computational efficiency

Example
-------
>>> # Create a 2-layer ViET for CIFAR-100
>>> model = viet_2l_cifar(num_classes=100)
>>>
>>> # Forward pass with energy information
>>> images = torch.randn(32, 3, 32, 32)
>>> result = model(images, return_energy_info=True)
>>> logits = result['logits']
>>> energy_trajectory = result['energy_info']['block_energies']

Parameters
----------
The key hyperparameters for ViET are:
- `et_steps`: Number of energy minimization steps (default: 4-6)
- `hopfield_hidden_dim`: Hidden dimension for Hopfield networks

References
----------
.. [1] Hoover, B., Liang, Y., Pham, B., Panda, R., Strobelt, H., Chau, D. H.,
       Zaki, M. J., & Krotov, D. (2023). Energy Transformer.
       arXiv preprint arXiv:2302.07253.
"""

from __future__ import annotations

from typing import Any, cast

__all__ = [
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
]

import torch
from torch import Tensor, nn

from energy_transformer.layers.attention import MultiheadEnergyAttention
from energy_transformer.layers.embeddings import ConvPatchEmbed, PosEmbed2D
from energy_transformer.layers.heads import ClassifierHead
from energy_transformer.layers.hopfield import HopfieldNetwork
from energy_transformer.layers.layer_norm import EnergyLayerNorm
from energy_transformer.layers.validation import validate_shape_match
from energy_transformer.models.base import EnergyTransformer


class VisionEnergyTransformer(nn.Module):  # type: ignore[misc]
    """Vision Energy Transformer (ViET).

    A Vision Transformer that replaces standard components with energy-based
    alternatives:
    - Standard Attention → Multi-Head Energy Attention
    - Standard LayerNorm → Energy-based LayerNorm
    - MLP → Hopfield Network
    - Feedforward computation → Energy minimization

    Parameters
    ----------
    img_size : int
        Input image size (assumed square).
    patch_size : int
        Size of image patches (assumed square).
    in_chans : int
        Number of input channels.
    num_classes : int
        Number of output classes.
    embed_dim : int
        Embedding dimension.
    depth : int
        Number of Energy Transformer blocks.
    num_heads : int
        Number of attention heads.
    head_dim : int
        Dimension of each attention head.
    hopfield_hidden_dim : int
        Hidden dimension for Hopfield networks.
    et_steps : int
        Number of energy optimization steps per block.
    drop_rate : float
        Dropout rate.
    representation_size : int | None
        Size of representation layer before classification head.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        _head_dim: int,
        hopfield_hidden_dim: int,
        et_steps: int,
        drop_rate: float = 0.0,
        _representation_size: int | None = None,
    ) -> None:
        """Initialize VisionImageTransformer."""
        super().__init__()

        # Store configuration
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_classes = num_classes

        # Patch embedding
        self.patch_embed = ConvPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # CLS token parameter
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings (include CLS token)
        self.pos_embed = PosEmbed2D(
            num_patches=num_patches,
            embed_dim=embed_dim,
            cls_token=True,
        )

        # Positional dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Energy Transformer blocks
        self.et_blocks = nn.ModuleList(
            [
                EnergyTransformer(
                    layer_norm=EnergyLayerNorm(embed_dim),
                    attention=MultiheadEnergyAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                    ),
                    hopfield=HopfieldNetwork(
                        embed_dim,
                        hidden_dim=hopfield_hidden_dim,
                    ),
                    steps=et_steps,
                )
                for _ in range(depth)
            ],
        )

        # Final layer normalization
        self.norm = EnergyLayerNorm(embed_dim)

        # Classification head
        self.head = ClassifierHead(
            in_features=embed_dim,
            num_classes=num_classes,
            pool_type="token",
            drop_rate=drop_rate,
        )

        # Initialize special tokens
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed.pos_embed, std=0.02)

    def forward(
        self,
        x: Tensor,
        return_energies: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        """Forward pass through Vision Energy Transformer.

        Parameters
        ----------
        x : Tensor
            Input images of shape (B, C, H, W).
        return_energies : bool
            If True, return averaged energies across blocks.

        Returns
        -------
        Tensor | tuple[Tensor, tuple[Tensor, Tensor]]
            If return_energies is False: logits of shape (B, num_classes)
            If return_energies is True: (logits, (avg_e_att, avg_e_hop))
        """
        validate_shape_match(
            x,
            (-1, -1, self.img_size, self.img_size),
            self.__class__.__name__,
            dims_to_check=(2, 3),
        )

        # 1. Patch embedding
        x = self.patch_embed(x)  # (B, N, D)

        # 2. Prepend CLS token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # 3. Add positional embeddings
        x = self.pos_embed(x)  # (B, N+1, D)
        x = self.pos_drop(x)

        # 4. Energy Transformer blocks
        all_energies = []
        for et_block in self.et_blocks:
            if return_energies:
                x, energies = et_block(x, return_energies=True)
                if energies:
                    all_energies.append(energies[0])
            else:
                x = et_block(x)

        # 5. Final layer normalization
        x = self.norm(x)  # (B, N+1, D)

        # 6. Classification
        logits = cast(Tensor, self.head(x))  # (B, num_classes)

        # 7. Return based on flag
        if return_energies and all_energies:
            avg_e_att = torch.stack([e[0] for e in all_energies]).mean()
            avg_e_hop = torch.stack([e[1] for e in all_energies]).mean()
            return logits, (avg_e_att, avg_e_hop)

        return logits


# Factory functions


def viet_tiny(**kwargs: Any) -> VisionEnergyTransformer:
    """ViET-Tiny configuration."""
    config: dict[str, Any] = {
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "_head_dim": 64,
        "hopfield_hidden_dim": 768,  # 4x embed_dim
        "et_steps": 4,
        "in_chans": 3,
    }
    config.update(kwargs)
    return VisionEnergyTransformer(**config)


def viet_small(**kwargs: Any) -> VisionEnergyTransformer:
    """ViET-Small configuration."""
    config: dict[str, Any] = {
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "_head_dim": 64,
        "hopfield_hidden_dim": 1536,  # 4x embed_dim
        "et_steps": 4,
        "in_chans": 3,
    }
    config.update(kwargs)
    return VisionEnergyTransformer(**config)


def viet_base(**kwargs: Any) -> VisionEnergyTransformer:
    """ViET-Base configuration."""
    config: dict[str, Any] = {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "_head_dim": 64,
        "hopfield_hidden_dim": 3072,  # 4x embed_dim
        "et_steps": 4,
        "in_chans": 3,
    }
    config.update(kwargs)
    return VisionEnergyTransformer(**config)


def viet_large(**kwargs: Any) -> VisionEnergyTransformer:
    """ViET-Large configuration."""
    config: dict[str, Any] = {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "_head_dim": 64,
        "hopfield_hidden_dim": 4096,  # 4x embed_dim
        "et_steps": 4,
        "in_chans": 3,
    }
    config.update(kwargs)
    return VisionEnergyTransformer(**config)


# CIFAR-specific configurations


def viet_tiny_cifar(
    num_classes: int = 100,
    **kwargs: Any,
) -> VisionEnergyTransformer:
    """ViET-Tiny for CIFAR datasets."""
    config: dict[str, Any] = {
        "img_size": 32,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": num_classes,
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "_head_dim": 64,
        "hopfield_hidden_dim": 768,
        "et_steps": 4,
        "drop_rate": 0.1,
    }
    config.update(kwargs)
    return VisionEnergyTransformer(**config)


def viet_small_cifar(
    num_classes: int = 100,
    **kwargs: Any,
) -> VisionEnergyTransformer:
    """ViET-Small for CIFAR datasets."""
    config: dict[str, Any] = {
        "img_size": 32,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": num_classes,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "_head_dim": 64,
        "hopfield_hidden_dim": 1536,
        "et_steps": 4,
        "drop_rate": 0.1,
    }
    config.update(kwargs)
    return VisionEnergyTransformer(**config)


# Shallow CIFAR configurations for testing


def viet_2l_cifar(
    num_classes: int = 100,
    **kwargs: Any,
) -> VisionEnergyTransformer:
    """Vision Energy Transformer with only 2 layers for CIFAR datasets."""
    config: dict[str, Any] = {
        "img_size": 32,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": num_classes,
        "embed_dim": 192,
        "depth": 2,  # Shallow!
        "num_heads": 8,
        "_head_dim": 64,
        "hopfield_hidden_dim": 576,  # 3x embed_dim
        "et_steps": 6,
        "drop_rate": 0.1,
    }
    config.update(kwargs)
    return VisionEnergyTransformer(**config)


def viet_4l_cifar(
    num_classes: int = 100,
    **kwargs: Any,
) -> VisionEnergyTransformer:
    """Vision Energy Transformer with 4 layers for CIFAR datasets."""
    config: dict[str, Any] = {
        "img_size": 32,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": num_classes,
        "embed_dim": 192,
        "depth": 4,
        "num_heads": 8,
        "_head_dim": 64,
        "hopfield_hidden_dim": 576,
        "et_steps": 5,
        "drop_rate": 0.1,
    }
    config.update(kwargs)
    return VisionEnergyTransformer(**config)


def viet_6l_cifar(
    num_classes: int = 100,
    **kwargs: Any,
) -> VisionEnergyTransformer:
    """Vision Energy Transformer with 6 layers for CIFAR datasets."""
    config: dict[str, Any] = {
        "img_size": 32,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": num_classes,
        "embed_dim": 192,
        "depth": 6,
        "num_heads": 8,
        "_head_dim": 64,
        "hopfield_hidden_dim": 576,
        "et_steps": 4,
        "drop_rate": 0.1,
    }
    config.update(kwargs)
    return VisionEnergyTransformer(**config)
