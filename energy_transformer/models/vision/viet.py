"""Vision Energy Transformer (ViET).

Vision Transformer variant using energy-based components such as
:class:`MultiheadEnergyAttention` and :class:`HopfieldNetwork`.
It is based on the Energy Transformer architecture described in
[Hoover2023ViET]_.

References
----------
.. [Hoover2023ViET] Hoover, B., Liang, Y., Pham, B., Panda, R., Strobelt, H.,
   Chau, D. H., Zaki, M. J., & Krotov, D. (2023). *Energy Transformer*.
   arXiv preprint arXiv:2302.07253.
"""

from __future__ import annotations

from typing import Any, cast

from energy_transformer.models.configs import ViETConfig

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


class VisionEnergyTransformer(nn.Module):
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
    hopfield_hidden_dim : int
        Hidden dimension for Hopfield networks.
    et_steps : int
        Number of energy optimization steps per block.
    drop_rate : float
        Dropout rate.
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
        hopfield_hidden_dim: int,
        et_steps: int,
        drop_rate: float = 0.0,
    ) -> None:
        """Initialize :class:`VisionEnergyTransformer`."""
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

    @property
    def num_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Factory functions


def viet_tiny(**kwargs: Any) -> VisionEnergyTransformer:
    """ViET-Tiny configuration."""
    config = ViETConfig(
        embed_dim=192,
        depth=12,
        num_heads=3,
        hopfield_hidden_dim=768,
        et_steps=4,
        in_chans=3,
    )
    config.apply_overrides(**kwargs)
    return config.build()


def viet_small(**kwargs: Any) -> VisionEnergyTransformer:
    """ViET-Small configuration."""
    config = ViETConfig(
        embed_dim=384,
        depth=12,
        num_heads=6,
        hopfield_hidden_dim=1536,
        et_steps=4,
        in_chans=3,
    )
    config.apply_overrides(**kwargs)
    return config.build()


def viet_base(**kwargs: Any) -> VisionEnergyTransformer:
    """ViET-Base configuration."""
    config = ViETConfig(
        embed_dim=768,
        depth=12,
        num_heads=12,
        hopfield_hidden_dim=3072,
        et_steps=4,
        in_chans=3,
    )
    config.apply_overrides(**kwargs)
    return config.build()


def viet_large(**kwargs: Any) -> VisionEnergyTransformer:
    """ViET-Large configuration."""
    config = ViETConfig(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        hopfield_hidden_dim=4096,
        et_steps=4,
        in_chans=3,
    )
    config.apply_overrides(**kwargs)
    return config.build()


# CIFAR-specific configurations


def viet_tiny_cifar(
    num_classes: int = 100,
    **kwargs: Any,
) -> VisionEnergyTransformer:
    """ViET-Tiny for CIFAR datasets."""
    config = ViETConfig(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
        hopfield_hidden_dim=768,
        et_steps=4,
        drop_rate=0.1,
    )
    config.apply_overrides(**kwargs)
    return config.build()


def viet_small_cifar(
    num_classes: int = 100,
    **kwargs: Any,
) -> VisionEnergyTransformer:
    """ViET-Small for CIFAR datasets."""
    config = ViETConfig(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        hopfield_hidden_dim=1536,
        et_steps=4,
        drop_rate=0.1,
    )
    config.apply_overrides(**kwargs)
    return config.build()


# Shallow CIFAR configurations for testing


def viet_2l_cifar(
    num_classes: int = 100,
    **kwargs: Any,
) -> VisionEnergyTransformer:
    """Vision Energy Transformer with only 2 layers for CIFAR datasets."""
    config = ViETConfig(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=192,
        depth=2,
        num_heads=8,
        hopfield_hidden_dim=576,
        et_steps=6,
        drop_rate=0.1,
    )
    config.apply_overrides(**kwargs)
    return config.build()


def viet_4l_cifar(
    num_classes: int = 100,
    **kwargs: Any,
) -> VisionEnergyTransformer:
    """Vision Energy Transformer with 4 layers for CIFAR datasets."""
    config = ViETConfig(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=192,
        depth=4,
        num_heads=8,
        hopfield_hidden_dim=576,
        et_steps=5,
        drop_rate=0.1,
    )
    config.apply_overrides(**kwargs)
    return config.build()


def viet_6l_cifar(
    num_classes: int = 100,
    **kwargs: Any,
) -> VisionEnergyTransformer:
    """Vision Energy Transformer with 6 layers for CIFAR datasets."""
    config = ViETConfig(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=192,
        depth=6,
        num_heads=8,
        hopfield_hidden_dim=576,
        et_steps=4,
        drop_rate=0.1,
    )
    config.apply_overrides(**kwargs)
    return config.build()
