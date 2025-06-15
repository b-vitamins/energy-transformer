"""Vision Transformer (ViT) implementation.

This module provides a clean, minimal implementation of the Vision Transformer
architecture as described in "An Image is Worth 16x16 Words: Transformers for
Image Recognition at Scale" (Dosovitskiy et al., 2020).

The Vision Transformer treats an image as a sequence of patches and processes
them using a standard Transformer encoder, demonstrating that a pure transformer
architecture can achieve excellent results on image classification tasks without
the inductive biases inherent to CNNs.

Classes
-------
PatchEmbedding
    Converts images into sequences of patch embeddings
VisionTransformer
    Main ViT model implementation
TransformerBlock
    Standard transformer encoder block
Attention
    Multi-head self-attention mechanism
MLP
    Feedforward network with GELU activation

Factory Functions
-----------------
vit_tiny, vit_small, vit_base, vit_large
    Standard ViT configurations
vit_tiny_cifar, vit_small_cifar
    CIFAR-optimized configurations with 4x4 patches
    See [Dosovitskiy2020]_ for details on the Vision Transformer architecture.

Example
-------
>>> # Create a ViT-Tiny model for CIFAR-100
>>> model = vit_tiny_cifar(num_classes=100)
>>>
>>> # Process a batch of images
>>> images = torch.randn(32, 3, 32, 32)
>>> logits = model(images)  # Shape: (32, 100)

References
----------
.. [Dosovitskiy2020] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X.,
       Unterthiner, T., ... & Houlsby, N. (2020). An Image is Worth 16x16 Words:
       Transformers for Image Recognition at Scale. arXiv preprint arXiv:2010.11929.
"""

from __future__ import annotations

from typing import Any, cast

from ..configs import ViTConfig

__all__ = [
    "PatchEmbedding",
    "VisionTransformer",
    "vit_base",
    "vit_large",
    "vit_small",
    "vit_small_cifar",
    "vit_tiny",
    "vit_tiny_cifar",
]

import torch
from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
    ) -> None:
        """Initialize PatchEmbedding."""
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Convert image patches to embeddings.

        Parameters
        ----------
        x : Tensor
            Input images of shape (B, C, H, W).

        Returns
        -------
        Tensor
            Patch embeddings of shape (B, N, D).
        """
        x = self.proj(x)  # (B, D, H/P, W/P)
        return x.flatten(2).transpose(1, 2)  # (B, N, D)


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT).

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
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float
        Ratio of MLP hidden dim to embedding dim.
    qkv_bias : bool
        Whether to use bias in QKV projection.
    drop_rate : float
        Dropout rate.
    attn_drop_rate : float
        Attention dropout rate.
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
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ) -> None:
        """Initialize VisionTransformer."""
        super().__init__()

        # Store key config values for external access
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.depth = depth
        self.img_size = img_size

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + num_patches, embed_dim),
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for _ in range(depth)
            ],
        )

        # Final norm and classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input images of shape (B, C, H, W).

        Returns
        -------
        Tensor
            Logits of shape (B, num_classes).
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification
        x = self.norm(x)
        x = x[:, 0]  # CLS token

        return cast(Tensor, self.head(x))


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        """Initialize TransformerBlock."""
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply transformer block.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, N, D).

        Returns
        -------
        Tensor
            Output tensor of shape (B, N, D).
        """
        x = x + cast(Tensor, self.attn(self.norm1(x)))
        return x + cast(Tensor, self.mlp(self.norm2(x)))


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Initialize Attention."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        """Apply multi-head self-attention.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, N, D).

        Returns
        -------
        Tensor
            Output tensor of shape (B, N, D).
        """
        batch_size, seq_len, embed_dim = x.shape

        # QKV projection and reshape
        qkv = self.qkv(x).reshape(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D_h)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x = cast(Tensor, self.proj(x))
        return cast(Tensor, self.proj_drop(x))


class MLP(nn.Module):
    """MLP block with GELU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        """Initialize MLP."""
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """Apply MLP block.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """
        x = cast(Tensor, self.fc1(x))
        x = cast(Tensor, self.act(x))
        x = cast(Tensor, self.drop(x))
        x = cast(Tensor, self.fc2(x))
        return cast(Tensor, self.drop(x))


# Factory functions


def vit_tiny(**kwargs: Any) -> VisionTransformer:
    """ViT-Tiny configuration."""
    config = ViTConfig(
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        in_chans=3,
    )
    config.apply_overrides(**kwargs)
    return config.build()


def vit_small(**kwargs: Any) -> VisionTransformer:
    """ViT-Small configuration."""
    config = ViTConfig(
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        in_chans=3,
    )
    config.apply_overrides(**kwargs)
    return config.build()


def vit_base(**kwargs: Any) -> VisionTransformer:
    """ViT-Base configuration."""
    config = ViTConfig(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        in_chans=3,
    )
    config.apply_overrides(**kwargs)
    return config.build()


def vit_large(**kwargs: Any) -> VisionTransformer:
    """ViT-Large configuration."""
    config = ViTConfig(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        in_chans=3,
    )
    config.apply_overrides(**kwargs)
    return config.build()


# CIFAR-specific configurations


def vit_tiny_cifar(num_classes: int = 100, **kwargs: Any) -> VisionTransformer:
    """ViT-Tiny for CIFAR datasets."""
    config = ViTConfig(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        drop_rate=0.1,
    )
    config.apply_overrides(**kwargs)
    return config.build()


def vit_small_cifar(num_classes: int = 100, **kwargs: Any) -> VisionTransformer:
    """ViT-Small for CIFAR datasets."""
    config = ViTConfig(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        drop_rate=0.1,
    )
    config.apply_overrides(**kwargs)
    return config.build()
