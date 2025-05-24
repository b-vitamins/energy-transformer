"""Embedding layers for Energy Transformer models."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "PatchEmbedding",
    "PositionalEmbedding2D",
]


def _to_pair(x: int | tuple[int, int]) -> tuple[int, int]:
    """Convert input to pair of integers.

    Parameters
    ----------
    x : int or tuple of int
        Single integer or tuple of two integers.

    Returns
    -------
    tuple of int
        Pair of integers (h, w).
    """
    if isinstance(x, tuple):
        return x
    return (x, x)


class PatchEmbedding(nn.Module):
    """Convert images to patch token sequences.

    Implements the standard Vision Transformer patch embedding using a
    convolutional layer to extract non-overlapping patches from input images
    and project them to the embedding dimension.

    Parameters
    ----------
    img_size : int or tuple of int, default=224
        Input image spatial size. If int, assumes square image.
    patch_size : int or tuple of int, default=16
        Size of each square patch. If int, assumes square patches.
    in_chans : int, default=3
        Number of input channels.
    embed_dim : int, default=768
        Output embedding dimension.
    bias : bool, default=True
        Whether to include bias in the projection layer.

    Attributes
    ----------
    img_size : tuple of int
        Input image size as (height, width).
    patch_size : tuple of int
        Patch size as (height, width).
    num_patches : int
        Total number of patches that will be extracted.
    proj : nn.Conv2d
        Convolutional projection layer.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
    ) -> None:
        """Initialize PatchEmbedding module.

        Parameters
        ----------
        img_size : int or tuple of int, default=224
            Input image spatial size. If int, assumes square image.
        patch_size : int or tuple of int, default=16
            Size of each square patch. If int, assumes square patches.
        in_chans : int, default=3
            Number of input channels.
        embed_dim : int, default=768
            Output embedding dimension.
        bias : bool, default=True
            Whether to include bias in the projection layer.
        """
        super().__init__()
        img_size = _to_pair(img_size)
        patch_size = _to_pair(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1]
        )

        # Use Conv2d for efficient patch extraction and projection
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Convert images to patch token sequences.

        Parameters
        ----------
        x : Tensor
            Input images of shape (B, C, H, W).

        Returns
        -------
        Tensor
            Patch embeddings of shape (B, N, D) where N is the number
            of patches and D is the embedding dimension.

        Raises
        ------
        AssertionError
            If input image size doesn't match expected size.
        """
        b, c, h, w = x.shape

        # Validate input dimensions
        assert h == self.img_size[0] and w == self.img_size[1], (
            f"Input size ({h}*{w}) doesn't match model "
            f"({self.img_size[0]}*{self.img_size[1]})"
        )

        # Extract patches and project to embedding space
        x = self.proj(x)  # (b, embed_dim, h//patch_h, w//patch_w)
        x = x.flatten(2)  # (b, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (b, num_patches, embed_dim)

        return x


class PositionalEmbedding2D(nn.Module):
    """Learnable 2D positional embeddings for image patches.

    Implements learnable positional embeddings that are added to patch
    embeddings to provide spatial position information to the model.

    Parameters
    ----------
    num_patches : int
        Number of patches in the sequence.
    embed_dim : int
        Embedding dimension of tokens.
    include_cls : bool, default=False
        Whether to include position for a CLS token at the beginning.

    Attributes
    ----------
    pos_embed : nn.Parameter
        Learnable positional embedding parameters of shape
        (1, seq_len, embed_dim) where seq_len accounts for CLS token if used.
    """

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        include_cls: bool = False,
        init_std: float = 0.02,
    ) -> None:
        """Initialize PositionalEmbedding2D module.

        Parameters
        ----------
        num_patches : int
            Number of patches in the sequence.
        embed_dim : int
            Embedding dimension of tokens.
        include_cls : bool, default=False
            Whether to include position for a CLS token at the beginning.
        init_std : float, default=0.02
            Standard deviation for parameter initialization.
        """
        super().__init__()
        seq_len = num_patches + (1 if include_cls else 0)
        self.init_std = init_std
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize positional embedding weights."""
        nn.init.trunc_normal_(self.pos_embed, std=self.init_std)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional embeddings to input tokens.

        Parameters
        ----------
        x : Tensor
            Token embeddings of shape (B, N, D).

        Returns
        -------
        Tensor
            Token embeddings with positional information added,
            shape (B, N, D).
        """
        return x + self.pos_embed.to(x.dtype)
