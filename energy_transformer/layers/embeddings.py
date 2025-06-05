"""Embedding layers for Energy Transformer models."""

from __future__ import annotations

import torch
from typing import cast

from einops import rearrange
from torch import Tensor, nn

__all__ = [
    "ConvPatchEmbed",
    "PatchifyEmbed",
    "PosEmbed2D",
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


class ConvPatchEmbed(nn.Module):  # type: ignore[misc]
    """Convolutional patch embedding layer.

    Extracts non-overlapping patches and projects them to embedding dimension
    in a single Conv2d operation. This is the standard Vision Transformer
    approach.

    Notes
    -----
    Use this when:
    - You need efficient patch extraction and projection
    - Working with standard classification tasks
    - You don't need to manipulate patches before projection
    - Following standard ViT implementations

    Parameters
    ----------
    img_size : int or tuple of int
        Input image size.
    patch_size : int or tuple of int
        Patch size.
    embed_dim : int
        Embedding dimension.
    in_chans : int, default=3
        Number of input image channels.
    norm_layer : nn.Module or None, default=None
        Normalization layer to apply after projection.
    flatten : bool, default=True
        Whether to flatten spatial dimensions.
    bias : bool, default=True
        Whether to use bias in projection layer.

    Attributes
    ----------
    img_size : tuple of int
        Input image size as (height, width).
    patch_size : tuple of int
        Patch size as (height, width).
    num_patches : int
        Total number of patches.
    flatten : bool
        Whether spatial dimensions are flattened.
    proj : nn.Conv2d
        Convolutional projection layer.
    norm : nn.Module
        Normalization layer.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        embed_dim: int,
        in_chans: int = 3,
        norm_layer: nn.Module | None = None,
        flatten: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        img_size = _to_pair(img_size)
        patch_size = _to_pair(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1]
        )
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        Tensor
            Output tensor of shape (B, N, D) if flatten=True,
            otherwise (B, D, H', W') where H'=H/patch_size.
        """
        b, c, h, w = x.shape
        assert h == self.img_size[0], (
            f"Input size height {h} doesn't match model {self.img_size[0]}"
        )
        assert w == self.img_size[1], (
            f"Input size width {w} doesn't match model {self.img_size[1]}"
        )

        x = self.proj(x)  # (B, D, H/p, W/p)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return cast(Tensor, self.norm(x))


class PatchifyEmbed(nn.Module):  # type: ignore[misc]
    """Two-stage patch embedding with separate patchification and projection.

    First reshapes image into patches preserving spatial structure, then
    projects flattened patches to embedding dimension. This approach allows
    patch manipulation before projection.

    Notes
    -----
    Use this when:
    - You need masked image modeling (can mask patches before projection)
    - Working with self-supervised pretraining
    - You want to visualize or manipulate individual patches
    - You need a reconstruction decoder (can invert the process)

    Parameters
    ----------
    img_size : int or tuple of int
        Input image size.
    patch_size : int or tuple of int
        Patch size.
    embed_dim : int
        Embedding dimension.
    in_chans : int, default=3
        Number of input image channels.
    norm_layer : nn.Module or None, default=None
        Normalization layer to apply after projection.
    bias : bool, default=True
        Whether to use bias in projection layer.

    Attributes
    ----------
    img_size : tuple of int
        Input image size as (height, width).
    patch_size : tuple of int
        Patch size as (height, width).
    grid_size : tuple of int
        Number of patches in (height, width) directions.
    num_patches : int
        Total number of patches.
    patch_shape : tuple of int
        Shape of each patch as (C, H, W).
    patch_dim : int
        Flattened dimension of each patch.
    proj : nn.Linear
        Linear projection layer.
    norm : nn.Module
        Normalization layer.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        embed_dim: int,
        in_chans: int = 3,
        norm_layer: nn.Module | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        img_size = _to_pair(img_size)
        patch_size = _to_pair(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_shape = (in_chans, patch_size[0], patch_size[1])
        self.patch_dim = in_chans * patch_size[0] * patch_size[1]

        self.proj = nn.Linear(self.patch_dim, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def patchify(self, x: Tensor) -> Tensor:
        """Extract patches from images.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        Tensor
            Patches of shape (B, N, C, pH, pW).
        """
        b, c, h, w = x.shape
        assert h == self.img_size[0], (
            f"Input size height {h} doesn't match model {self.img_size[0]}"
        )
        assert w == self.img_size[1], (
            f"Input size width {w} doesn't match model {self.img_size[1]}"
        )

        ph, pw = self.patch_size
        return rearrange(
            x,
            "b c (gh ph) (gw pw) -> b (gh gw) c ph pw",
            gh=self.grid_size[0],
            gw=self.grid_size[1],
            ph=ph,
            pw=pw,
        )

    def unpatchify(self, patches: Tensor) -> Tensor:
        """Reconstruct images from patches.

        Parameters
        ----------
        patches : Tensor
            Patches of shape (B, N, C, pH, pW).

        Returns
        -------
        Tensor
            Reconstructed images of shape (B, C, H, W).
        """
        return rearrange(
            patches,
            "b (gh gw) c ph pw -> b c (gh ph) (gw pw)",
            gh=self.grid_size[0],
            gw=self.grid_size[1],
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        Tensor
            Output tensor of shape (B, N, D).
        """
        x = self.patchify(x)  # (B, N, C, pH, pW)
        x = rearrange(x, "b n c h w -> b n (c h w)")  # (B, N, patch_dim)
        x = self.proj(x)  # (B, N, D)
        return cast(Tensor, self.norm(x))


class PosEmbed2D(nn.Module):  # type: ignore[misc]
    """Learnable 2D positional embeddings.

    Supports both batched and unbatched inputs by storing positional
    embeddings without batch dimension in the parameters.

    Parameters
    ----------
    num_patches : int
        Number of patches in the sequence.
    embed_dim : int
        Embedding dimension.
    cls_token : bool, default=False
        Whether to include position for a class token.
    dropout : float, default=0.0
        Dropout rate to apply after adding positional embeddings.

    Attributes
    ----------
    num_patches : int
        Number of patches.
    embed_dim : int
        Embedding dimension.
    cls_token : bool
        Whether class token position is included.
    pos_embed : nn.Parameter
        Positional embedding parameters of shape (L, D) where
        L = num_patches + 1 if cls_token else num_patches.
    pos_drop : nn.Dropout
        Dropout layer.
    """

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        cls_token: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.cls_token = cls_token

        length = num_patches + (1 if cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(length, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, L, D) or (L, D).

        Returns
        -------
        Tensor
            Output tensor with positional embeddings added, same shape as input.

        Raises
        ------
        ValueError
            If input has unexpected number of dimensions.
        RuntimeError
            If sequence length doesn't match positional embedding length.
        """
        if x.ndim == 3:  # (B, L, D)  # noqa: PLR2004
            if x.size(1) != self.pos_embed.size(0):
                raise RuntimeError(
                    f"Sequence length {x.size(1)} doesn't match "
                    f"positional embedding length {self.pos_embed.size(0)}."
                )
            x = x + self.pos_embed.unsqueeze(0).to(x.dtype)
        elif x.ndim == 2:  # (L, D)  # noqa: PLR2004
            if x.size(0) != self.pos_embed.size(0):
                raise RuntimeError(
                    f"Sequence length {x.size(0)} doesn't match "
                    f"positional embedding length {self.pos_embed.size(0)}."
                )
            x = x + self.pos_embed.to(x.dtype)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.ndim}D.")

        return cast(Tensor, self.pos_drop(x))
