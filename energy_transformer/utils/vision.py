"""Vision utilities for Energy Transformer models."""

import math
import warnings

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor

# Import spec classes to avoid circular imports
from energy_transformer.spec.primitives import (
    CLSTokenSpec,
    MaskTokenSpec,
    PatchSpec,
    PosEncSpec,
)

__all__ = [
    "PatchEmbed",
    "Learnable2DPosEnc",
    "SinCos2DPosEnc",
    "CLSToken",
    "MaskToken",
    "get_2d_sincos_pos_embed",
    # Add spec classes to export
    "PatchSpec",
    "PosEncSpec",
    "CLSTokenSpec",
    "MaskTokenSpec",
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


def _init_trunc_normal(param: nn.Parameter, std: float = 0.02) -> nn.Parameter:
    """Initialize parameter with truncated normal distribution.

    Parameters
    ----------
    param : nn.Parameter
        Parameter tensor to initialize.
    std : float, default=0.02
        Standard deviation for truncated normal distribution.

    Returns
    -------
    nn.Parameter
        The initialized parameter (same object as input).
    """
    torch.nn.init.trunc_normal_(param, std=std)
    return param


class PatchEmbed(nn.Module):  # type: ignore
    """Image to patch sequence embedder using patch extraction and linear projection.

    Converts input images into sequences of embedded patches by extracting
    non-overlapping patches and applying a shared linear transformation.
    Supports both fixed and variable resolution inputs.

    Parameters
    ----------
    img_size : int or tuple of int, default=32
        Input image spatial size. If int, assumes square image.
    patch_size : int or tuple of int, default=4
        Size of each square patch. If int, assumes square patches.
    in_chans : int, default=3
        Number of input channels.
    embed_dim : int, default=256
        Output embedding dimension (token dimension D).
    flatten : bool, default=True
        If True, returns shape (B, N, D); else (B, D, H', W').

    Attributes
    ----------
    patch_size : tuple of int
        Size of patches as (height, width).
    embed_dim : int
        Embedding dimension.
    num_patches : int
        Total patch count N = (H/patch_h) * (W/patch_w).
        Updated dynamically on first forward pass for variable resolution
        support.
    flatten : bool
        Whether to flatten output to sequence format.
    patch_dim : int
        Dimension of each flattened patch (in_chans * patch_h * patch_w).
    proj : nn.Linear
        Linear layer for projecting flattened patches to embedding space.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 32,
        patch_size: int | tuple[int, int] = 4,
        in_chans: int = 3,
        embed_dim: int = 256,
        flatten: bool = True,
    ) -> None:
        """Initialize PatchEmbed module.

        Parameters
        ----------
        img_size : int or tuple of int, default=32
            Input image spatial size. If int, assumes square image.
        patch_size : int or tuple of int, default=4
            Size of each square patch. If int, assumes square patches.
        in_chans : int, default=3
            Number of input channels.
        embed_dim : int, default=256
            Output embedding dimension (token dimension D).
        flatten : bool, default=True
            If True, returns shape (B, N, D); else (B, D, H', W').
        """
        super().__init__()
        img_h, img_w = _to_pair(img_size)
        patch_h, patch_w = _to_pair(patch_size)

        self.patch_size: tuple[int, int] = (patch_h, patch_w)
        self.embed_dim: int = embed_dim
        self.num_patches: int = (img_h // patch_h) * (img_w // patch_w)
        self.flatten: bool = flatten
        self._dynamic_resolution: bool = False

        # Calculate patch dimension (flattened patch size)
        self.patch_dim = in_chans * patch_h * patch_w

        # Linear projection for embedding patches
        self.proj: nn.Linear = nn.Linear(self.patch_dim, embed_dim)

        # Initialize weights with truncated normal
        _init_trunc_normal(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for patch embedding.

        Extracts non-overlapping patches from input images and projects them
        to embedding space using a shared linear transformation.

        Parameters
        ----------
        x : Tensor
            Input images of shape (B, C, H, W).

        Returns
        -------
        Tensor
            Embedded patches, either of shape:
            - (B, N, D) if flatten=True
            - (B, D, H', W') if flatten=False
            where N is the number of patches, D is embed_dim,
            and H', W' are the height and width after patch division.
        """
        batch_dim, channels, height, width = x.shape

        # Ensure module parameters match input dtype for mixed precision compatibility
        if x.dtype != self.proj.weight.dtype:
            self.proj = self.proj.to(dtype=x.dtype)

        # Extract patches using unfold
        patch_h, patch_w = self.patch_size

        # Use unfold to extract non-overlapping patches
        patches = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        # patches shape: [B, C, num_patches_h, num_patches_w, patch_h, patch_w]

        # Reshape to [B, num_patches, patch_dim]
        num_patches_h = height // patch_h
        num_patches_w = width // patch_w
        patches = patches.contiguous().view(
            batch_dim, channels, num_patches_h, num_patches_w, -1
        )
        patches = patches.permute(0, 2, 3, 1, 4).contiguous()
        patches = patches.view(batch_dim, num_patches_h * num_patches_w, -1)

        # Update num_patches dynamically on first forward pass for true
        # variable resolution support
        if not self._dynamic_resolution:
            self.num_patches = patches.shape[1]
            self._dynamic_resolution = True

        # Apply linear projection: [B, N, patch_dim] -> [B, N, embed_dim]
        x = self.proj(patches)

        if not self.flatten:
            # Reshape back to spatial format: [B, N, D] -> [B, D, H', W']
            x = x.transpose(1, 2)  # [B, D, N]
            x = x.view(batch_dim, self.embed_dim, num_patches_h, num_patches_w)

        return x


class Learnable2DPosEnc(nn.Module):  # type: ignore
    """Learned 2D absolute positional embeddings.

    Implements learnable positional embeddings that are added to input token
    embeddings to provide positional information.

    Parameters
    ----------
    num_tokens : int
        Number of tokens in the sequence.
    embed_dim : int
        Embedding dimension of tokens.

    Attributes
    ----------
    pos : nn.Parameter
        Learnable positional embedding parameters of shape
        (1, num_tokens, embed_dim).
    """

    def __init__(self, num_tokens: int, embed_dim: int) -> None:
        """Initialize Learnable2DPosEnc module.

        Parameters
        ----------
        num_tokens : int
            Number of tokens in the sequence.
        embed_dim : int
            Embedding dimension of tokens.
        """
        super().__init__()
        self.pos: nn.Parameter = nn.Parameter(
            torch.zeros(1, num_tokens, embed_dim)
        )
        _init_trunc_normal(self.pos)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional embeddings to input tokens.

        Parameters
        ----------
        x : Tensor
            Token embeddings of shape (B, N, D).

        Returns
        -------
        Tensor
            Token embeddings with added positional information,
            shape (B, N, D).
        """
        # Ensure correct dtype for mixed precision compatibility
        return x + self.pos.to(x.dtype)


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, include_cls: bool = True
) -> NDArray[np.float32]:
    """Generate 2D sinusoidal positional embeddings.

    Creates sinusoidal positional embeddings for a 2D grid of positions,
    following the pattern used in MAE/BEiT models.

    Parameters
    ----------
    embed_dim : int
        Output dimension for each position. Must be divisible by 4.
    grid_size : int
        Number of positions in each spatial dimension (height and width).
    include_cls : bool, default=True
        Whether to include an additional position for a CLS token at the
        beginning.

    Returns
    -------
    NDArray[np.float32]
        Positional embeddings of shape
        (grid_size*grid_size + include_cls, embed_dim).
        If include_cls is True, the first row is all zeros for the CLS token.

    Raises
    ------
    AssertionError
        If embed_dim is not divisible by 4.
    """
    # Ensure embed_dim is divisible by 4
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"

    # Compute dimension for sin and cos
    omega = np.arange(embed_dim // 4, dtype=np.float32) / (embed_dim // 4 - 1)
    omega = 1.0 / (10000**omega)  # (D/4,)

    # Create position grid
    pos_h = np.arange(grid_size, dtype=np.float32)  # (H,)
    pos_w = np.arange(grid_size, dtype=np.float32)  # (W,)

    # Outer product to get 2D coordinates
    pos_grid = np.stack(np.meshgrid(pos_w, pos_h), axis=0)  # (2, H, W)
    pos_grid = pos_grid.reshape(2, -1)  # (2, H*W)
    pos_grid = pos_grid.transpose(1, 0)  # (H*W, 2)

    # Apply standard BEiT/MAE sin/cos pattern to both spatial dimensions
    emb = np.zeros((pos_grid.shape[0], embed_dim), dtype=np.float32)
    emb[:, 0::4] = np.sin(pos_grid[:, 0:1] * omega)
    emb[:, 1::4] = np.cos(pos_grid[:, 0:1] * omega)
    emb[:, 2::4] = np.sin(pos_grid[:, 1:2] * omega)
    emb[:, 3::4] = np.cos(pos_grid[:, 1:2] * omega)

    # Add class token if needed
    if include_cls:
        # Add a zero vector for cls token
        emb = np.concatenate(
            [np.zeros((1, embed_dim), dtype=np.float32), emb], axis=0
        )

    return emb


class SinCos2DPosEnc(nn.Module):  # type: ignore
    """Sinusoidal 2D positional encoding.

    Implements fixed sinusoidal positional embeddings for 2D spatial positions.
    The embeddings are pre-computed and stored as a non-persistent buffer.

    Parameters
    ----------
    num_tokens : int
        Total number of tokens in the sequence.
    embed_dim : int
        Embedding dimension of tokens.
    include_cls : bool, default=True
        Whether to include a CLS token position at the beginning.

    Attributes
    ----------
    pos_embed : Tensor
        Pre-computed sinusoidal positional embeddings stored as a buffer.

    Warnings
    --------
    UserWarning
        If the number of grid tokens is not a perfect square, a warning is
        issued and the grid size is adjusted using ceiling division.
    """

    def __init__(
        self, num_tokens: int, embed_dim: int, include_cls: bool = True
    ) -> None:
        """Initialize SinCos2DPosEnc module.

        Parameters
        ----------
        num_tokens : int
            Total number of tokens in the sequence.
        embed_dim : int
            Embedding dimension of tokens.
        include_cls : bool, default=True
            Whether to include a CLS token position at the beginning.
        """
        super().__init__()

        # Adjust for CLS token when calculating grid size
        grid_tokens = num_tokens
        if include_cls:
            grid_tokens -= 1

        # Check if grid_tokens is a perfect square
        grid_size = int(math.sqrt(grid_tokens))
        is_perfect_square = grid_size * grid_size == grid_tokens

        if not is_perfect_square:
            # For non-square grids, use ceiling and slice
            grid_size = math.ceil(math.sqrt(grid_tokens))
            warnings.warn(
                f"grid_tokens {grid_tokens} is not a perfect square. "
                f"Using grid_size {grid_size}.",
                stacklevel=2,
            )

        # Pre-compute embeddings using the utility function
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim, grid_size, include_cls=include_cls
        )

        # If not perfect square, slice to the exact number of tokens needed
        if not is_perfect_square:
            if include_cls:
                # Keep cls token (at index 0) and slice the rest
                pos_embed = np.concatenate(
                    [
                        pos_embed[:1],  # CLS token position
                        pos_embed[
                            1 : grid_tokens + 1
                        ],  # Exactly the number of grid tokens needed
                    ]
                )
            else:
                pos_embed = pos_embed[:grid_tokens]

        pos_embed_tensor = torch.from_numpy(pos_embed).float()

        # Register as a non-persistent buffer
        self.register_buffer("pos_embed", pos_embed_tensor, persistent=False)
        self.pos_embed: Tensor  # Type hint for mypy

    def forward(self, x: Tensor) -> Tensor:
        """Add positional embeddings to input tokens.

        Parameters
        ----------
        x : Tensor
            Token embeddings of shape (B, N, D).

        Returns
        -------
        Tensor
            Token embeddings with added positional information,
            shape (B, N, D).
        """
        # Ensure correct dtype for mixed precision compatibility
        return x + self.pos_embed.to(x.dtype)


class CLSToken(nn.Module):  # type: ignore
    """Global representation token to prepend to sequence.

    Implements a learnable CLS (classification) token that is prepended to
    the beginning of token sequences to provide a global representation.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension of the CLS token.

    Attributes
    ----------
    cls_token : nn.Parameter
        Learnable CLS token parameter of shape (1, 1, embed_dim).
    """

    def __init__(self, embed_dim: int) -> None:
        """Initialize CLSToken module.

        Parameters
        ----------
        embed_dim : int
            Embedding dimension of the CLS token.
        """
        super().__init__()
        self.cls_token: nn.Parameter = _init_trunc_normal(
            nn.Parameter(torch.zeros(1, 1, embed_dim))
        )

    def forward(self, x: Tensor) -> Tensor:
        """Prepend CLS token to input sequence.

        Parameters
        ----------
        x : Tensor
            Token embeddings of shape (B, N, D).

        Returns
        -------
        Tensor
            Token embeddings with CLS token prepended, shape (B, N+1, D).
        """
        batch_dim, _, _ = x.shape
        # Ensure correct dtype for mixed precision compatibility
        cls_tokens = self.cls_token.to(dtype=x.dtype).expand(
            batch_dim, -1, -1
        )  # shape: [B, 1, D]
        return torch.cat([cls_tokens, x], dim=1)  # shape: [B, N+1, D]


class MaskToken(nn.Module):  # type: ignore
    """Token used to replace masked patches.

    Implements a learnable mask token that replaces original tokens at
    positions indicated by a binary mask, commonly used in masked language
    modeling or masked image modeling tasks.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension of the mask token.

    Attributes
    ----------
    mask_token : nn.Parameter
        Learnable mask token parameter of shape (1, 1, embed_dim).
    """

    def __init__(self, embed_dim: int) -> None:
        """Initialize MaskToken module.

        Parameters
        ----------
        embed_dim : int
            Embedding dimension of the mask token.
        """
        super().__init__()
        self.mask_token: nn.Parameter = _init_trunc_normal(
            nn.Parameter(torch.zeros(1, 1, embed_dim))
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Replace token positions where mask indicates masking.

        Parameters
        ----------
        x : Tensor
            Token embeddings of shape (B, N, D).
        mask : Tensor
            Binary mask of shape (B, N) where True/1 indicates masked
            positions. Accepts both boolean and float masks.

        Returns
        -------
        Tensor
            Token embeddings with masked positions replaced by mask tokens,
            shape (B, N, D).

        Raises
        ------
        AssertionError
            If mask shape doesn't match the expected (B, N) dimensions.
        """
        batch_dim, seq_len, in_dim = x.shape

        # Check mask dimensions
        assert mask.shape == (batch_dim, seq_len), (
            f"Mask shape {mask.shape} doesn't match expected "
            f"({batch_dim}, {seq_len})"
        )

        # Ensure mask has correct dtype for operations
        if mask.dtype == torch.bool:
            mask = mask.to(dtype=x.dtype, device=x.device)

        # Expand mask tokens to batch size and masked positions
        # Ensure correct dtype for mixed precision compatibility
        mask_tokens = self.mask_token.to(dtype=x.dtype).expand(
            batch_dim, seq_len, -1
        )  # shape: [B, N, D]

        # Create replacement tensor where mask==1 positions come from
        # mask_tokens, others from x
        mask = mask.unsqueeze(-1)  # shape: [B, N, 1]
        return x * (1 - mask) + mask_tokens * mask  # shape: [B, N, D]
