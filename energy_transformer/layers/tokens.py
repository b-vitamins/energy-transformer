"""Special token implementations for Energy Transformer models."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# Expose as tuple for faster imports
__all__ = (
    "CLSToken",
)


class CLSToken(nn.Module):  # type: ignore[misc]
    """Learnable classification token.

    Implements a learnable classification token that is prepended to the
    beginning of token sequences to provide a global representation for
    downstream tasks like classification.

    This follows the approach used in BERT and Vision Transformers where
    a special [CLS] token aggregates information from the entire sequence
    through self-attention mechanisms.

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
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize CLS token weights.

        Uses truncated normal initialization following Vision Transformer
        conventions for stable training.
        """
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Prepend CLS token to input sequence.

        Parameters
        ----------
        x : Tensor
            Token embeddings of shape (B, N, D) where B is batch size,
            N is sequence length, and D is embedding dimension.

        Returns
        -------
        Tensor
            Token embeddings with CLS token prepended, shape (B, N+1, D).
            The CLS token is at position 0 in the sequence dimension.
        """
        b = x.shape[0]

        # Expand CLS token to match batch size and dtype
        cls_tokens = self.cls_token
        if cls_tokens.dtype != x.dtype:
            cls_tokens = cls_tokens.to(x.dtype)
        cls_tokens = cls_tokens.expand(b, -1, -1)  # (B, 1, D)

        # Prepend CLS token to sequence
        return torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)
