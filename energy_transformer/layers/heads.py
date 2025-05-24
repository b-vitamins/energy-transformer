"""Task-specific head implementations for Energy Transformer models."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from typing import cast

__all__ = [
    "ClassificationHead",
    "FeatureHead",
]


class ClassificationHead(nn.Module):
    """Classification head for vision models.

    Implements a standard classification head that processes the CLS token
    output from a vision transformer to produce class logits. Supports
    optional representation layer following the original Vision Transformer
    design.

    Parameters
    ----------
    embed_dim : int
        Input embedding dimension.
    num_classes : int
        Number of output classes.
    representation_size : int, optional
        Size of intermediate representation layer. If provided, adds
        a linear layer followed by Tanh activation before the final
        classification layer, by default None.
    drop_rate : float, default=0.0
        Dropout rate applied before the final classification layer.
    use_cls_token : bool, default=True
        Whether to extract the CLS token (position 0) or use global
        average pooling across all tokens.

    Attributes
    ----------
    use_cls_token : bool
        Whether to use CLS token for classification.
    pre_logits : nn.Module
        Pre-logits processing layer (Identity or Linear + Tanh).
    drop : nn.Dropout
        Dropout layer applied before classification.
    head : nn.Linear
        Final classification layer.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        representation_size: int | None = None,
        drop_rate: float = 0.0,
        use_cls_token: bool = True,
    ) -> None:
        """Initialize ClassificationHead module.

        Parameters
        ----------
        embed_dim : int
            Input embedding dimension.
        num_classes : int
            Number of output classes.
        representation_size : int, optional
            Size of intermediate representation layer, by default None.
        drop_rate : float, default=0.0
            Dropout rate applied before the final classification layer.
        use_cls_token : bool, default=True
            Whether to extract the CLS token or use global average pooling.
        """
        super().__init__()
        self.use_cls_token = use_cls_token

        # Pre-logits processing (following original ViT design)
        self.pre_logits: nn.Module
        if representation_size is not None:
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size), nn.Tanh()
            )
            head_input_dim = representation_size
        else:
            self.pre_logits = nn.Identity()
            head_input_dim = embed_dim

        # Dropout and classification layers
        self.drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(head_input_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize classification head weights.

        Uses zero initialization for the final classification layer
        following Vision Transformer conventions.
        """
        # Zero initialization for classification head
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # Initialize pre_logits if it's a Linear layer
        if isinstance(self.pre_logits, nn.Sequential):
            linear_layer = self.pre_logits[0]
            if isinstance(linear_layer, nn.Linear):
                nn.init.trunc_normal_(linear_layer.weight, std=0.02)
                nn.init.zeros_(linear_layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Extract features and produce classification logits.

        Parameters
        ----------
        x : Tensor
            Token representations of shape (B, N, D) where B is batch size,
            N is sequence length (including CLS token if present), and
            D is embedding dimension.

        Returns
        -------
        Tensor
            Classification logits of shape (B, num_classes).
        """
        # Extract global representation
        if self.use_cls_token:
            # Use CLS token (first token in sequence)
            x = x[:, 0]  # (B, D)
        else:
            # Use global average pooling over all tokens
            x = x.mean(dim=1)  # (B, D)

        # Apply pre-logits processing
        x = self.pre_logits(x)  # (B, representation_size) or (B, D)

        # Apply dropout and classification
        x = self.drop(x)
        logits = cast(Tensor, self.head(x))  # (B, num_classes)

        return logits


class FeatureHead(nn.Module):
    """Feature extraction head for vision models.

    Extracts feature representations from token sequences without applying
    classification. Useful for transfer learning, representation analysis,
    and downstream tasks that require raw features.

    Parameters
    ----------
    use_cls_token : bool, default=True
        Whether to extract the CLS token (position 0) or use global
        average pooling across all tokens.

    Attributes
    ----------
    use_cls_token : bool
        Whether to use CLS token for feature extraction.
    """

    def __init__(self, use_cls_token: bool = True) -> None:
        """Initialize FeatureHead module.

        Parameters
        ----------
        use_cls_token : bool, default=True
            Whether to extract the CLS token or use global average pooling.
        """
        super().__init__()
        self.use_cls_token = use_cls_token

    def forward(self, x: Tensor) -> Tensor:
        """Extract feature representations.

        Parameters
        ----------
        x : Tensor
            Token representations of shape (B, N, D) where B is batch size,
            N is sequence length (including CLS token if present), and
            D is embedding dimension.

        Returns
        -------
        Tensor
            Feature representations of shape (B, D).
        """
        if self.use_cls_token:
            # Extract CLS token features (first token in sequence)
            return x[:, 0]  # (B, D)
        else:
            # Use global average pooling over all tokens
            return x.mean(dim=1)  # (B, D)
