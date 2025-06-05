"""MLP module for transformer blocks."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from torch import Tensor, nn

from .constants import DEFAULT_MLP_RATIO

__all__ = ["MLP"]


class MLP(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks.

    Standard two-layer MLP with configurable hidden dimension, activation,
    and dropout. Follows the timm implementation conventions.

    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : int or None, default=None
        Number of hidden features. If None, defaults to in_features.
    out_features : int or None, default=None
        Number of output features. If None, defaults to in_features.
    act_layer : Callable[..., nn.Module], default=nn.GELU
        Activation layer class. Will be instantiated with no arguments.
    bias : bool, default=True
        If True, adds a learnable bias to the linear layers.
    drop : float, default=0.0
        Dropout probability.

    Attributes
    ----------
    fc1 : nn.Linear
        First linear transformation.
    act : nn.Module
        Activation function.
    drop : nn.Dropout
        Dropout layer.
    fc2 : nn.Linear
        Second linear transformation.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(
            in_features * DEFAULT_MLP_RATIO
        )

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., in_features).

        Returns
        -------
        Tensor
            Output tensor of shape (..., out_features).
        """
        if x.size(-1) != self.fc1.in_features:
            raise ValueError(
                f"MLP: Feature dimension mismatch. "
                f"Expected: {self.fc1.in_features}, got: {x.size(-1)}."
            )

        x = cast(Tensor, self.fc1(x))
        x = cast(Tensor, self.act(x))
        x = cast(Tensor, self.drop(x))
        return cast(Tensor, self.fc2(x))
