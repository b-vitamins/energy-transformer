"""Energy-based LayerNorm implementation."""

import torch
import torch.nn as nn
from torch import Tensor


class EnergyLayerNorm(nn.Module):
    """
    Modified LayerNorm with energy formulation via Lagrangian.

    Uses a scalar scale (gamma) and optional bias (delta) instead of standard per-feature parameters.
    """

    def __init__(
        self,
        d_model: int,
        use_bias: bool = True,
        eps: float = 1e-5,
    ):
        """
        Initialize the EnergyLayerNorm layer.

        Parameters
        ----------
        d_model : int
            Feature dimension.
        use_bias : bool, optional
            Whether to include a bias term (default: True).
        eps : float, optional
            Epsilon for numerical stability (default: 1e-5).
        """
        super().__init__()
        self.d_model = d_model
        self.use_bias = use_bias
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(()))
        self.delta = nn.Parameter(torch.zeros(d_model)) if use_bias else None

    def lagrangian(self, x: Tensor) -> Tensor:
        """
        Compute the Lagrangian (integral) of the LayerNorm operation.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., d_model).

        Returns
        -------
        Tensor
            Scalar Lagrangian value.
        """
        d = x.shape[-1]
        x_centered = x - x.mean(dim=-1, keepdim=True)
        var_sum = (x_centered**2).sum(dim=-1)
        term1 = d * self.gamma * torch.sqrt(var_sum / d + self.eps)

        if self.use_bias and self.delta is not None:
            term2 = (self.delta * x).sum()
            return term1.sum() + term2

        return term1.sum()

    def g(self, x: Tensor) -> Tensor:
        """
        Apply the activation (gradient of the Lagrangian), equivalent to LayerNorm.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., d_model).

        Returns
        -------
        Tensor
            Normalized output tensor.
        """
        x_centered = x - x.mean(dim=-1, keepdim=True)
        var = (x_centered**2).mean(dim=-1, keepdim=True)
        normalized = self.gamma * x_centered / torch.sqrt(var + self.eps)

        if self.use_bias and self.delta is not None:
            normalized = normalized + self.delta

        return normalized

    def forward(self, x: Tensor) -> Tensor:
        """Alias for the LayerNorm operation (g)."""
        return self.g(x)

    def energy(self, x: Tensor) -> Tensor:
        """
        Compute energy via Legendre transform: g(x)*x - Lagrangian(x).

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., d_model).

        Returns
        -------
        Tensor
            Scalar energy value.
        """
        g_x = self.g(x)
        return (g_x * x).sum() - self.lagrangian(x)
