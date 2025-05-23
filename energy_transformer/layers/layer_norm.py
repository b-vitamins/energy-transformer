"""Energy-based LayerNorm implementation."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from .base import BaseLayerNorm


class LayerNorm(BaseLayerNorm):
    """Layer normalized token representation with strict energy interpretation.

    Parameters
    ----------
    in_dim : int
        Feature dimension D of token vectors
    eps : float, optional
        Epsilon for numerical stability, by default 1e-5

    Notes
    -----
    Each token is represented by a vector x ∈ ℝᴰ.
    ET block operations use a layer-normalized token representation:

    gᵢ = γ·(xᵢ - x̄)/√(1/D·∑ⱼ(xⱼ - x̄)² + ε) + δᵢ

    where x̄ = 1/D·∑ₖ₌₁ᴰ xₖ

    The scalar γ > 0 and vector elements δᵢ are learnable parameters.
    ε is a small regularization constant.

    This operation can be defined as a partial derivative
    of the Lagrangian (energy) function:

    L = D·γ·√(1/D·∑ⱼ(xⱼ - x̄)² + ε) + ∑ⱼδⱼ·xⱼ

    such that gᵢ = ∂L/∂xᵢ

    The positivity constraint on γ ensures that L is bounded below,
    which is essential for a valid energy-based interpretation where
    probability distributions proportional to e^(-L) must be normalizable.
    """

    def __init__(
        self,
        in_dim: int,
        eps: float = 1e-5,
    ):
        """Initialize LayerNorm module.

        Parameters
        ----------
        in_dim : int
            Feature dimension D of token vectors
        eps : float, optional
            Small constant for numerical stability, by default 1e-5
        """
        super().__init__()
        self.eps = eps

        # Store log(γ) and apply softplus to ensure γ > 0
        # Initialize to make softplus(logγ) = 1.0
        self.logγ = nn.Parameter(
            torch.tensor(math.log(math.exp(1.0) - 1))
        )  # shape: scalar

        # δ ∈ ℝᴰ
        self.δ = nn.Parameter(torch.empty(in_dim))  # shape: [D]

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters.

        logγ is initialized to make softplus(logγ) = 1.0,
        maintaining the standard identity initialization.
        """
        # Initialize logγ to make softplus(logγ) = 1.0
        with torch.no_grad():
            self.logγ.fill_(math.log(math.exp(1.0) - 1.0))
        nn.init.zeros_(self.δ)

    def forward(self, x: Tensor) -> Tensor:
        """Apply layer normalization to input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [..., D] where D is the
            feature dimension

        Returns
        -------
        Tensor
            Normalized output tensor of shape [..., D]
        """
        orig_dtype = x.dtype

        # For mixed precision calculations, use at least float32 internally
        # but ensure we return the same dtype as input
        calc_dtype = orig_dtype
        if orig_dtype == torch.float16:
            # For numerical stability, use float32 for internal calculations
            x = x.to(torch.float32)
            calc_dtype = torch.float32

        # Get positive γ using softplus
        γ = F.softplus(self.logγ).to(dtype=calc_dtype)

        # Ensure δ has the correct dtype
        δ = self.δ.to(dtype=calc_dtype)

        # x̄ = 1/D·∑ₖ₌₁ᴰ xₖ
        x_mean = x.mean(dim=-1, keepdim=True)  # shape: [..., 1]

        # xᵢ - x̄
        x_c = x - x_mean  # shape: [..., D]

        # 1/D·∑ⱼ(xⱼ - x̄)²
        var = (x_c**2).mean(dim=-1, keepdim=True)  # shape: [..., 1]

        # gᵢ = γ·(xᵢ - x̄)/√(1/D·∑ⱼ(xⱼ - x̄)² + ε) + δᵢ
        g = γ * x_c / torch.sqrt(var + self.eps) + δ  # shape: [..., D]

        # Convert back to original dtype if necessary
        if calc_dtype != orig_dtype:
            g = g.to(orig_dtype)

        return g
