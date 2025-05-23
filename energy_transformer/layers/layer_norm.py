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
        self.in_dim = in_dim
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
        # Handle both float16 and bfloat16 for mixed precision training
        calc_dtype = orig_dtype
        if orig_dtype in {torch.float16, torch.bfloat16}:
            # For numerical stability, use float32 for internal calculations
            x = x.to(torch.float32)
            calc_dtype = torch.float32

        # Get positive γ using softplus
        γ = F.softplus(self.logγ).to(dtype=calc_dtype)

        # Ensure δ has the correct dtype
        δ = self.δ.to(dtype=calc_dtype)

        # x̄ = 1/D·∑ₖ₌₁ᴰ xₖ
        x_mean = x.mean(dim=-1, keepdim=True)  # shape: [..., 1]

        # More efficient variance computation using torch.var
        # 1/D·∑ⱼ(xⱼ - x̄)² computed in-kernel
        var = torch.var(
            x, dim=-1, unbiased=False, keepdim=True
        )  # shape: [..., 1]

        # xᵢ - x̄
        x_c = x - x_mean  # shape: [..., D]

        # gᵢ = γ·(xᵢ - x̄)/√(1/D·∑ⱼ(xⱼ - x̄)² + ε) + δᵢ
        g = γ * x_c / torch.sqrt(var + self.eps) + δ  # shape: [..., D]

        # Convert back to original dtype if necessary
        if calc_dtype != orig_dtype:
            g = g.to(orig_dtype)

        return g

    def export_standard_layernorm(self) -> nn.LayerNorm:
        """Export as standard PyTorch LayerNorm for optimized inference.

        Returns
        -------
        nn.LayerNorm
            Equivalent standard LayerNorm module with frozen parameters

        Notes
        -----
        This creates a standard nn.LayerNorm instance with the same
        normalization behavior as this energy-based implementation.
        Useful for deployment when you want maximum inference speed
        and don't need the energy interpretation or gradient computation.
        """
        # Create standard LayerNorm
        standard_ln = nn.LayerNorm(
            normalized_shape=self.in_dim, eps=self.eps, elementwise_affine=True
        )

        # Copy the learned parameters
        with torch.no_grad():
            # γ parameter (weight in standard LayerNorm)
            γ = F.softplus(self.logγ)
            standard_ln.weight.fill_(γ.item())

            # δ parameter (bias in standard LayerNorm)
            standard_ln.bias.copy_(self.δ)

        return standard_ln

    def get_energy_lagrangian(self, x: Tensor) -> Tensor:
        """Compute the energy Lagrangian L for given input.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [..., D]

        Returns
        -------
        Tensor
            Energy Lagrangian L of shape [...] (scalar per sample)

        Notes
        -----
        Computes: L = D·γ·√(1/D·∑ⱼ(xⱼ - x̄)² + ε) + ∑ⱼδⱼ·xⱼ

        This is the energy function whose partial derivative gives
        the layer normalization operation: gᵢ = ∂L/∂xᵢ
        """
        # Handle mixed precision
        orig_dtype = x.dtype
        calc_dtype = orig_dtype
        if orig_dtype in {torch.float16, torch.bfloat16}:
            x = x.to(torch.float32)
            calc_dtype = torch.float32

        # Get positive γ
        γ = F.softplus(self.logγ).to(dtype=calc_dtype)
        δ = self.δ.to(dtype=calc_dtype)

        # Compute variance term: 1/D·∑ⱼ(xⱼ - x̄)²
        var = torch.var(
            x, dim=-1, unbiased=False, keepdim=False
        )  # shape: [...]

        # First term: D·γ·√(1/D·∑ⱼ(xⱼ - x̄)² + ε)
        energy_norm = self.in_dim * γ * torch.sqrt(var + self.eps)

        # Second term: ∑ⱼδⱼ·xⱼ
        energy_bias = torch.sum(δ * x, dim=-1)  # shape: [...]

        # Total Lagrangian
        lagrangian = energy_norm + energy_bias

        # Convert back to original dtype
        if calc_dtype != orig_dtype:
            lagrangian = lagrangian.to(orig_dtype)

        return lagrangian


# Utility function for TorchInductor optimization preparation
def _functional_layernorm_energy(
    x: Tensor, logγ: Tensor, δ: Tensor, eps: float = 1e-5
) -> Tensor:
    """Functional version of energy LayerNorm for compilation optimization.

    This can be used with torch.func.functional_call for better
    TorchInductor kernel generation and graph optimization.

    Parameters
    ----------
    x : Tensor
        Input tensor
    logγ : Tensor
        Log-gamma parameter (scalar)
    δ : Tensor
        Delta bias parameter
    eps : float
        Epsilon for numerical stability

    Returns
    -------
    Tensor
        Normalized output
    """
    # Handle mixed precision
    orig_dtype = x.dtype
    if orig_dtype in {torch.float16, torch.bfloat16}:
        x = x.to(torch.float32)

    # Get positive γ
    γ = F.softplus(logγ)

    # Compute normalization
    x_mean = x.mean(dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
    x_c = x - x_mean
    g = γ * x_c / torch.sqrt(var + eps) + δ

    # Convert back to original dtype
    if orig_dtype in {torch.float16, torch.bfloat16}:
        g = g.to(orig_dtype)

    return g
