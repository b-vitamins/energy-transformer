"""Energy-based multi-head attention module implementation."""

import math

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseEnergyAttention


class MultiHeadEnergyAttention(BaseEnergyAttention):
    """Multi-Head Energy Attention.

    Defines an energy function whose gradient
    implicitly defines the attention operation.

    Parameters
    ----------
    in_dim : int
        Input dimension D of token vectors
    num_heads : int
        Number of attention heads H
    head_dim : int
        Dimension of the key/query space Y
    beta : float, optional
        Temperature parameter β. If None, defaults to **1 / √(head_dim)**.

    Notes
    -----
    The energy-based attention operation is
    described by the following energy function:

    E^ATT = -(1/β)·∑ₕ₌₁ᴴ·∑ᶜ₌₁ᴺ·log(∑ᴮ≠ᶜ exp(β·Aₕᴮᶜ))

    where the attention matrix A is computed from query and key tensors:

    Aₕᴮᶜ = ∑ₐ Kₐₕᴮ·Qₐₕᶜ,       A ∈ ℝᴴˣᴺˣᴺ
    Kₐₕᴮ = ∑ⱼ W^K_ₐₕⱼ·gⱼᴮ,    K ∈ ℝʸˣᴴˣᴺ
    Qₐₕᶜ = ∑ⱼ W^Q_ₐₕⱼ·gⱼᶜ,    Q ∈ ℝʸˣᴴˣᴺ

    - The tensors W^K ∈ ℝʸˣᴴˣᴰ and W^Q ∈ ℝʸˣᴴˣᴰ are learnable parameters,
    - N is the sequence length (number of tokens),
    - H is the number of attention heads,
    - D is the input dimension of each token,
    - Y is the key/query projection dimension.

    Each token generates two representations:
    - query: where should it look for prompts on how to evolve?
    - key: what should be the contents of tokens that attend to it?
    """

    def __init__(
        self,
        in_dim: int,
        num_heads: int = 12,
        head_dim: int = 64,
        beta: float = None,
    ):
        """Initialize the Energy Attention layer.

        Parameters
        ----------
        in_dim : int
            Input dimension D of token vectors
        num_heads : int
            Number of attention heads H
        head_dim : int
            Dimension of the key/query space Y
        beta : float, optional
            Temperature parameter β. If None, defaults to **1 / √(head_dim)**.
        """
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # W^K ∈ ℝʸˣᴴˣᴰ
        # Note: [H, Y, D] for computational efficiency in PyTorch
        self.w_k = nn.Parameter(
            torch.empty(num_heads, head_dim, in_dim)
        )  # shape: [H, Y, D]

        # W^Q ∈ ℝʸˣᴴˣᴰ
        # Note: [H, Y, D] for computational efficiency in PyTorch
        self.w_q = nn.Parameter(
            torch.empty(num_heads, head_dim, in_dim)
        )  # shape: [H, Y, D]

        # β – same default as the original ET implementation
        self.β = beta if beta is not None else 1.0 / math.sqrt(head_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters."""
        nn.init.normal_(self.w_k, std=0.02)
        nn.init.normal_(self.w_q, std=0.02)

    def forward(self, g: Tensor) -> Tensor:
        """Compute attention energy.

        Parameters
        ----------
        g : Tensor
            Input tensor of shape [..., N, D] where N is the number
            of tokens and D is the embedding dimension

        Returns
        -------
        Tensor
            Scalar energy value (summed over batch, heads, and tokens).
            Returns zero for inputs with sequence length N=1, as
            self-attention is undefined for single-token sequences.
        """
        # Extract sequence length
        seq_len = g.shape[-2]

        # Single token edge case - return 0 energy (scalar)
        if seq_len == 1:
            return torch.zeros((), device=g.device, dtype=g.dtype)

        # Kₐₕᴮ = ∑ⱼ W^K_ₐₕⱼ·gⱼᴮ,    K ∈ ℝʸˣᴴˣᴺ
        k = torch.einsum(
            "...nd,hyd->...nhy", g, self.w_k
        )  # shape: [..., N, H, Y]

        # Qₐₕᶜ = ∑ⱼ W^Q_ₐₕⱼ·gⱼᶜ,    Q ∈ ℝʸˣᴴˣᴺ
        q = torch.einsum(
            "...nd,hyd->...nhy", g, self.w_q
        )  # shape: [..., N, H, Y]

        # Aₕᴮᶜ = ∑ₐ Kₐₕᴮ·Qₐₕᶜ,     A ∈ ℝᴴˣᴺˣᴺ
        a = torch.einsum("...nhy,...mhy->...hnm", k, q)  # shape: [..., H, N, N]
        # TODO: chunked attention calculation to reduce memory footprint
        # TODO: flash attention to avoid full N×N materialization
        # TODO: linear attention approximations for sub-quadratic scaling

        # ∑ᴮ≠ᶜ - Mask first for efficiency
        # Use broadcasting instead of materializing full mask tensor
        diag_mask = torch.eye(seq_len, device=g.device).bool()[None, None]
        a = a.masked_fill(diag_mask, float("-inf"))  # [..., H, N, N]

        # β·Aₕᴮᶜ - Scale after masking
        βa = self.β * a  # shape: [..., H, N, N]

        # log(∑ᴮ≠ᶜ exp(β·Aₕᴮᶜ))
        lse = torch.logsumexp(βa, dim=-2)  # [..., H, N]

        # E^ATT = -(1/β)·∑ₕ₌₁ᴴ·∑ᶜ₌₁ᴺ·log(∑ᴮ≠ᶜ exp(β·Aₕᴮᶜ))
        β_inv = 1.0 / self.β
        e_att = -(β_inv * lse).sum()

        return e_att
