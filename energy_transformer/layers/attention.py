"""Energy-based multi-head attention module implementation."""

import math

import torch
from torch import Tensor, nn

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
    bias : bool, optional
        Whether to include bias terms for key/query projections. Default is False
    init_std : float, optional
        Standard deviation for weight initialization. Default is 0.002

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
        beta: float | None = None,
        bias: bool = False,
        init_std: float = 0.002,
    ) -> None:
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
        bias : bool, optional
            Whether to include bias terms for key/query projections
        init_std : float, optional
            Standard deviation for weight initialization
        """
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # W^K ∈ ℝʸˣᴴˣᴰ
        # Note: [H, Y, D] for computational efficiency in PyTorch
        self.w_k = nn.Parameter(
            torch.empty(num_heads, head_dim, in_dim),
        )  # shape: [H, Y, D]

        # W^Q ∈ ℝʸˣᴴˣᴰ
        # Note: [H, Y, D] for computational efficiency in PyTorch
        self.w_q = nn.Parameter(
            torch.empty(num_heads, head_dim, in_dim),
        )  # shape: [H, Y, D]

        # Optional bias terms
        if bias:
            self.b_k = nn.Parameter(torch.zeros(head_dim))  # shape: [Y]
            self.b_q = nn.Parameter(torch.zeros(head_dim))  # shape: [Y]
            # Note: Current bias broadcasts across both N and H dimensions.
            # For per-head bias, use shape (num_heads, head_dim) instead.
        else:
            self.register_parameter("b_k", None)
            self.register_parameter("b_q", None)

        # β – same default as the original ET implementation
        self.β = beta if beta is not None else 1.0 / math.sqrt(head_dim)

        # Store initialization std for reset_parameters
        self.init_std = init_std

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters."""
        nn.init.normal_(self.w_k, std=self.init_std)
        nn.init.normal_(self.w_q, std=self.init_std)
        if self.b_k is not None:
            nn.init.zeros_(self.b_k)
        if self.b_q is not None:
            nn.init.zeros_(self.b_q)

    def forward(
        self,
        g: Tensor,
        attn_mask: Tensor | None = None,
        include_diag: bool = True,
    ) -> Tensor:
        """Compute attention energy.

        Parameters
        ----------
        g : Tensor
            Input tensor of shape [..., N, D] where N is the number
            of tokens and D is the embedding dimension
        attn_mask : Tensor, optional
            Attention mask of shape [..., H, N, N] or broadcastable.
            Added to attention logits before logsumexp. Use 0 for allowed
            positions and -∞ for masked positions (not 0/1 masks).
            Compatible with causal masks: `torch.triu(torch.full((N, N), -inf), 1)`
        include_diag : bool, optional
            Whether to include diagonal (self-attention) entries.
            Default is True for exact ET behavior

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

        # Mixed-precision safety: cast weights to input dtype for pure bf16
        if g.dtype in {torch.float16, torch.bfloat16}:
            w_k_cast = self.w_k.to(g.dtype)
            w_q_cast = self.w_q.to(g.dtype)
        else:
            w_k_cast = self.w_k
            w_q_cast = self.w_q

        # Kₐₕᴮ = ∑ⱼ W^K_ₐₕⱼ·gⱼᴮ,    K ∈ ℝʸˣᴴˣᴺ
        k = torch.einsum(
            "...nd,hyd->...nhy",
            g,
            w_k_cast,
        )  # shape: [..., N, H, Y]

        # Qₐₕᶜ = ∑ⱼ W^Q_ₐₕⱼ·gⱼᶜ,    Q ∈ ℝʸˣᴴˣᴺ
        q = torch.einsum(
            "...nd,hyd->...nhy",
            g,
            w_q_cast,
        )  # shape: [..., N, H, Y]

        # Add bias if present
        if self.b_k is not None:
            k = k + self.b_k  # shape: [..., N, H, Y]
        if self.b_q is not None:
            q = q + self.b_q  # shape: [..., N, H, Y]

        # Aₕᴮᶜ = ∑ₐ Kₐₕᴮ·Qₐₕᶜ,     A ∈ ℝᴴˣᴺˣᴺ
        a = torch.einsum("...nhy,...mhy->...hnm", k, q)  # shape: [..., H, N, N]
        # TODO: avoid full N×N materialization

        # Mask diagonal (self-token) entries if requested
        if not include_diag:
            # ∑ᴮ≠ᶜ - Create mask for self-attention exclusion
            diag_mask = torch.eye(seq_len, device=g.device, dtype=torch.bool)
            diag_mask = diag_mask[
                None,
                None,
            ]  # Broadcast for heads [..., 1, 1, N, N]
            a = a.masked_fill(diag_mask, float("-inf"))  # shape: [..., H, N, N]

        # Apply external attention mask if provided
        if attn_mask is not None:
            # Use additive masking (compatible with -inf values)
            a = a + attn_mask  # shape: [..., H, N, N]

        # β·Aₕᴮᶜ - Scale attention matrix by temperature
        βa = self.β * a  # shape: [..., H, N, N]

        # log(∑ᴮ exp(β·Aₕᴮᶜ)) - LogSumExp over keys dimension
        lse = torch.logsumexp(βa, dim=-2)  # shape: [..., H, N]

        # E^ATT = -(1/β)·∑ₕ₌₁ᴴ·∑ᶜ₌₁ᴺ·log(∑ᴮ exp(β·Aₕᴮᶜ))
        β_inv = 1.0 / self.β
        return -(β_inv * lse).sum()  # scalar
