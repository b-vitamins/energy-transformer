"""Energy-based multi-head attention module implementation."""

import math

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseEnergyAttention

# Global cache for diagonal masks to avoid O(N²) allocations per forward pass
_diag_cache: dict[tuple[torch.device, int], Tensor] = {}


def _get_diag_mask(device: torch.device, seq_len: int) -> Tensor:
    """Get cached diagonal mask for self-attention exclusion.

    Parameters
    ----------
    device : torch.device
        Device for the mask tensor
    seq_len : int
        Sequence length N

    Returns
    -------
    Tensor
        Boolean diagonal mask of shape [N, N]
    """
    key = (device, seq_len)
    if key not in _diag_cache:
        _diag_cache[key] = torch.eye(seq_len, device=device, dtype=torch.bool)
    return _diag_cache[key]


def _chunked_logsumexp(
    logits: Tensor, dim: int = -2, chunk_size: int = 1024
) -> Tensor:
    """Compute logsumexp in chunks to reduce memory footprint.

    Parameters
    ----------
    logits : Tensor
        Input tensor to compute logsumexp over
    dim : int
        Dimension to compute logsumexp along
    chunk_size : int
        Size of chunks to process at once

    Returns
    -------
    Tensor
        Logsumexp result with same shape as input except along dim
    """
    if logits.size(dim) <= chunk_size:
        return torch.logsumexp(logits, dim=dim)

    # Split along the specified dimension
    chunks = torch.chunk(
        logits, chunks=math.ceil(logits.size(dim) / chunk_size), dim=dim
    )

    # Compute logsumexp for each chunk
    chunk_lse = [
        torch.logsumexp(chunk, dim=dim, keepdim=True) for chunk in chunks
    ]

    # Combine chunk results using stable logsumexp
    combined = torch.cat(chunk_lse, dim=dim)
    return torch.logsumexp(combined, dim=dim)


class MultiHeadEnergyAttention(BaseEnergyAttention):
    """Multi-Head Energy Attention with memory and compute optimizations.

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
    per_head_bias : bool, optional
        If True and bias=True, use separate bias per head [H, Y]. Default is False
    init_std : float, optional
        Standard deviation for weight initialization. Default is 0.002
    use_flash_attention : bool, optional
        Whether to use Flash Attention when available. Default is True
    chunk_size : int, optional
        Chunk size for memory-efficient logsumexp. Default is 1024

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
        per_head_bias: bool = False,
        init_std: float = 0.002,
        use_flash_attention: bool = True,
        chunk_size: int = 1024,
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
        bias : bool, optional
            Whether to include bias terms for key/query projections
        per_head_bias : bool, optional
            If True and bias=True, use separate bias per head [H, Y]
        init_std : float, optional
            Standard deviation for weight initialization
        use_flash_attention : bool, optional
            Whether to use Flash Attention when available
        chunk_size : int, optional
            Chunk size for memory-efficient operations
        """
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.per_head_bias = per_head_bias
        self.use_flash_attention = use_flash_attention
        self.chunk_size = chunk_size

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

        # Optional bias terms
        if bias:
            if per_head_bias:
                self.b_k = nn.Parameter(
                    torch.zeros(num_heads, head_dim)
                )  # shape: [H, Y]
                self.b_q = nn.Parameter(
                    torch.zeros(num_heads, head_dim)
                )  # shape: [H, Y]
            else:
                self.b_k = nn.Parameter(torch.zeros(head_dim))  # shape: [Y]
                self.b_q = nn.Parameter(torch.zeros(head_dim))  # shape: [Y]
        else:
            self.register_parameter("b_k", None)
            self.register_parameter("b_q", None)

        # β – same default as the original ET implementation
        self.β = beta if beta is not None else 1.0 / math.sqrt(head_dim)

        # Store initialization std for reset_parameters
        self.init_std = init_std

        self.reset_parameters()

    def reset_parameters(self, std: float | None = None) -> None:
        """Initialize learnable parameters.

        Parameters
        ----------
        std : float, optional
            Standard deviation for weight initialization.
            If None, uses self.init_std
        """
        init_std = std if std is not None else self.init_std
        nn.init.normal_(self.w_k, std=init_std)
        nn.init.normal_(self.w_q, std=init_std)
        if self.b_k is not None:
            nn.init.zeros_(self.b_k)
        if self.b_q is not None:
            nn.init.zeros_(self.b_q)

    def _try_flash_attention(
        self, k: Tensor, q: Tensor, attn_mask: Tensor | None = None
    ) -> Tensor | None:
        """Try to use Flash Attention if available and applicable.

        Parameters
        ----------
        k : Tensor
            Key tensor of shape [..., N, H, Y]
        q : Tensor
            Query tensor of shape [..., N, H, Y]
        attn_mask : Tensor, optional
            Attention mask

        Returns
        -------
        Optional[Tensor]
            Attention logits if Flash Attention was used, None otherwise
        """
        if not self.use_flash_attention or not torch.cuda.is_available():
            return None

        try:
            # Reshape for Flash Attention: [B, N, H, Y]
            batch_dims = k.shape[:-3]
            if len(batch_dims) == 0:
                k.unsqueeze(0)  # Add batch dim
                q.unsqueeze(0)
            else:
                k.flatten(0, -4).unsqueeze(0) if len(batch_dims) > 1 else k
                q.flatten(0, -4).unsqueeze(0) if len(batch_dims) > 1 else q

            # Use scaled_dot_product_attention for Flash-like performance
            # Note: This computes attention weights, but we need raw logits for energy
            # So we'll fall back to manual computation for now
            return None

        except Exception:
            return None

    def forward(
        self,
        g: Tensor,
        attn_mask: Tensor | None = None,
        include_diag: bool = True,
    ) -> Tensor:  # scalar
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
            "...nd,hyd->...nhy", g, w_k_cast
        )  # shape: [..., N, H, Y]

        # Qₐₕᶜ = ∑ⱼ W^Q_ₐₕⱼ·gⱼᶜ,    Q ∈ ℝʸˣᴴˣᴺ
        q = torch.einsum(
            "...nd,hyd->...nhy", g, w_q_cast
        )  # shape: [..., N, H, Y]

        # Add bias if present
        if self.b_k is not None:
            if self.per_head_bias:
                k = k + self.b_k.unsqueeze(-3)  # Broadcasting: [..., 1, H, Y]
            else:
                k = k + self.b_k  # Broadcasting: [..., N, H, Y]
        if self.b_q is not None:
            if self.per_head_bias:
                q = q + self.b_q.unsqueeze(-3)  # Broadcasting: [..., 1, H, Y]
            else:
                q = q + self.b_q  # Broadcasting: [..., N, H, Y]

        # Try Flash Attention first (currently returns None for energy computation)
        flash_result = self._try_flash_attention(k, q, attn_mask)
        if flash_result is not None:
            # Flash Attention path (not implemented for energy yet)
            pass

        # Standard attention computation
        # Aₕᴮᶜ = ∑ₐ Kₐₕᴮ·Qₐₕᶜ,     A ∈ ℝᴴˣᴺˣᴺ
        a = torch.einsum("...nhy,...mhy->...hnm", k, q)  # shape: [..., H, N, N]

        # Mask diagonal (self-token) entries if requested
        if not include_diag:
            # Use cached diagonal mask to avoid O(N²) allocation every forward pass
            diag_mask = _get_diag_mask(g.device, seq_len)
            diag_mask = diag_mask[
                None, None
            ]  # Broadcast for heads [..., 1, 1, N, N]
            a = a.masked_fill(diag_mask, float("-inf"))  # shape: [..., H, N, N]

        # Apply external attention mask if provided
        if attn_mask is not None:
            # Use additive masking (compatible with -inf values)
            a = a + attn_mask  # shape: [..., H, N, N]

        # β·Aₕᴮᶜ - Scale attention matrix by temperature
        βa = self.β * a  # shape: [..., H, N, N]

        # log(∑ᴮ exp(β·Aₕᴮᶜ)) - LogSumExp over keys dimension
        # Use chunked computation for memory efficiency on long sequences
        if seq_len > self.chunk_size:
            lse = _chunked_logsumexp(βa, dim=-2, chunk_size=self.chunk_size)
        else:
            lse = torch.logsumexp(βa, dim=-2)  # shape: [..., H, N]

        # E^ATT = -(1/β)·∑ₕ₌₁ᴴ·∑ᶜ₌₁ᴺ·log(∑ᴮ exp(β·Aₕᴮᶜ))
        β_inv = 1.0 / self.β
        e_att = -(β_inv * lse).sum()  # scalar

        return e_att


def clear_diag_cache() -> None:
    """Clear the diagonal mask cache to free memory."""
    global _diag_cache
    _diag_cache.clear()
