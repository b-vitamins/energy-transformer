"""Energy-based multi-head attention module implementation."""

import math
from typing import ClassVar

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .constants import (
    ATTENTION_EPSILON,
    ATTENTION_INIT_STD,
)
from .validation import (
    validate_divisibility,
    validate_positive,
)


class MultiheadEnergyAttention(nn.Module):
    r"""Multi-head energy-based attention.

    This module implements energy-based attention where attention patterns emerge
    from minimizing an energy function rather than explicit softmax normalization.
    The energy function implicitly defines the attention operation through its
    gradient.

    Parameters
    ----------
    embed_dim : int
        Total dimension of the model (D in paper notation).
    num_heads : int
        Number of parallel attention heads (H in paper notation). Note that ``embed_dim``
        will be split across ``num_heads`` (i.e. each head will have dimension
        ``embed_dim // num_heads``).
    beta : float | Tensor | None, default=None
        Temperature parameter(s). Can be:
        - None: defaults to ``1 / sqrt(head_dim)``
        - float: same temperature for all heads
        - Tensor of shape (num_heads,): per-head temperatures
    init_std : float, default=0.002
        Standard deviation for parameter initialization.
    batch_first : bool, default=True
        If ``True``, then the input and output tensors are provided
        as (batch, seq, feature). Default: ``True``.
    device : torch.device, optional
        Device for parameters.
    dtype : torch.dtype, optional
        Data type for parameters.

    Attributes
    ----------
    q_proj_weight : nn.Parameter
        Query projection weights W^Q in R^{H x Y x D}.
    k_proj_weight : nn.Parameter
        Key projection weights W^K in R^{H x Y x D}.
    beta : Tensor
        Temperature parameters β for each head.

    Notes
    -----
    Mathematical Foundation:
    The energy-based attention mechanism is designed to evolve tokens such that keys
    of open patches align with queries of masked patches. The energy function is:

    .. math::
        E^{ATT} = -\frac{1}{\beta} \sum_{h=1}^{H} \sum_{C=1}^{N} \log\left(\sum_{B \neq C} \exp(\beta A_{hBC})\right)

    where the attention matrix A is computed as:

    .. math::
        A_{hBC} = \sum_{\alpha} K_{\alpha hB} Q_{\alpha hC} = K_{hB}^T Q_{hC}

    with keys and queries:

    .. math::
        K_{\alpha hB} = \sum_{j} W_{\alpha hj}^K g_{jB}, \quad Q_{\alpha hC} = \sum_{j} W_{\alpha hj}^Q g_{jC}

    Key Theoretical Properties:

    1. **Self-Attention Exclusion**: The sum explicitly excludes B=C, meaning tokens
       do not attend to themselves. This is crucial for the energy formulation.

    2. **Two-Term Gradient**: The gradient with respect to g has two terms:

       .. math::
           -\frac{\partial E^{ATT}}{\partial g_{iA}} = \text{Term}_1 + \text{Term}_2

       where:

       - Term 1: :math:`\sum_{C \neq A} W_i^Q K_C \text{softmax}_C(\beta K_C^T Q_A)`
         This is conventional attention with value matrix V = (W^Q)^T K

       - Term 2: :math:`\sum_{C \neq A} W_i^K Q_C \text{softmax}_A(\beta K_A^T Q_C)`
         This is the novel contribution ensuring energy minimization

    3. **Relationship to Hopfield Networks**: While inspired by Modern Hopfield Networks,
       this differs fundamentally because keys are dynamic variables that evolve with
       queries, not fixed memories.

    Examples
    --------
    >>> # Default initialization
    >>> attn = MultiheadAttention(embed_dim=768, num_heads=12)
    >>> x = torch.randn(32, 100, 768)  # (batch, seq, feature)
    >>> energy = attn(x)  # scalar energy

    >>> # With per-head temperatures
    >>> betas = torch.linspace(0.1, 0.3, 12)
    >>> attn = MultiheadAttention(768, 12, beta=betas)

    >>> # Get gradient without autograd
    >>> grad = attn.compute_grad(x)  # (32, 100, 768)

    References
    ----------
    .. [1] Hoover et al. (2023). Energy Transformer. See equations (3) and (4).
    """

    __constants__: ClassVar[list[str]] = [
        "embed_dim",
        "num_heads",
        "batch_first",
    ]
    beta: Tensor

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        beta: float | Tensor | None = None,
        init_std: float = ATTENTION_INIT_STD,
        batch_first: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first

        validate_divisibility(
            embed_dim,
            num_heads,
            "MultiheadEnergyAttention",
            "embed_dim",
            "num_heads",
        )

        # Projection weights
        self.q_proj_weight = nn.Parameter(
            torch.empty((num_heads, self.head_dim, embed_dim), **factory_kwargs)  # type: ignore[arg-type]
        )
        self.k_proj_weight = nn.Parameter(
            torch.empty((num_heads, self.head_dim, embed_dim), **factory_kwargs)  # type: ignore[arg-type]
        )

        # Temperature parameters
        default_beta = 1.0 / math.sqrt(self.head_dim)

        match beta:
            case None:
                beta_tensor = torch.full(
                    (num_heads,), default_beta, device=device, dtype=dtype
                )
            case float() | int():
                validate_positive(
                    float(beta), "MultiheadEnergyAttention", "beta"
                )
                beta_tensor = torch.full(
                    (num_heads,), float(beta), device=device, dtype=dtype
                )
            case Tensor():
                if beta.shape != (num_heads,):
                    raise ValueError(
                        f"beta tensor must have shape ({num_heads},), "
                        f"got {beta.shape}"
                    )
                beta_tensor = beta.clone().to(device=device, dtype=dtype)
            case _:
                raise TypeError(
                    f"beta must be float, Tensor, or None, got {type(beta)}"
                )

        self.register_buffer("beta", beta_tensor)

        # Initialize
        self.init_std = init_std
        self._reset_parameters()

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.embed_dim // self.num_heads

    def _reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.normal_(self.q_proj_weight, std=self.init_std)
        nn.init.normal_(self.k_proj_weight, std=self.init_std)

    def _get_compute_dtype(self, x: Tensor) -> torch.dtype:
        """Get appropriate dtype for computation to avoid numerical issues."""
        if x.dtype in {torch.float16, torch.bfloat16}:
            return torch.float32
        return x.dtype

    def _validate_and_prepare_input(self, x: Tensor) -> tuple[Tensor, int, int]:
        """Validate input and extract dimensions.

        Returns
        -------
        tuple[Tensor, int, int]
            Prepared tensor, batch_size, sequence_length
        """
        if self.batch_first:
            if x.dim() != 3:  # noqa: PLR2004
                raise ValueError(
                    "MultiheadEnergyAttention: Expected 3D input (batch, seq, embed) "
                    "when batch_first=True. Got "
                    f"{x.dim()}D tensor."
                )
            batch_size, seq_len, embed_dim = x.shape
            if embed_dim != self.embed_dim:
                raise ValueError(
                    "MultiheadEnergyAttention: Embedding dimension mismatch. "
                    f"Expected {self.embed_dim}, got {embed_dim}."
                )
        else:
            if x.dim() != 3:  # noqa: PLR2004
                raise ValueError(
                    "MultiheadEnergyAttention: Expected 3D input (seq, batch, embed) "
                    "when batch_first=False. Got "
                    f"{x.dim()}D tensor."
                )
            seq_len, batch_size, embed_dim = x.shape
            if embed_dim != self.embed_dim:
                raise ValueError(
                    "MultiheadEnergyAttention: Embedding dimension mismatch. "
                    f"Expected {self.embed_dim}, got {embed_dim}."
                )
            x = x.transpose(0, 1)

        return x, batch_size, seq_len

    def _project_qk(
        self, x: Tensor, compute_dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        """Project input to queries and keys."""
        x_compute = x.to(compute_dtype)

        q = torch.einsum(
            "bse,hde->bshd",
            x_compute,
            self.q_proj_weight.to(compute_dtype),
        )

        k = torch.einsum(
            "bse,hde->bshd",
            x_compute,
            self.k_proj_weight.to(compute_dtype),
        )

        return q, k

    def _compute_attention_scores(
        self,
        q: Tensor,
        k: Tensor,
        compute_dtype: torch.dtype,
    ) -> Tensor:
        """Compute scaled attention scores."""
        return torch.einsum(
            "bshd,bthd,h->bhst",
            q,
            k,
            self.beta.to(compute_dtype),
        )

    def _apply_self_attention_mask(
        self, scores: Tensor, seq_len: int
    ) -> Tensor:
        """Apply self-attention exclusion mask (diagonal)."""
        if seq_len > 1:
            diag = torch.eye(seq_len, device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(
                diag.unsqueeze(0).unsqueeze(0),
                float("-inf"),
            )
        return scores

    def _apply_attention_masks(
        self,
        scores: Tensor,
        attn_mask: Tensor | None,
        is_causal: bool,
        seq_len: int,
        batch_size: int,
    ) -> Tensor:
        """Apply optional attention masks (causal, custom)."""
        if is_causal:
            if attn_mask is not None:
                raise ValueError(
                    "MultiheadEnergyAttention: Cannot use both is_causal=True and attn_mask. "
                    "Choose one masking strategy."
                )
            causal_mask = torch.triu(
                torch.full(
                    (seq_len, seq_len),
                    float("-inf"),
                    device=scores.device,
                    dtype=scores.dtype,
                ),
                diagonal=1,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        if attn_mask is not None:
            if attn_mask.dim() == 2:  # noqa: PLR2004
                scores = scores + attn_mask.to(scores.dtype).unsqueeze(
                    0
                ).unsqueeze(0)
            elif attn_mask.dim() == 3:  # noqa: PLR2004
                if attn_mask.size(0) != batch_size * self.num_heads:
                    raise ValueError(
                        "MultiheadEnergyAttention: Attention mask batch dimension mismatch. "
                        f"Expected {batch_size * self.num_heads}, got {attn_mask.size(0)}."
                    )
                scores_flat = scores.view(-1, seq_len, seq_len)
                scores_flat = scores_flat + attn_mask.to(scores.dtype)
                scores = scores_flat.view(
                    batch_size, self.num_heads, seq_len, seq_len
                )
            else:
                raise ValueError(
                    "MultiheadEnergyAttention: Attention mask must be 2D or 3D. "
                    f"Got {attn_mask.dim()}D tensor."
                )

        return scores

    def _compute_energy_from_scores(
        self,
        scores: Tensor,
        compute_dtype: torch.dtype,
        original_dtype: torch.dtype,
    ) -> Tensor:
        """Compute energy from attention scores."""
        lse = torch.logsumexp(scores, dim=-1)
        energy = -(lse / self.beta.view(1, -1, 1).to(compute_dtype)).sum()
        return energy.to(original_dtype)

    def _compute_gradient_terms(
        self,
        q: Tensor,
        k: Tensor,
        attn: Tensor,
        compute_dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Compute the two gradient terms."""
        f1 = torch.einsum(
            "hde,bthd->bthe",
            self.q_proj_weight.to(compute_dtype),
            k,
        )
        f2 = torch.einsum(
            "hde,bshd->bshe",
            self.k_proj_weight.to(compute_dtype),
            q,
        )
        grad1 = -torch.einsum("bthe,bhst->bse", f1, attn)
        grad2 = -torch.einsum("bshe,bhst->bte", f2, attn)
        return grad1, grad2

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Forward pass computing energy."""
        x, batch_size, seq_len = self._validate_and_prepare_input(x)

        if seq_len == 1:
            return torch.zeros((), device=x.device, dtype=x.dtype)

        compute_dtype = self._get_compute_dtype(x)

        q, k = self._project_qk(x, compute_dtype)
        scores = self._compute_attention_scores(q, k, compute_dtype)
        scores = self._apply_self_attention_mask(scores, seq_len)
        scores = self._apply_attention_masks(
            scores, attn_mask, is_causal, seq_len, batch_size
        )

        return self._compute_energy_from_scores(scores, compute_dtype, x.dtype)

    def compute_energy(self, x: Tensor) -> Tensor:
        """Compute mean energy per sample.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch, seq, embed_dim) if batch_first.

        Returns
        -------
        Tensor
            Mean energy value.
        """
        batch_size = x.shape[0] if self.batch_first else x.shape[1]
        energy = self.forward(x)
        return energy / batch_size

    def compute_grad(self, x: Tensor) -> Tensor:
        r"""Compute gradient directly without autograd.

        This implements the two-term gradient structure described in the paper,
        which includes both conventional attention and the novel energy-consistency term.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch, seq, embed_dim) if batch_first.

        Returns
        -------
        Tensor
            Gradient tensor of same shape as input.

        Notes
        -----
        The gradient has two terms as shown in section "Relationship to Modern
        Hopfield Networks and Conventional Attention" of the paper:

        .. math::
            -\\frac{\\partial E^{ATT}}{\\partial g_{iA}} = \text{Term}_1 + \text{Term}_2

        Both terms use softmax normalization over the key dimension.
        """
        needs_transpose = not self.batch_first

        x, batch_size, seq_len = self._validate_and_prepare_input(x)

        compute_dtype = self._get_compute_dtype(x)

        q, k = self._project_qk(x, compute_dtype)

        scores = self._compute_attention_scores(q, k, compute_dtype)
        scores = self._apply_self_attention_mask(scores, seq_len)
        attn = F.softmax(scores, dim=-1)

        grad1, grad2 = self._compute_gradient_terms(q, k, attn, compute_dtype)
        grad = (grad1 + grad2).to(x.dtype)

        if needs_transpose:
            grad = grad.transpose(0, 1)

        return grad

    def extra_repr(self) -> str:
        """Return string representation for printing."""
        s = f"embed_dim={self.embed_dim}, num_heads={self.num_heads}"

        # Check if beta is non-default
        default_beta = 1.0 / math.sqrt(self.head_dim)
        is_default = torch.allclose(
            self.beta, torch.full_like(self.beta, default_beta)
        )

        if not is_default:
            if self.beta.std() < ATTENTION_EPSILON:  # All same value
                s += f", beta={self.beta[0].item():.4f}"
            else:
                s += ", beta=per_head"

        if not self.batch_first:
            s += ", batch_first=False"

        return s
