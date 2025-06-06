"""Energy-based multi-head attention module implementation."""

import math
from typing import Any, ClassVar, Literal, overload

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .constants import (
    ATTENTION_EPSILON,
    ATTENTION_INIT_STD,
    MEMORY_EFFICIENT_SEQ_THRESHOLD,
)
from .types import Device, Dtype, EmbedDim, NumHeads
from .validation import (
    validate_divisibility,
    validate_positive,
)


class MultiheadEnergyAttention(nn.Module):
    r"""Multi-Head Energy Attention mechanism.

    Mathematical Foundation
    -----------------------
    The energy-based attention exchanges information between tokens (particles)
    by aligning queries and keys in internal space. The energy function is:

    .. math::
        E^{ATT} = -\frac{1}{\beta} \sum_{h=1}^{H} \sum_{C=1}^{N} \log\left(\sum_{B \neq C} \exp(\beta A_{hBC})\right)

    where the attention matrix is computed as:

    .. math::
        A_{hBC} &= \sum_\alpha K_{\alpha hB} Q_{\alpha hC} \\
        K_{\alpha hB} &= \sum_j W^K_{\alpha hj} g_{jB} \\
        Q_{\alpha hC} &= \sum_j W^Q_{\alpha hj} g_{jC}

    The gradient contribution to token updates is:

    .. math::
        -\frac{\partial E^{ATT}}{\partial g_{iA}} = \sum_{C \neq A} \sum_\alpha W^Q_{\alpha i} K_{\alpha C} \text{softmax}_C\left(\beta \sum_\gamma K_{\gamma C} Q_{\gamma A}\right) \\
        \quad + W^K_{\alpha i} Q_{\alpha C} \text{softmax}_A\left(\beta \sum_\gamma K_{\gamma A} Q_{\gamma C}\right)

    The first term is conventional attention with ``V = (W^Q)^T K``. The second
    term is novel and crucial for ensuring energy minimization under recurrent
    application. This distinguishes Energy Attention from Modern Hopfield
    Networks.

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

    Notes
    -----
    Performance Considerations:
    - For sequences longer than 512, consider ``use_memory_efficient=True``.
    - Mixed precision (float16/bfloat16) is automatically handled.
    - Avoid creating attention masks on every forward pass - cache them.
    - The module is optimized for ``batch_first=True`` (default).
    """

    __constants__: ClassVar[list[str]] = [
        "embed_dim",
        "num_heads",
        "batch_first",
    ]
    beta: Tensor

    def __init__(
        self,
        embed_dim: EmbedDim,
        num_heads: NumHeads,
        beta: float | Tensor | None = None,
        init_std: float = ATTENTION_INIT_STD,
        batch_first: bool = True,
        device: Device = None,
        dtype: Dtype = None,
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
        """Dimension of each attention head (Y in paper notation)."""
        return self.embed_dim // self.num_heads

    @property
    def total_params(self) -> int:
        """Total number of parameters in this module."""
        return 2 * self.num_heads * self.head_dim * self.embed_dim

    @property
    def requires_grad_(self) -> bool:  # type: ignore[override]
        """Check if any parameter requires gradients."""
        return any(p.requires_grad for p in self.parameters())

    @property
    def device(self) -> torch.device:
        """Device of the module parameters."""
        return self.q_proj_weight.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the module parameters."""
        return self.q_proj_weight.dtype

    @property
    def is_mixed_precision(self) -> bool:
        """Whether mixed precision computation is recommended for current dtype."""
        return self.q_proj_weight.dtype in {torch.float16, torch.bfloat16}

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
            if embed_dim not in {self.embed_dim, 1}:
                raise ValueError(
                    "MultiheadEnergyAttention: Embedding dimension mismatch. "
                    f"Expected {self.embed_dim} or 1, got {embed_dim}."
                )
        else:
            if x.dim() != 3:  # noqa: PLR2004
                raise ValueError(
                    "MultiheadEnergyAttention: Expected 3D input (seq, batch, embed) "
                    "when batch_first=False. Got "
                    f"{x.dim()}D tensor."
                )
            seq_len, batch_size, embed_dim = x.shape
            if embed_dim not in {self.embed_dim, 1}:
                raise ValueError(
                    "MultiheadEnergyAttention: Embedding dimension mismatch. "
                    f"Expected {self.embed_dim} or 1, got {embed_dim}."
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
                if attn_mask.size(0) not in {1, batch_size * self.num_heads}:
                    raise ValueError(
                        "MultiheadEnergyAttention: Attention mask batch dimension mismatch. "
                        f"Expected 1 or {batch_size * self.num_heads}, got {attn_mask.size(0)}."
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

    @overload
    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: None = None,
        is_causal: Literal[False] = False,
        use_memory_efficient: bool = False,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: Tensor,
        is_causal: Literal[False] = False,
        use_memory_efficient: bool = False,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: None = None,
        is_causal: Literal[True],
        use_memory_efficient: bool = False,
    ) -> Tensor: ...

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
        use_memory_efficient: bool = False,
    ) -> Tensor:
        """Forward pass computing energy."""
        if __debug__:
            assert isinstance(x, Tensor), (
                "MultiheadEnergyAttention: Expected Tensor input, "
                f"got {type(x).__name__}"
            )
            assert x.dim() == 3, f"Expected 3D input, got {x.dim()}D"  # noqa: PLR2004

        x, batch_size, seq_len = self._validate_and_prepare_input(x)

        if seq_len == 1:
            return torch.zeros((), device=x.device, dtype=x.dtype)

        if use_memory_efficient and seq_len > MEMORY_EFFICIENT_SEQ_THRESHOLD:
            return self._forward_chunked(x, attn_mask, is_causal)

        compute_dtype = self._get_compute_dtype(x)

        q, k = self._project_qk(x, compute_dtype)
        scores = self._compute_attention_scores(q, k, compute_dtype)
        scores = self._apply_self_attention_mask(scores, seq_len)
        scores = self._apply_attention_masks(
            scores, attn_mask, is_causal, seq_len, batch_size
        )
        if __debug__:
            assert not torch.isnan(scores).any(), "NaN in attention scores"

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

    def _forward_chunked(  # noqa: PLR0912,C901
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        is_causal: bool,
        chunk_size: int = 128,
    ) -> Tensor:
        """Memory-efficient forward using chunked attention computation."""
        x, batch_size, seq_len = self._validate_and_prepare_input(x)
        compute_dtype = self._get_compute_dtype(x)

        q, k = self._project_qk(x, compute_dtype)

        diag = torch.eye(seq_len, device=x.device, dtype=torch.bool)
        if is_causal:
            if attn_mask is not None:
                raise ValueError(
                    "MultiheadEnergyAttention: Cannot use both is_causal=True and attn_mask."
                )
            causal_full = torch.triu(
                torch.full(
                    (seq_len, seq_len),
                    float("-inf"),
                    device=x.device,
                    dtype=compute_dtype,
                ),
                diagonal=1,
            )
        else:
            causal_full = None

        if attn_mask is not None:
            if attn_mask.dim() == 2:  # noqa: PLR2004
                attn_full = attn_mask.to(compute_dtype)
            elif attn_mask.dim() == 3:  # noqa: PLR2004
                if attn_mask.size(0) not in {1, batch_size * self.num_heads}:
                    raise ValueError(
                        "MultiheadEnergyAttention: Attention mask batch dimension mismatch."
                    )
                attn_full = attn_mask.to(compute_dtype).view(
                    -1, seq_len, seq_len
                )
            else:
                raise ValueError(
                    "MultiheadEnergyAttention: Attention mask must be 2D or 3D."
                )

        energy = torch.zeros((), device=x.device, dtype=compute_dtype)
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            q_chunk = q[:, start:end]
            scores = self._compute_attention_scores(q_chunk, k, compute_dtype)

            self_mask = diag[start:end].unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(self_mask, float("-inf"))

            if is_causal:
                assert causal_full is not None
                scores = scores + causal_full[start:end].unsqueeze(0).unsqueeze(
                    0
                )

            if attn_mask is not None:
                if attn_mask.dim() == 2:  # noqa: PLR2004
                    scores = scores + attn_full[start:end].unsqueeze(
                        0
                    ).unsqueeze(0)
                else:
                    scores_flat = scores.view(-1, end - start, seq_len)
                    scores_flat = scores_flat + attn_full[:, start:end]
                    scores = scores_flat.view(
                        batch_size, self.num_heads, end - start, seq_len
                    )

            lse = torch.logsumexp(scores, dim=-1)
            energy = (
                energy
                - (lse / self.beta.view(1, -1, 1).to(compute_dtype)).sum()
            )

        return energy.to(x.dtype)

    def extra_repr(self) -> str:
        """Return string representation for module printing."""
        parts = [f"embed_dim={self.embed_dim}", f"num_heads={self.num_heads}"]

        default_beta = 1.0 / math.sqrt(self.head_dim)
        is_default = torch.allclose(
            self.beta, torch.full_like(self.beta, default_beta)
        )
        if not is_default:
            if self.beta.std() < ATTENTION_EPSILON:
                parts.append(f"beta={self.beta[0].item():.4f}")
            else:
                parts.append("beta=per_head")

        if not self.batch_first:
            parts.append("batch_first=False")

        return ", ".join(parts)

    # ------------------------------------------------------------------
    # State dict hooks
    # ------------------------------------------------------------------
    def _save_to_state_dict(
        self,
        destination: dict[str, Any],
        prefix: str,
        keep_vars: bool,
    ) -> None:
        """Save module state with additional metadata."""
        super()._save_to_state_dict(destination, prefix, keep_vars)  # type: ignore[misc,no-untyped-call]
        destination[prefix + "_metadata.version"] = "1.0"
        destination[prefix + "_metadata.config"] = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
        }
