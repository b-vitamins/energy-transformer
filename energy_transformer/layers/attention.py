"""Energy-based multi-head attention module implementation.

This module defines an energy-based attention mechanism, where attention operations are
implicitly defined via an energy function. The gradient of this energy function produces
the attention outputs.
"""

import math

import torch
import torch.nn as nn


class EnergyAttention(nn.Module):
    """Energy-based multi-head attention.

    Instead of directly computing attention weights and outputs, this module
    defines an energy function whose gradient gives the attention operation.

    Parameters
    ----------
    d_model : int
        Dimension of the model embeddings.
    n_heads : int
        Number of attention heads.
    d_head : int
        Dimension per attention head.
    per_head_beta : bool, default False
        Whether to use separate temperature parameters per head.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        per_head_beta: bool = False,
    ):
        """Initialize the EnergyAttention module."""
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.per_head_beta = per_head_beta

        # Initialize query and key projection matrices
        self.wq = nn.Parameter(torch.randn(n_heads, d_head, d_model) * 0.002)
        self.wk = nn.Parameter(torch.randn(n_heads, d_head, d_model) * 0.002)

        # Temperature parameters
        if per_head_beta:
            self.beta = nn.Parameter(torch.ones(n_heads) / math.sqrt(d_head))
        else:
            self.register_buffer("beta", torch.tensor(1.0 / math.sqrt(d_head)))

    def _broadcast_beta(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Broadcast temperature parameter beta for scores.

        Parameters
        ----------
        scores : torch.Tensor
            Scores tensor of shape (..., n_heads, query_len, key_len).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing beta and inverse beta tensors broadcastable to scores.
        """
        if self.per_head_beta:
            extra_dims = scores.dim() - 3
            head_shape = [1] * extra_dims + [self.n_heads, 1, 1]
            beta = self.beta.view(head_shape)  # (..., n_heads, 1, 1)
            inv_beta = (1.0 / self.beta).view(
                [1] * extra_dims + [self.n_heads, 1]
            )  # (..., n_heads, 1)
        else:
            beta = self.beta
            inv_beta = 1.0 / self.beta
        return beta, inv_beta

    def energy(self, g: torch.Tensor) -> torch.Tensor:
        """Compute attention energy.

        Parameters
        ----------
        g : torch.Tensor
            Input tensor with shape (..., seq_len, d_model).

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the computed energy.
        """
        # (1) Linear projections
        query = torch.einsum(
            "...qd,hzd->...qhz", g, self.wq
        )  # (..., query_len, n_heads, d_head)
        key = torch.einsum(
            "...kd,hzd->...khz", g, self.wk
        )  # (..., key_len, n_heads, d_head)

        # (2) Dot-product scores in head space
        scores = torch.einsum(
            "...qhz,...khz->...hqk", query, key
        )  # (..., n_heads, query_len, key_len)

        # (3) Temperature scaling and log-partition
        beta, inv_beta = self._broadcast_beta(scores)
        log_partition = torch.logsumexp(
            beta * scores, dim=-1
        )  # (..., n_heads, query_len)

        # (4) Aggregate log-partitions to compute main energy term
        energy_main = -(inv_beta * log_partition).sum()

        # (5) Positional symmetry-breaking term
        seq_len = g.shape[-2]
        pos_ids = torch.linspace(
            -0.5, 0.5, seq_len, device=g.device, dtype=g.dtype
        )
        pos_ids = pos_ids.view(*([1] * (g.dim() - 2)), seq_len, 1)
        pos_term = (g * pos_ids).sum() * 1e-3

        return energy_main + pos_term

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient of energy w.r.t. input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (..., seq_len, d_model).

        Returns
        -------
        torch.Tensor
            Gradient tensor of the same shape as `x`.
        """
        x.requires_grad_(True)
        energy = self.energy(x)
        grad = torch.autograd.grad(energy, x, create_graph=True)[0]
        return grad
