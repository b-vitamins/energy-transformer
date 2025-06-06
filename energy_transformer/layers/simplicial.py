"""Simplicial Hopfield Network with direct gradients."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .types import Device, Dtype

__all__ = ["SHNReLU", "SHNSoftmax", "SimplicialHopfieldNetwork"]


class SimplicialHopfieldNetwork(nn.Module):
    """Simplicial Hopfield Network with direct gradient computation."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        hidden_ratio: float = 4.0,
        order: int = 3,
        activation: str = "relu",
        beta: float = 0.01,
        bias: bool = True,
        init_std: float = 0.02,
        device: Device = None,
        dtype: Dtype = None,
    ) -> None:
        super().__init__()

        if order < 2:  # noqa: PLR2004
            raise ValueError(f"order must be >= 2, got {order}")

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or int(embed_dim * hidden_ratio)
        self.order = order
        self.activation = activation

        self.kernel = nn.Parameter(
            torch.randn(
                order, embed_dim, self.hidden_dim, device=device, dtype=dtype
            )
            * init_std
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    order, 1, 1, self.hidden_dim, device=device, dtype=dtype
                )
            )
        else:
            self.register_parameter("bias", None)

        if activation == "softmax":
            self.beta = nn.Parameter(
                torch.tensor(beta, device=device, dtype=dtype)
            )
        else:
            self.register_buffer("beta", None)

    def compute_energy(self, g: Tensor) -> Tensor:
        """Compute energy for monitoring."""
        h = torch.einsum("bnd,vdk->bvnk", g, self.kernel)

        if self.bias is not None:
            h = h + self.bias

        if self.activation == "relu":
            a_v = F.relu(h)
            sum_a = a_v.sum(dim=1)
            energy = -0.5 / self.order * (sum_a**2).sum()
        else:
            h = h * self.beta
            lse = torch.logsumexp(h, dim=-1)
            energy = -(1.0 / (self.order * self.beta)) * lse.sum()

        return energy / (g.size(0) * g.size(1))

    def compute_grad(self, g: Tensor) -> Tensor:
        """Compute gradient directly."""
        h = torch.einsum("bnd,vdk->bvnk", g, self.kernel)

        if self.bias is not None:
            h = h + self.bias

        if self.activation == "relu":
            a_v = F.relu(h)
            sum_a = a_v.sum(dim=1, keepdim=True)
            indicator = (h > 0).float()
            factor = (sum_a * indicator) / self.order
            grad = -torch.einsum("vdk,bvnk->bnd", self.kernel, factor)
        else:
            h = h * self.beta
            a_v = F.softmax(h, dim=-1)
            grad = -torch.einsum("vdk,bvnk->bnd", self.kernel, a_v) / self.order

        return grad

    def forward(self, g: Tensor) -> Tensor:
        """Compute energy - for compatibility."""
        return self.compute_energy(g)


class SHNReLU(SimplicialHopfieldNetwork):
    """Classical Simplicial Hopfield Network with ReLU activation."""

    def __init__(
        self, embed_dim: int, **kwargs: int | float | Device | Dtype
    ) -> None:
        super().__init__(embed_dim=embed_dim, activation="relu", **kwargs)


class SHNSoftmax(SimplicialHopfieldNetwork):
    """Modern Simplicial Hopfield Network with softmax activation."""

    def __init__(
        self,
        embed_dim: int,
        beta: float = 0.01,
        **kwargs: int | float | Device | Dtype,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim, activation="softmax", beta=beta, **kwargs
        )
