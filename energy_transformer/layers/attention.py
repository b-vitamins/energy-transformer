"""Energy-based multi-head attention."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

__all__ = ["MultiheadEnergyAttention"]

from .base import EnergyModule
from .constants import MASK_FILL_VALUE
from .types import Device, Dtype
from .validation import validate_divisibility


class MultiheadEnergyAttention(EnergyModule):
    """Multi-Head Energy Attention with direct gradient computation."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        beta: float | Tensor | None = None,
        init_std: float = 0.002,
        batch_first: bool = True,
        device: Device = None,
        dtype: Dtype = None,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads

        validate_divisibility(
            embed_dim,
            num_heads,
            self.__class__.__name__,
            "embed_dim",
            "num_heads",
        )

        # Projection weights
        self.wk = nn.Parameter(
            torch.randn(
                num_heads, self.head_dim, embed_dim, device=device, dtype=dtype
            )
            * init_std
        )
        self.wq = nn.Parameter(
            torch.randn(
                num_heads, self.head_dim, embed_dim, device=device, dtype=dtype
            )
            * init_std
        )

        # Temperature parameters
        if beta is None:
            beta = 1.0 / math.sqrt(self.head_dim)

        if isinstance(beta, int | float):
            beta_tensor = torch.full(
                (num_heads,), float(beta), device=device, dtype=dtype
            )
        else:
            beta_tensor = beta.to(device=device, dtype=dtype)

        self.register_buffer("betas", beta_tensor)

    def _project(self, g: Tensor) -> tuple[Tensor, Tensor]:
        """Project input into query and key tensors."""
        k = torch.einsum("bnd,hzd->bnhz", g, self.wk)
        q = torch.einsum("bnd,hzd->bnhz", g, self.wq)
        return k, q

    def _affinity(self, q: Tensor, k: Tensor) -> Tensor:
        """Compute masked attention affinities."""
        a = torch.einsum("h,bnhz,bmhz->bhnm", self.betas, q, k)
        n = a.size(-1)
        mask = torch.eye(n, device=a.device, dtype=torch.bool)
        return a.masked_fill(mask.unsqueeze(0).unsqueeze(0), MASK_FILL_VALUE)

    def compute_energy(self, g: Tensor) -> Tensor:
        """Compute scalar energy for monitoring."""
        k, q = self._project(g)
        a = self._affinity(q, k)

        lse = torch.logsumexp(a, dim=-1)
        betas = cast(Tensor, self.betas).view(1, -1, 1)
        energy = -(lse / betas).sum()

        return energy / (g.size(0) * g.size(1))

    def compute_grad(self, g: Tensor) -> Tensor:
        """Compute gradient directly without autograd."""
        k, q = self._project(g)
        a = F.softmax(self._affinity(q, k), dim=-1)

        f1: Tensor = torch.einsum("hzd,bmhz->bmhd", self.wq, k)
        grad1: Tensor = -torch.einsum("bmhd,bhnm->bnd", f1, a)

        f2: Tensor = torch.einsum("hzd,bnhz->bnhd", self.wk, q)
        grad2: Tensor = -torch.einsum("bnhd,bhnm->bmd", f2, a)

        return grad1 + grad2

    def forward(self, x: Tensor) -> Tensor:
        """Compute energy - for compatibility with EnergyTransformer."""
        return self.compute_energy(x)
