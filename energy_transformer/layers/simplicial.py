"""Simplicial Hopfield Network implementation.

This module implements higher-order Hopfield networks operating on simplicial
complexes, generalizing the standard Hopfield network to k-way interactions
between tokens as described in "Simplicial Hopfield Networks" (2023).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .constants import (
    DEFAULT_HOPFIELD_BETA,
    DEFAULT_HOPFIELD_MULTIPLIER,
    DEFAULT_INIT_STD,
)
from .types import ActivationType, Device, Dtype, EmbedDim, HiddenDim
from .validation import validate_positive, validate_tensor_dim

__all__ = ["SHNReLU", "SHNSoftmax", "SimplicialHopfieldNetwork"]


class SimplicialHopfieldNetwork(nn.Module):
    r"""Energy-based Simplicial Hopfield Network module.

    Mathematical Foundation
    -----------------------
    The Simplicial Hopfield Network extends memory storage to ``k``-way
    interactions between tokens. The energy function is:

    .. math::
        E^{SHN} = -\frac{1}{k} \sum_{B=1}^{N} \sum_{\mu=1}^{K} \left(\sum_{v=1}^{k} r\left(\sum_{j=1}^{D} \xi_{v,\mu j} g_{jB}\right)\right)^2

    where ``k`` is the order of the simplex, ``\xi_v`` are memory patterns for
    each vertex, and ``r`` is the activation function per vertex.

    The gradient is:

    .. math::
        -\frac{\partial E^{SHN}}{\partial g_{iA}} = \frac{1}{k} \sum_{v=1}^{k} \sum_{\mu=1}^{K} \xi_{v,\mu i} \left(\sum_{u=1}^{k} r_{\mu u}\right) r'_{\mu v}

    For softmax activation the expression simplifies to

    .. math::
        -\frac{\partial E^{SHN}}{\partial g_{iA}} = \frac{1}{k} \sum_{v=1}^{k} \xi_v^T \text{softmax}(\beta h_v)
    """

    activation: ActivationType
    beta: nn.Parameter | None

    def __init__(
        self,
        embed_dim: EmbedDim,
        hidden_dim: HiddenDim | None = None,
        hidden_ratio: float = DEFAULT_HOPFIELD_MULTIPLIER,
        order: int = 3,
        activation: ActivationType = "relu",
        beta: float = DEFAULT_HOPFIELD_BETA,
        bias: bool = False,
        init_std: float = DEFAULT_INIT_STD,
        device: Device = None,
        dtype: Dtype = None,
    ) -> None:
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        if order < 2:  # noqa: PLR2004
            raise ValueError(
                f"SimplicialHopfieldNetwork: order must be >= 2. Got: {order}. "
                "Use HopfieldNetwork for order=2."
            )

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or int(embed_dim * hidden_ratio)
        self.order = order
        self.activation = activation
        self.use_bias = bias
        self.init_std = init_std

        if activation not in ["relu", "softmax"]:
            raise ValueError(
                "SimplicialHopfieldNetwork: activation must be 'relu' or 'softmax'. "
                f"Got: '{activation}'."
            )

        self.kernel = nn.Parameter(
            torch.empty((order, embed_dim, self.hidden_dim), **factory_kwargs)  # type: ignore[arg-type]
        )

        if self.use_bias:
            self.bias = nn.Parameter(
                torch.zeros((order, 1, 1, self.hidden_dim), **factory_kwargs)  # type: ignore[arg-type]
            )
        else:
            self.register_parameter("bias", None)

        if activation == "softmax":
            validate_positive(beta, "SimplicialHopfieldNetwork", "beta")
            self.beta = nn.Parameter(torch.tensor(beta, **factory_kwargs))  # type: ignore[arg-type]
        else:
            self.register_buffer("beta", None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.kernel, std=self.init_std)
        # Bias initialized to zeros; keep beta as provided

    def _project(self, g: Tensor) -> Tensor:
        kernel = self.kernel.to(g.dtype)
        return torch.einsum("...nd,vdk->...vnk", g, kernel)

    def forward(self, g: Tensor) -> Tensor:
        """Compute energy for input tokens."""
        validate_tensor_dim(g, 3, "SimplicialHopfieldNetwork", "g")

        h = self._project(g)
        if self.use_bias:
            bias = self.bias.to(g.dtype).view(1, self.order, 1, self.hidden_dim)
            h = h + bias

        if self.activation == "relu":
            a_v = F.relu(h, inplace=True)
            sum_a = a_v.sum(dim=-3)
            energy = -0.5 / self.order * sum_a.pow(2).sum()
        else:
            assert self.beta is not None
            h = h * self.beta
            lse = torch.logsumexp(h, dim=-1)
            energy = -(1.0 / (self.order * self.beta)) * lse.sum()
        return energy

    def compute_grad(self, g: Tensor) -> Tensor:
        """Return gradient of energy with respect to input."""
        h = self._project(g)
        if self.use_bias:
            bias = self.bias.to(g.dtype).view(1, self.order, 1, self.hidden_dim)
            h = h + bias

        if self.activation == "relu":
            a_v = F.relu(h)
            sum_a = a_v.sum(dim=-3, keepdim=True)
            indicator = (h > 0).type_as(h)
            factor = (sum_a * indicator) / self.order
            grad = -torch.einsum(
                "vdk,...vnk->...nd", self.kernel.to(g.dtype), factor
            )
        else:
            assert self.beta is not None
            h = h * self.beta
            a_v = F.softmax(h, dim=-1)
            grad = (
                -torch.einsum("vdk,...vnk->...nd", self.kernel.to(g.dtype), a_v)
                / self.order
            )
        return grad

    @property
    def memory_dim(self) -> int:
        """Number of memory patterns stored."""
        return self.hidden_dim

    @property
    def input_dim(self) -> int:
        """Input feature dimension."""
        return self.embed_dim

    @property
    def simplex_order(self) -> int:
        """Order of simplicial interactions."""
        return self.order

    @property
    def activation_type(self) -> str:
        """Activation function type."""
        return self.activation

    @property
    def is_classical(self) -> bool:
        """Whether activation is ReLU."""
        return self.activation == "relu"

    @property
    def is_modern(self) -> bool:
        """Whether activation is softmax."""
        return self.activation == "softmax"

    @property
    def temperature(self) -> float | None:
        """Temperature parameter for softmax."""
        if self.activation == "softmax":
            assert self.beta is not None
            return self.beta.item()
        return None

    @property
    def total_params(self) -> int:
        """Total number of learnable parameters."""
        param_count = self.order * self.embed_dim * self.hidden_dim
        if self.use_bias:
            param_count += self.order * self.hidden_dim
        if self.activation == "softmax" and isinstance(self.beta, nn.Parameter):
            param_count += 1
        return param_count

    @property
    def device(self) -> torch.device:
        """Device of module parameters."""
        return self.kernel.device

    def extra_repr(self) -> str:
        """Return string representation for module printing."""
        parts = [
            f"embed_dim={self.embed_dim}",
            f"hidden_dim={self.hidden_dim}",
            f"order={self.order}",
            f"activation='{self.activation}'",
        ]
        if self.activation == "softmax":
            assert self.beta is not None
            parts.append(f"beta={self.beta.item():.3f}")
        if self.use_bias:
            parts.append("bias=True")
        return ", ".join(parts)


class SHNReLU(SimplicialHopfieldNetwork):
    """Classical Simplicial Hopfield Network with ReLU activation."""

    def __init__(
        self,
        embed_dim: int,
        hidden_ratio: float = DEFAULT_HOPFIELD_MULTIPLIER,
        order: int = 3,
        bias: bool = False,
        init_std: float = DEFAULT_INIT_STD,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            hidden_ratio=hidden_ratio,
            order=order,
            activation="relu",
            bias=bias,
            init_std=init_std,
            device=device,
            dtype=dtype,
        )


class SHNSoftmax(SimplicialHopfieldNetwork):
    """Modern Simplicial Hopfield Network with softmax activation."""

    def __init__(
        self,
        embed_dim: int,
        hidden_ratio: float = DEFAULT_HOPFIELD_MULTIPLIER,
        order: int = 3,
        beta: float = DEFAULT_HOPFIELD_BETA,
        bias: bool = False,
        init_std: float = DEFAULT_INIT_STD,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            hidden_ratio=hidden_ratio,
            order=order,
            activation="softmax",
            beta=beta,
            bias=bias,
            init_std=init_std,
            device=device,
            dtype=dtype,
        )
