"""Hopfield Network with explicit energy gradients."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

__all__ = ["HopfieldNetwork"]

from .base import EnergyModule
from .types import Device, Dtype


class HopfieldNetwork(EnergyModule):
    """Hopfield Network with direct gradient computation.

    Parameters
    ----------
    embed_dim : int
        Input embedding dimension ``D``.
    hidden_dim : int, optional
        Hidden dimension. Defaults to ``hidden_ratio * embed_dim``.
    hidden_ratio : float, default=4.0
        Ratio of hidden to input dimension when ``hidden_dim`` is ``None``.
    activation : {'relu', 'softmax'}, default='relu'
        Activation function used to compute pattern activations.
    beta : float, default=0.01
        Inverse temperature when ``activation='softmax'``.
    bias : bool, default=True
        If ``True`` includes bias parameters.
    init_std : float, default=0.02
        Standard deviation for weight initialization.
    device : Device, optional
        Device for parameters.
    dtype : Dtype, optional
        Data type for parameters.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        hidden_ratio: float = 4.0,
        activation: str = "relu",
        beta: float = 0.01,
        bias: bool = True,
        init_std: float = 0.02,
        device: Device = None,
        dtype: Dtype = None,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or int(embed_dim * hidden_ratio)
        if activation not in {"relu", "softmax"}:
            raise ValueError(
                f"{self.__class__.__name__}: activation must be 'relu' or 'softmax',"
                f" got {activation!r}."
            )
        self.activation = activation

        self.kernel = nn.Parameter(
            torch.randn(embed_dim, self.hidden_dim, device=device, dtype=dtype)
            * init_std
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.hidden_dim, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        if activation == "softmax":
            self.beta = nn.Parameter(
                torch.tensor(beta, device=device, dtype=dtype)
            )
        else:
            self.register_buffer("beta", None)

    def _preactivate(self, g: Tensor) -> Tensor:
        """Linear transform with optional bias."""
        h = torch.einsum("bnd,dk->bnk", g, self.kernel)
        if self.bias is not None:
            h = h + self.bias
        return h

    def compute_energy(self, g: Tensor) -> Tensor:
        """Compute energy for monitoring.

        Parameters
        ----------
        g : Tensor
            Input tensor of shape ``(B, N, D)``.

        Returns
        -------
        Tensor
            Scalar energy averaged over batch and sequence length.
        """
        h = self._preactivate(g)

        if self.activation == "relu":
            a = F.relu(h)
            energy = -0.5 * (a**2).sum()
        else:
            h = h * self.beta
            lse = torch.logsumexp(h, dim=-1)
            energy = -(1.0 / self.beta) * lse.sum()

        return energy / (g.size(0) * g.size(1))

    def compute_grad(self, g: Tensor) -> Tensor:
        """Compute gradient directly.

        Parameters
        ----------
        g : Tensor
            Input tensor of shape ``(B, N, D)``.

        Returns
        -------
        Tensor
            Gradient tensor of the same shape as ``g``.
        """
        h = self._preactivate(g)

        if self.activation == "relu":
            a = F.relu(h)
        else:
            h = h * self.beta
            a = F.softmax(h, dim=-1)

        return -torch.einsum("bnk,kd->bnd", a, self.kernel.T)

    def forward(self, g: Tensor) -> Tensor:
        """Return energy for compatibility with :class:`EnergyTransformer`."""
        return self.compute_energy(g)
