"""Energy-based Hopfield Network implementations."""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from .base import BaseHopfieldNetwork


class HopfieldNetwork(BaseHopfieldNetwork):
    """Modern Hopfield Network with energy-based formulation.

    Parameters
    ----------
    in_dim : int
        Input dimension D of token vectors
    hidden_dim : int
        Number of memory patterns K
    energy_fn : callable, optional
        Custom energy function applied to hidden activations

    Notes
    -----
    The Hopfield Network defines the following energy function:

    E^HN = -∑_{B=1}^{N} ∑_{μ=1}^{K} G(∑_{j=1}^{D} ξ_{μj} g_{jB})

    where:
    - ξ ∈ ℝᴷˣᴰ are learnable weights (memories)
    - G(·) is an integral of the activation function r(·), with G'(·) = r(·)
    - B indexes tokens (N total)
    - μ indexes memories (K total)
    - j indexes feature dimensions (D total)

    The energy contribution is minimized when token representations align
    with rows of the memory matrix ξ. Depending on the choice of activation
    function, this behaves either as:
    - Classical continuous Hopfield Network when r(·) grows slowly (e.g., ReLU)
    - Modern continuous Hopfield Network when r(·) is sharply peaked (e.g., power, softmax)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        energy_fn: Callable[[Tensor], Tensor] = (
            lambda h: -0.5 * (F.relu(h) ** 2).sum()
        ),
    ):
        """Initialize the Hopfield Network.

        Parameters
        ----------
        in_dim : int
            Input dimension D of token vectors
        hidden_dim : int
            Number of memory patterns K
        energy_fn : callable, optional
            Custom energy function that takes the hidden activations.
            Default is -0.5 * (ReLU(h) ** 2).sum(), corresponding to
            G(z) = 0.5 * (max(0, z))² where r(z) = G'(z) = max(0, z)
            The function must return a scalar tensor.
        """
        super().__init__()

        # Memory patterns ξ ∈ ℝᴷˣᴰ
        self.ξ = nn.Parameter(torch.empty(hidden_dim, in_dim))  # shape: [K, D]

        # Default: G(z) = 0.5 * (max(0, z))² where r(z) = G'(z) = max(0, z)
        self.energy_fn = energy_fn

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters."""
        nn.init.normal_(self.ξ, std=0.002)

    def forward(self, g: Tensor) -> Tensor:
        """Compute Hopfield Network energy.

        Parameters
        ----------
        g : Tensor
            Input tensor of shape [..., N, D] where N is the number
            of tokens and D is the feature dimension

        Returns
        -------
        Tensor
            Scalar energy value representing the total Hopfield energy
        """
        # Hidden activations: h_{μB} = ∑_{j=1}^{D} ξ_{μj} g_{jB}
        h = F.linear(g, self.ξ)  # shape: [..., N, K]

        # Energy: E^HN = -∑_{B=1}^{N} ∑_{μ=1}^{K} G(h_{μB})
        e_hn = self.energy_fn(h)  # scalar

        # Ensure the energy function returns a scalar
        assert e_hn.ndim == 0, "Energy function must return a scalar tensor"

        return e_hn
