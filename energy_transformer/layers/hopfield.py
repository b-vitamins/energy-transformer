"""Energy-based Hopfield Network implementations."""

import warnings
from collections.abc import Callable
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from .base import BaseHopfieldNetwork


class ActivationFunction(Enum):
    """Activation functions for Hopfield Network energy computation."""

    RELU = "relu"
    SOFTMAX = "softmax"
    POWER = "power"
    CUSTOM = "custom"


class HopfieldNetwork(BaseHopfieldNetwork):
    """Modern Hopfield Network with energy-based formulation.

    Parameters
    ----------
    in_dim : int
        Input dimension D of token vectors
    hidden_dim : int, optional
        Number of memory patterns K. If not provided, computed as
        int(in_dim * multiplier)
    multiplier : float, optional
        Multiplier for hidden dimension when hidden_dim is not specified.
        Default is 4.0
    bias : bool, optional
        Whether to include bias term. Default is False
    activation : ActivationFunction or callable, optional
        Activation function for energy computation. Default is RELU
    energy_fn : callable, optional
        Custom energy function applied to hidden activations. Only used
        when activation=CUSTOM

    Notes
    -----
    The Hopfield Network defines the following energy function:
    E^HN = -∑_{B=1}^{N} ∑_{μ=1}^{K} G(∑_{j=1}^{D} ξ_{μj} g_{jB} + b_μ)
    where:
    - ξ ∈ ℝᴷˣᴰ are learnable weights (memories)
    - b ∈ ℝᴷ are optional bias terms
    - G(·) is an integral of the activation function r(·), with G'(·) = r(·)
    - B indexes tokens (N total)
    - μ indexes memories (K total)
    - j indexes feature dimensions (D total)

    The energy contribution is minimized when token representations align
    with rows of the memory matrix ξ. Depending on the choice of activation
    function, this behaves either as:
    - Classical continuous Hopfield Network when r(·) grows slowly (e.g., ReLU)
    - Modern continuous Hopfield Network when r(·) is sharply peaked
      (e.g., exponential, softmax)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | None = None,
        multiplier: float = 4.0,
        bias: bool = False,
        activation: ActivationFunction | Callable[[Tensor], Tensor] = (
            ActivationFunction.RELU
        ),
        energy_fn: Callable[[Tensor], Tensor] | None = None,
    ):
        """Initialize the Hopfield Network.

        Parameters
        ----------
        in_dim : int
            Input dimension D of token vectors
        hidden_dim : int, optional
            Number of memory patterns K. If not provided, computed as
            int(in_dim * multiplier)
        multiplier : float, optional
            Multiplier for hidden dimension when hidden_dim is not specified
        bias : bool, optional
            Whether to include bias term
        activation : ActivationFunction or callable, optional
            Activation function for energy computation
        energy_fn : callable, optional
            Custom energy function that takes the hidden activations.
            Only used when activation=CUSTOM. Must return a scalar tensor.
        """
        super().__init__()

        # Determine hidden dimension
        if hidden_dim is None:
            hidden_dim = int(in_dim * multiplier)
        self.hidden_dim = hidden_dim

        # Memory patterns ξ ∈ ℝᴷˣᴰ
        self.ξ = nn.Parameter(torch.empty(hidden_dim, in_dim))  # shape: [K, D]

        # Optional bias term
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_dim))  # shape: [K]
        else:
            self.register_parameter("bias", None)

        # Warn if energy_fn provided but not using CUSTOM activation
        if (
            energy_fn is not None
            and isinstance(activation, ActivationFunction)
            and activation != ActivationFunction.CUSTOM
        ):
            warnings.warn(
                "energy_fn provided but activation is not CUSTOM. "
                "The energy_fn will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        # Set up energy function
        self.activation = activation
        self.energy_fn: Callable[[Tensor], Tensor]

        if isinstance(activation, ActivationFunction):
            if activation == ActivationFunction.RELU:

                def relu_energy(h: Tensor) -> Tensor:
                    return -0.5 * (F.relu(h) ** 2).sum()

                self.energy_fn = relu_energy
            elif activation == ActivationFunction.SOFTMAX:
                # Canonical modern Hopfield: G(z) = exp(z)
                def softmax_energy(h: Tensor) -> Tensor:
                    return -torch.exp(h).sum()

                self.energy_fn = softmax_energy
            elif activation == ActivationFunction.POWER:
                # Scaled to prevent explosion
                def power_energy(h: Tensor) -> Tensor:
                    return -(h.pow(4).mean())

                self.energy_fn = power_energy
            elif activation == ActivationFunction.CUSTOM:
                if energy_fn is None:
                    raise ValueError(
                        "Must provide energy_fn when using CUSTOM activation"
                    )
                self.energy_fn = energy_fn
        else:
            # Assume it's a callable
            self.energy_fn = activation

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters."""
        nn.init.normal_(self.ξ, std=0.02)
        if self.use_bias:
            nn.init.zeros_(self.bias)

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
        # Mixed-precision safety: cast ξ to input dtype
        if g.dtype in {torch.float16, torch.bfloat16}:
            ξ_cast = self.ξ.to(g.dtype)
        else:
            ξ_cast = self.ξ

        # Hidden activations: h_{μB} = ∑_{j=1}^{D} ξ_{μj} g_{jB}
        h = F.linear(g, ξ_cast)  # shape: [..., N, K]

        # Add bias if present
        if self.use_bias and self.bias is not None:
            h = h + self.bias.to(g.dtype)

        # Energy: E^HN = -∑_{B=1}^{N} ∑_{μ=1}^{K} G(h_{μB})
        e_hn = self.energy_fn(h)  # scalar

        # Ensure the energy function returns a scalar
        assert e_hn.ndim == 0, "Energy function must return a scalar tensor"

        return e_hn
