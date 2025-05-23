"""Energy-based Hopfield Network implementations."""

import math
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
    TANH = "tanh"
    CUSTOM = "custom"


# Registry for energy functions
_ENERGY_REGISTRY: dict[str, Callable[[Tensor], Tensor]] = {}


def register_energy_function(
    name: str,
) -> Callable[[Callable[[Tensor], Tensor]], Callable[[Tensor], Tensor]]:
    """Register custom energy functions as a decorator.

    Parameters
    ----------
    name : str
        Name of the energy function for registration

    Returns
    -------
    callable
        Decorator function

    Examples
    --------
    >>> @register_energy_function("my_energy")
    ... def my_custom_energy(h: Tensor) -> Tensor:
    ...     return -torch.sum(torch.sigmoid(h))
    """

    def decorator(
        energy_fn: Callable[[Tensor], Tensor],
    ) -> Callable[[Tensor], Tensor]:
        _ENERGY_REGISTRY[name] = energy_fn
        return energy_fn

    return decorator


def get_energy_function(
    activation: ActivationFunction,
) -> Callable[[Tensor], Tensor]:
    """Get energy function for given activation type.

    Parameters
    ----------
    activation : ActivationFunction
        Activation function enum

    Returns
    -------
    callable
        Energy function that takes hidden activations and returns scalar energy
    """
    if activation == ActivationFunction.RELU:

        def relu_energy(h: Tensor) -> Tensor:
            """Classical Hopfield: G(z) = 0.5 * z² for z > 0, 0 otherwise."""
            return -0.5 * (F.relu(h) ** 2).sum()

        _ENERGY_REGISTRY["relu"] = relu_energy
        return relu_energy

    elif activation == ActivationFunction.SOFTMAX:

        def softmax_energy(h: Tensor) -> Tensor:
            """Modern Hopfield: G(z) = exp(z)."""
            return -torch.exp(h).sum()

        _ENERGY_REGISTRY["softmax"] = softmax_energy
        return softmax_energy

    elif activation == ActivationFunction.POWER:

        def power_energy(h: Tensor) -> Tensor:
            """Power-law energy: G(z) = z⁴/4 (scaled to prevent explosion)."""
            return -(h.pow(4).mean())

        _ENERGY_REGISTRY["power"] = power_energy
        return power_energy

    elif activation == ActivationFunction.TANH:

        def tanh_energy(h: Tensor) -> Tensor:
            """Hyperbolic tangent energy: G(z) = log(cosh(z))."""
            # Clamp for stability
            return -torch.sum(torch.log(torch.cosh(h.clamp(-10, 10))))

        _ENERGY_REGISTRY["tanh"] = tanh_energy
        return tanh_energy

    else:
        raise ValueError(f"Unknown activation function: {activation}")


def he_scaled_init_std(fan_in: int, fan_out: int | None = None) -> float:
    """Compute He-scaled initialization standard deviation.

    Parameters
    ----------
    fan_in : int
        Number of input features
    fan_out : int, optional
        Number of output features. If None, uses fan_in

    Returns
    -------
    float
        Standard deviation for He initialization
    """
    effective_fan = fan_out if fan_out is not None else fan_in
    return math.sqrt(2.0 / effective_fan)


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
    batched_memories : bool, optional
        Whether to support batched memory patterns. Default is False
    num_memory_sets : int, optional
        Number of memory sets when using batched_memories. Default is 1
    debug_checks : bool, optional
        Whether to enable debug checks for NaN/inf values. Default is False

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
    - Classical continuous Hopfield Network when r(·) grows slowly (ReLU)
    - Modern continuous Hopfield Network when r(·) is sharply peaked
      (exponential, softmax)

    For batched memories, the weight tensor becomes ξ ∈ ℝᴹˣᴷˣᴰ where M is
    the number of memory sets, allowing different memory patterns per batch.
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
        batched_memories: bool = False,
        num_memory_sets: int = 1,
        debug_checks: bool = False,
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
        batched_memories : bool, optional
            Whether to support batched memory patterns
        num_memory_sets : int, optional
            Number of memory sets when using batched_memories
        debug_checks : bool, optional
            Whether to enable debug checks for NaN/inf values
        """
        super().__init__()

        # Determine hidden dimension
        if hidden_dim is None:
            hidden_dim = int(in_dim * multiplier)
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.batched_memories = batched_memories
        self.num_memory_sets = num_memory_sets if batched_memories else 1
        self.debug_checks = debug_checks

        # Memory patterns ξ ∈ ℝᴷˣᴰ or ℝᴹˣᴷˣᴰ for batched memories
        if batched_memories:
            self.ξ = nn.Parameter(
                torch.empty(num_memory_sets, hidden_dim, in_dim)
            )  # shape: [M, K, D]
        else:
            self.ξ = nn.Parameter(
                torch.empty(hidden_dim, in_dim)
            )  # shape: [K, D]

        # Optional bias term
        self.use_bias = bias
        if bias:
            if batched_memories:
                self.bias = nn.Parameter(
                    torch.zeros(num_memory_sets, hidden_dim)
                )  # shape: [M, K]
            else:
                self.bias = nn.Parameter(torch.zeros(hidden_dim))  # shape: [K]
        else:
            self.register_parameter("bias", None)

        # Set up energy function
        self.activation = activation
        self._setup_energy_function(activation, energy_fn)

        self.reset_parameters()

    def _setup_energy_function(
        self,
        activation: ActivationFunction | Callable[[Tensor], Tensor],
        energy_fn: Callable[[Tensor], Tensor] | None,
    ) -> None:
        """Set up the energy function based on activation type."""
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
                stacklevel=3,
            )

        if isinstance(activation, ActivationFunction):
            if activation == ActivationFunction.CUSTOM:
                if energy_fn is None:
                    raise ValueError(
                        "Must provide energy_fn when using CUSTOM activation"
                    )
                self.energy_fn = energy_fn
            else:
                self.energy_fn = get_energy_function(activation)
        else:
            # Assume it's a callable
            self.energy_fn = activation

    def reset_parameters(self) -> None:
        """Initialize learnable parameters using He scaling."""
        # Use He initialization scaled by input dimension
        init_std = he_scaled_init_std(self.in_dim, self.hidden_dim)
        nn.init.normal_(self.ξ, std=init_std)

        if self.use_bias and self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self, g: Tensor, memory_idx: int | None = None
    ) -> Tensor:  # scalar
        """Compute Hopfield Network energy.

        Parameters
        ----------
        g : Tensor
            Input tensor of shape [..., N, D] where N is the number
            of tokens and D is the feature dimension
        memory_idx : int, optional
            Index of memory set to use when batched_memories=True.
            If None, uses all memory sets and sums energies.

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

        if self.batched_memories:
            if memory_idx is not None:
                # Use specific memory set
                ξ_selected = ξ_cast[memory_idx]  # shape: [K, D]
                h = F.linear(g, ξ_selected)  # shape: [..., N, K]

                # Add bias if present
                if self.use_bias and self.bias is not None:
                    bias_selected = self.bias[memory_idx].to(g.dtype)
                    h = h + bias_selected
            else:
                # Use all memory sets and sum energies
                total_energy = torch.zeros((), device=g.device, dtype=g.dtype)

                for i in range(self.num_memory_sets):
                    ξ_i = ξ_cast[i]  # shape: [K, D]
                    h_i = F.linear(g, ξ_i)  # shape: [..., N, K]

                    if self.use_bias and self.bias is not None:
                        bias_i = self.bias[i].to(g.dtype)
                        h_i = h_i + bias_i

                    energy_i = self.energy_fn(h_i)

                    # Debug checks
                    if self.debug_checks:
                        self._debug_check_energy(energy_i, f"memory_set_{i}")

                    total_energy = total_energy + energy_i

                return total_energy
        else:
            # Standard single memory set
            h = F.linear(g, ξ_cast)  # shape: [..., N, K]

            # Add bias if present
            if self.use_bias and self.bias is not None:
                h = h + self.bias.to(g.dtype)

        # Energy: E^HN = -∑_{B=1}^{N} ∑_{μ=1}^{K} G(h_{μB})
        e_hn = self.energy_fn(h)  # scalar

        # Ensure the energy function returns a scalar
        assert e_hn.ndim == 0, "Energy function must return a scalar tensor"

        # Debug checks
        if self.debug_checks:
            self._debug_check_energy(e_hn)

        return e_hn

    def _debug_check_energy(
        self, energy: Tensor, context: str = "main"
    ) -> None:
        """Debug check for NaN/inf values in energy computation."""
        if torch.isnan(energy).any():
            raise RuntimeError(
                f"NaN detected in Hopfield energy computation ({context})"
            )
        if torch.isinf(energy).any():
            raise RuntimeError(
                f"Inf detected in Hopfield energy computation ({context})"
            )

    def get_memory_patterns(self, memory_idx: int | None = None) -> Tensor:
        """Get memory patterns for analysis or visualization.

        Parameters
        ----------
        memory_idx : int, optional
            Index of memory set to return when batched_memories=True.
            If None, returns all memory patterns.

        Returns
        -------
        Tensor
            Memory patterns of shape [K, D] or [M, K, D]
        """
        if self.batched_memories and memory_idx is not None:
            return self.ξ[memory_idx].detach().clone()
        else:
            return self.ξ.detach().clone()

    def add_custom_energy_function(
        self, name: str, energy_fn: Callable[[Tensor], Tensor]
    ) -> None:
        """Add a custom energy function to the registry.

        Parameters
        ----------
        name : str
            Name for the energy function
        energy_fn : callable
            Energy function takes hidden activations, returns scalar energy
        """
        _ENERGY_REGISTRY[name] = energy_fn

    @classmethod
    def list_available_energy_functions(cls) -> list[str]:
        """List all available registered energy functions.

        Returns
        -------
        list[str]
            Names of available energy functions
        """
        builtin_functions = [
            e.value
            for e in ActivationFunction
            if e != ActivationFunction.CUSTOM
        ]
        custom_functions = list(_ENERGY_REGISTRY.keys())
        return builtin_functions + custom_functions


# Example custom energy functions
def quadratic_energy(h: Tensor) -> Tensor:
    """Quadratic energy function: G(z) = z²/2."""
    return -0.5 * (h**2).sum()


def log_cosh_energy(h: Tensor) -> Tensor:
    """Log-cosh energy function: G(z) = log(cosh(z))."""
    return -torch.sum(torch.log(torch.cosh(h.clamp(-10, 10))))


# Register the example functions
_ENERGY_REGISTRY["quadratic"] = quadratic_energy
_ENERGY_REGISTRY["log_cosh"] = log_cosh_energy


# Utility function for consistent initialization across modules
def get_energy_transformer_init_std(
    fan_in: int, fan_out: int | None = None
) -> float:
    """Get standard initialization std for Energy Transformer modules.

    Parameters
    ----------
    fan_in : int
        Number of input features
    fan_out : int, optional
        Number of output features

    Returns
    -------
    float
        Standard deviation for weight initialization
    """
    return he_scaled_init_std(fan_in, fan_out)
