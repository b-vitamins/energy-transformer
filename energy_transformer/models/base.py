"""Base Energy Transformer model definition."""

from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from energy_transformer.layers.base import (
    BaseEnergyAttention,
    BaseHopfieldNetwork,
    BaseLayerNorm,
)

# Registry for model classes to enable lookups from realiser
REALISER_REGISTRY: dict[str, type[nn.Module]] = {}


class ETOutput(NamedTuple):
    """Output container for EnergyTransformer forward pass.

    Attributes
    ----------
    tokens : Tensor
        Optimized token configuration of shape [..., N, D]
    final_energy : Tensor, optional
        Final scalar energy value after optimization
    trajectory : Tensor, optional
        Energy trajectory during optimization of shape [steps]
    """

    tokens: Tensor
    final_energy: Tensor | None = None
    trajectory: Tensor | None = None


class EnergyTransformer(nn.Module):  # type: ignore
    """Base Energy Transformer with gradient descent optimization.

    Defines a composite energy function that combines attention-based and
    memory-based energy components. The model optimizes token configurations
    through gradient descent on the energy landscape.

    Parameters
    ----------
    layer_norm : BaseLayerNorm
        Layer normalization component that transforms input tokens x ∈ ℝᴺˣᴰ
        into normalized representation g ∈ ℝᴺˣᴰ
    attention : BaseEnergyAttention
        Energy-based attention component that computes E^ATT from normalized tokens
    hopfield : BaseHopfieldNetwork
        Hopfield network component that computes E^HN from normalized tokens
    steps : int, optional
        Number of gradient descent steps T for energy optimization, by default 12
    α : float, optional
        Step size for gradient descent optimization, by default 0.125

    Notes
    -----
    The Energy Transformer defines a composite energy function:

    E^TOTAL(x) = E^ATT(g) + E^HN(g)

    where:
    - x ∈ ℝᴺˣᴰ are the input token representations
    - g = LayerNorm(x) ∈ ℝᴺˣᴰ are normalized tokens
    - E^ATT(g) is the attention energy from multi-head attention
    - E^HN(g) is the memory energy from the Hopfield network

    The model performs iterative optimization via gradient descent:

    x^(t+1) = x^(t) - α · ∇ₓ E^TOTAL(x^(t))

    for t = 0, 1, ..., T-1 steps, where α is the learning rate and
    ∇ₓ denotes the gradient with respect to token positions x.

    This optimization process allows tokens to settle into configurations
    that minimize the combined energy, effectively performing associative
    memory retrieval and attention-based reasoning simultaneously.
    """

    def __init__(
        self,
        layer_norm: BaseLayerNorm,
        attention: BaseEnergyAttention,
        hopfield: BaseHopfieldNetwork,
        steps: int = 12,
        α: float = 0.125,
    ) -> None:
        """Initialize the Energy Transformer with its energy components.

        Parameters
        ----------
        layer_norm : BaseLayerNorm
            Layer normalization component for token preprocessing
        attention : BaseEnergyAttention
            Energy-based attention component for relational reasoning
        hopfield : BaseHopfieldNetwork
            Hopfield network component for memory-based associations
        steps : int, optional
            Number of gradient descent steps T, by default 12
        α : float, optional
            Step size for gradient descent, by default 0.125
        """
        super().__init__()
        self.layer_norm = layer_norm
        self.attention = attention
        self.hopfield = hopfield
        self.steps = steps
        self.α = α

    def energy(
        self,
        x: Tensor,
        energy_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute the composite energy of the input token configuration.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [..., N, D] where N is the number
            of tokens and D is the feature dimension of each token
        energy_mask : Tensor, optional
            Energy mask for attention computation of shape [..., H, N, N]
            or broadcastable. Uses additive masking: 0 for allowed positions,
            -∞ for masked positions.

        Returns
        -------
        Tensor
            Scalar energy value E^TOTAL representing the combined energy.
            Lower energy corresponds to more favorable token configurations.

        Notes
        -----
        The energy computation follows these steps:
        1. g = LayerNorm(x) - normalize input tokens to ℝᴺˣᴷ
        2. E^ATT = attention(g, mask) - compute attention energy
        3. E^HN = hopfield(g) - compute Hopfield memory energy
        4. E^TOTAL = E^ATT + E^HN - return combined energy

        For mixed-precision training, components handle dtype casting
        internally when torch.cuda.amp.autocast() is active.
        """
        # g = LayerNorm(x): x ∈ ℝᴺˣᴰ → g ∈ ℝᴺˣᴷ
        g = self.layer_norm(x)  # shape: [..., N, K]

        # E^ATT = attention(g, mask)
        try:
            e_att = self.attention(g, attn_mask=energy_mask)  # scalar
        except TypeError:
            e_att = self.attention(g)  # scalar

        # E^HN = hopfield(g)
        e_hn = self.hopfield(g)  # scalar

        # E^TOTAL = E^ATT + E^HN
        return e_att + e_hn  # scalar

    def forward(
        self,
        x: Tensor,
        detach: bool = False,
        return_energy: bool = False,
        return_trajectory: bool = False,
        α: float | None = None,
        energy_mask: Tensor | None = None,
        force_clone: bool | None = None,
    ) -> Tensor | ETOutput:
        """Optimize token configuration via gradient descent on energy landscape.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [..., N, D] where N is the number
            of tokens and D is the feature dimension of each token
        detach : bool, optional
            If True, detaches gradients after optimization. Useful for
            frozen-backbone scenarios, inference-only modes, and preventing
            gradient flow through optimization trajectory, by default False
        return_energy : bool, optional
            If True, returns the final energy value E^TOTAL(x^(T)) alongside
            optimized tokens, by default False
        return_trajectory : bool, optional
            If True, returns energy trajectory [E^TOTAL(x^(0)), ..., E^TOTAL(x^(T-1))]
            during optimization for visualization and analysis, by default False
        α : float, optional
            Step size for gradient descent. If provided, overrides self.α
            for this forward pass. Useful for inference-time tuning
        energy_mask : Tensor, optional
            Energy mask for attention computation of shape [..., H, N, N].
            Supports domain-specific masking like padding, causal attention,
            or span-drop. Uses additive masking: 0 for allowed, -∞ for masked
        force_clone : bool, optional
            If provided, overrides default cloning heuristic. Use False for
            test-time energy descent with learnable prompts that should
            mutate in-place

        Returns
        -------
        Union[Tensor, ETOutput]
            If no additional outputs requested: optimized tokens x^(T) of shape [..., N, D]
            Otherwise: ETOutput namedtuple containing (tokens, final_energy, trajectory)

        Notes
        -----
        The optimization procedure implements iterative gradient descent:

        for t = 0, 1, ..., T-1:
            E^(t) = E^TOTAL(x^(t))                    # Forward: compute energy
            ∇E^(t) = ∇ₓ E^TOTAL(x^(t))                # Backward: compute gradient
            x^(t+1) = x^(t) - α · ∇E^(t)             # Update: gradient descent step

        The final optimized configuration x^(T) represents tokens that have
        settled into a local minimum of the energy landscape, balancing
        attention-based relational constraints and memory-based associations.

        Gradient computation uses autograd.grad with create_graph=True to
        maintain differentiability for higher-order derivatives and
        end-to-end training through the optimization process.
        """
        # Conditional cloning: preserve original input when needed
        should_clone = (
            force_clone
            if force_clone is not None
            else (self.training or detach)
        )
        if should_clone:
            x = x.clone()  # shape: [..., N, D]

        # Gradient isolation for frozen-backbone scenarios
        if detach:
            x = x.detach()  # shape: [..., N, D]

        # Ensure x requires gradients for optimization
        x.requires_grad_(True)

        # α - step size selection: override or use default
        step_size = α if α is not None else self.α

        # Initialize trajectory tracking for energy visualization
        energy_trajectory: list[Tensor] | None = (
            [] if return_trajectory else None
        )
        final_energy = None

        # Iterative gradient descent optimization: x^(t+1) = x^(t) - α · ∇ₓ E^TOTAL(x^(t))
        for _i in range(self.steps):
            # Forward pass: E^(t) = E^TOTAL(x^(t))
            energy = self.energy(x, energy_mask=energy_mask)  # scalar

            # Record energy for trajectory visualization
            if return_trajectory and energy_trajectory is not None:
                energy_trajectory.append(energy.detach().clone())  # scalar

            # Backward pass: ∇E^(t) = ∇ₓ E^TOTAL(x^(t))
            (grad,) = torch.autograd.grad(
                energy,  # scalar energy to differentiate
                x,  # [..., N, D] - differentiate w.r.t. tokens
                create_graph=True,  # Essential for higher-order derivatives
            )  # grad shape: [..., N, D]

            # Gradient descent step: x^(t+1) = x^(t) - α · ∇E^(t)
            if detach:
                # Detached mode: use no_grad for efficiency, break computation graph
                with torch.no_grad():
                    x = x - step_size * grad  # shape: [..., N, D]
                # Re-enable gradients for next iteration (except final step)
                if _i < self.steps - 1:
                    x.requires_grad_(True)
            else:
                # Training mode: preserve computational graph for end-to-end training
                x = x - step_size * grad  # shape: [..., N, D]

        # Compute final energy: E^TOTAL(x^(T))
        if return_energy or return_trajectory:
            with torch.no_grad():
                final_energy = self.energy(x, energy_mask=energy_mask)  # scalar

        # Gradient isolation for inference scenarios
        if detach:
            x = x.detach()  # shape: [..., N, D]
            if final_energy is not None:
                final_energy = final_energy.detach()  # scalar
            if energy_trajectory is not None:
                energy_trajectory = [
                    e.detach() for e in energy_trajectory
                ]  # [scalar, ...]

        # Return format selection based on requested outputs
        if return_energy or return_trajectory:
            # Stack trajectory: [E^(0), E^(1), ..., E^(T-1)] → tensor of shape [T]
            trajectory_tensor = (
                torch.stack(energy_trajectory) if energy_trajectory else None
            )  # shape: [steps] or None
            return ETOutput(
                tokens=x,  # shape: [..., N, D]
                final_energy=final_energy,  # scalar or None
                trajectory=trajectory_tensor,  # shape: [steps] or None
            )
        else:
            return x  # shape: [..., N, D]


# Register the EnergyTransformer class in the registry
REALISER_REGISTRY["EnergyTransformer"] = EnergyTransformer
