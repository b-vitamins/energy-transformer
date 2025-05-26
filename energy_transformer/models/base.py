"""Base Energy Transformer model definition with gradient override support."""

from typing import Literal, NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from energy_transformer.layers.base import (
    BaseEnergyAttention,
    BaseHopfieldNetwork,
    BaseLayerNorm,
)

__all__ = ["EnergyTransformer", "Track", "ETOutput", "DescentMode"]

# Registry for model classes to enable lookups from realiser
REALISER_REGISTRY: dict[str, type[nn.Module]] = {}

Track = Literal["none", "energy", "trajectory", "both"]
DescentMode = Literal["sgd", "bb"]


def force_enable_grad():
    """Use torch.enable_grad() to temporarily enable gradient computation.

    This context manager allows internal gradient computation for optimization
    even when the outer scope has disabled gradients via torch.no_grad().
    Does not work with torch.inference_mode(), which fundamentally prevents
    gradient tracking.
    """
    return torch.enable_grad()


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
        Layer normalization component that transforms input tokens
    attention : BaseEnergyAttention
        Energy-based attention component
    hopfield : BaseHopfieldNetwork
        Hopfield network component for memory-based associations
    steps : int, optional
        Number of gradient descent steps T, by default 12
    alpha : float, optional
        Step size for gradient descent optimization, by default 1.0
    """

    def __init__(
        self,
        layer_norm: BaseLayerNorm,
        attention: BaseEnergyAttention,
        hopfield: BaseHopfieldNetwork,
        steps: int = 12,
        alpha: float = 1.0,
    ) -> None:
        """Initialize the Energy Transformer with its energy components."""
        super().__init__()
        self.layer_norm = layer_norm
        self.attention = attention
        self.hopfield = hopfield
        self.steps = steps
        self.alpha = alpha

    def energy(
        self,
        x: Tensor,
    ) -> Tensor:
        """Compute the composite energy of the input token configuration.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [..., N, D]

        Returns
        -------
        Tensor
            Scalar energy value E^TOTAL
        """
        # g = LayerNorm(x): x ∈ ℝᴺˣᴰ → g ∈ ℝᴺˣᴷ
        g = self.layer_norm(x)

        # E^ATT = attention(g, mask)
        e_att = self.attention(g)

        # E^HN = hopfield(g)
        e_hn = self.hopfield(g)

        # E^TOTAL = E^ATT + E^HN
        return e_att + e_hn

    def _energy_descent(
        self,
        x: Tensor,
        steps: int,
        mode: DescentMode = "bb",
        detach_mode: bool = False,
        track_trajectory: bool = False,
        armijo_gamma: float = 0.5,
        armijo_max_iter: int = 4,
    ) -> tuple[Tensor, list[Tensor]]:
        """Perform gradient descent optimization on the energy landscape.

        Parameters
        ----------
        x : Tensor
            Initial token configuration. Must have requires_grad=True.
        steps : int
            Number of gradient descent steps
        mode : DescentMode, optional
            Descent method to use:
            - "sgd": Fixed step size (classic)
            - "bb": Barzilai-Borwein with Armijo backtracking (default)
        detach_mode : bool, optional
            If True, performs detached updates (no gradient flow)
        track_trajectory : bool, optional
            If True, records energy values at each step
        armijo_gamma : float, optional
            Armijo backtracking factor, by default 0.5
        armijo_max_iter : int, optional
            Maximum Armijo iterations, by default 4

        Returns
        -------
        tuple[Tensor, list[Tensor]]
            Optimized tokens and energy trajectory (empty if not tracking)

        Notes
        -----
        This method can work even within torch.no_grad() contexts by temporarily
        re-enabling gradients. However, it cannot work within torch.inference_mode()
        which fundamentally prevents gradient tracking.
        """
        trajectory = []
        create_graph = not detach_mode

        # Check for inference_mode early - we cannot override this
        if torch.is_inference_mode_enabled() and not detach_mode:
            raise RuntimeError(
                "EnergyTransformer optimization requires gradient computation, "
                "which is not possible within torch.inference_mode(). "
                "Either call the model outside inference_mode() or use detach=True."
            )

        # Initialize BB buffers
        prev_x, prev_grad = None, None

        for t in range(steps):
            # In detach mode with inference_mode, we cannot compute gradients
            if detach_mode and torch.is_inference_mode_enabled():
                # Skip optimization in inference mode with detach
                # Just track energy if requested
                if track_trajectory:
                    with torch.no_grad():
                        energy = self.energy(x)
                        trajectory.append(energy.detach().clone())
                continue

            # Use force_enable_grad to ensure gradients work even in no_grad context
            # This is only needed when we're actually computing gradients
            if not detach_mode:
                with force_enable_grad():
                    # Clone x inside enable_grad to ensure proper gradient tracking
                    # This is necessary when the original x was created in a no_grad context
                    x_grad = x.clone().requires_grad_(True)

                    # Compute energy at current position
                    energy = self.energy(x_grad)

                    # Track energy if requested
                    if track_trajectory:
                        trajectory.append(energy.detach().clone())

                    # Compute gradient ∇ₓ E^TOTAL(x)
                    (grad,) = torch.autograd.grad(
                        energy, x_grad, create_graph=create_graph
                    )
            else:
                # In detach mode, we don't need gradients
                with torch.no_grad():
                    energy = self.energy(x)
                    if track_trajectory:
                        trajectory.append(energy.detach().clone())

                # Still compute gradient for the update, but detached
                with force_enable_grad():
                    # Clone x inside enable_grad for proper gradient tracking
                    x_grad = x.clone().requires_grad_(True)
                    energy_for_grad = self.energy(x_grad)
                    (grad,) = torch.autograd.grad(
                        energy_for_grad, x_grad, create_graph=False
                    )

            # Skip update if we didn't compute gradients
            if detach_mode and torch.is_inference_mode_enabled():
                continue

            if mode == "bb":
                # Compute BB step size if we have history
                if prev_x is not None and prev_grad is not None:
                    s = (x - prev_x).flatten()
                    y = (grad - prev_grad).flatten()
                    denom = torch.dot(s, y)
                    lr = torch.dot(s, s) / denom.clamp_min(1e-8)
                else:
                    # First step uses default alpha
                    lr = self.alpha

                # Armijo backtracking line search for safety
                for _ in range(armijo_max_iter):
                    new_x = x - lr * grad

                    # Evaluate new energy with appropriate gradient context
                    if not detach_mode:
                        with force_enable_grad():
                            # Clone new_x inside enable_grad for proper gradient tracking
                            new_x_grad = new_x.clone().requires_grad_(True)
                            new_energy = self.energy(new_x_grad)
                    else:
                        with torch.no_grad():
                            new_energy = self.energy(new_x.detach())

                    if new_energy < energy:  # sufficient decrease
                        step = lr * grad
                        break
                    lr *= armijo_gamma
                else:
                    # If all backtracking fails, use minimal step
                    step = lr * grad

            else:  # "sgd"
                step = self.alpha * grad

            # Apply update
            if detach_mode:
                with torch.no_grad():
                    x = x - step
                # Re-enable gradients for next iteration (except last step)
                if t < steps - 1:
                    x.requires_grad_(True)
            else:
                # In non-detach mode, x already has gradient tracking from above
                x = x - step

            # Store for BB
            if mode == "bb":
                prev_x = x.detach() if detach_mode else x.clone().detach()
                prev_grad = grad.detach()

        return x, trajectory

    def _should_clone(
        self,
        training: bool,
        detach: bool,
        force_clone: bool | None,
    ) -> bool:
        """Determine whether to clone input tokens.

        Cloning is needed to:
        - Preserve original inputs during training (for gradient safety)
        - Isolate computations when detaching (frozen backbone scenarios)
        - Unless explicitly overridden by force_clone
        """
        if force_clone is not None:
            return force_clone
        return training or detach

    def forward(
        self,
        x: Tensor,
        detach: bool = False,
        track: Track = "none",
        mode: DescentMode = "bb",
        force_clone: bool | None = None,
        armijo_gamma: float = 0.5,
        armijo_max_iter: int = 4,
    ) -> Tensor | ETOutput:
        """Optimize token configuration via gradient descent on energy landscape.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [..., N, D]
        detach : bool, optional
            If True, detaches gradients after optimization
        track : Track, optional
            Controls what to track during optimization:
            - "none": Return only optimized tokens (default)
            - "energy": Return tokens and final energy
            - "trajectory": Return tokens and energy trajectory
            - "both": Return tokens, final energy, and trajectory
        mode : DescentMode, optional
            Descent method to use:
            - "sgd": Fixed step size
            - "bb": Barzilai-Borwein with Armijo backtracking (default)
        force_clone : bool, optional
            Overrides default cloning heuristic if provided
        armijo_gamma : float, optional
            Backtracking factor for BB mode, by default 0.5
        armijo_max_iter : int, optional
            Maximum backtracking iterations for BB mode, by default 4

        Returns
        -------
        Union[Tensor, ETOutput]
            Optimized tokens or ETOutput with additional tracked values

        Notes
        -----
        This method can operate within torch.no_grad() contexts by temporarily
        enabling gradients for internal optimization. However, it cannot work
        within torch.inference_mode() unless detach=True is used.
        """
        # Conditional cloning: preserve original input when needed
        if self._should_clone(self.training, detach, force_clone):
            x = x.clone()

        # Gradient isolation for frozen-backbone scenarios
        if detach:
            x = x.detach()

        # Ensure x requires gradients for optimization
        x.requires_grad_(True)

        # Determine tracking requirements
        track_trajectory = track in ("trajectory", "both")
        return_energy = track in ("energy", "both")

        # Perform energy descent optimization
        x, trajectory = self._energy_descent(
            x=x,
            steps=self.steps,
            mode=mode,
            detach_mode=detach,
            track_trajectory=track_trajectory,
            armijo_gamma=armijo_gamma,
            armijo_max_iter=armijo_max_iter,
        )

        # Compute final energy if needed
        final_energy = None
        if return_energy:
            with torch.no_grad():
                final_energy = self.energy(x)

        # Final detach if needed
        if detach:
            x = x.detach()
            if final_energy is not None:
                final_energy = final_energy.detach()

        # Return appropriate output format
        if track != "none":
            # Only create trajectory tensor if we tracked it
            trajectory_tensor = (
                torch.stack(trajectory, dim=0)
                if track_trajectory and trajectory
                else None
            )
            return ETOutput(
                tokens=x,
                final_energy=final_energy,
                trajectory=trajectory_tensor,
            )
        else:
            return x


# Register the EnergyTransformer class in the registry
REALISER_REGISTRY["EnergyTransformer"] = EnergyTransformer
