"""Base Energy Transformer."""

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import (
    Literal,
    NamedTuple,
    cast,
)

import torch
from torch import Tensor, nn

from energy_transformer.utils.observers import StepInfo
from energy_transformer.utils.optimizers import SGD, EnergyOptimizer

__all__ = ["DescentMode", "ETOutput", "EnergyTransformer", "Track"]

# Registry for model classes to enable lookups from realiser
REALISER_REGISTRY: dict[str, type[nn.Module]] = {}

Track = Literal["none", "energy", "trajectory", "both"]
DescentMode = Literal["sgd", "bb"]
GCtx = AbstractContextManager[None, bool | None]


def force_enable_grad() -> GCtx:
    """Use torch.enable_grad() to temporarily enable gradient computation.

    This context manager allows internal gradient computation for optimization
    even when the outer scope has disabled gradients via torch.no_grad().
    Does not work with torch.inference_mode(), which fundamentally prevents
    gradient tracking.
    """
    return cast(GCtx, torch.enable_grad())  # type: ignore[no-untyped-call]


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


class EnergyTransformer(nn.Module):  # type: ignore[misc]
    """Base Energy Transformer with gradient descent optimization.

    Defines a composite energy function that combines attention-based and
    memory-based energy components. The model optimizes token configurations
    through gradient descent on the energy landscape.

    Parameters
    ----------
    layer_norm : nn.Module
        Layer normalization component that transforms input tokens
    attention : nn.Module
        Energy-based attention component
    hopfield : nn.Module
        Hopfield network component for memory-based associations
    steps : int, optional
        Number of optimization steps, by default 12
    optimizer : EnergyOptimizer, optional
        Energy landscape optimizer. If None, uses SGD(alpha=0.1)
    """

    def __init__(
        self,
        layer_norm: nn.Module,
        attention: nn.Module,
        hopfield: nn.Module,
        steps: int = 12,
        optimizer: EnergyOptimizer | None = None,
    ) -> None:
        """Initialize the Energy Transformer with its energy components."""
        super().__init__()
        self.layer_norm = layer_norm
        self.attention = attention
        self.hopfield = hopfield
        self.steps = steps
        self.optimizer = optimizer or SGD(alpha=0.1)
        self.alpha = getattr(self.optimizer, "alpha", 0.1)

        # Hook management
        self._step_hooks: list[Callable] = []
        self._pre_descent_hooks: list[Callable] = []
        self._post_descent_hooks: list[Callable] = []
        self._capture_tokens = False

    def energy(
        self,
        x: Tensor,
    ) -> Tensor:
        """Compute the composite energy of the input token configuration.

        Note: This computes E^ATT(g) + E^HN(g) where g = LayerNorm(x).
        The LayerNorm energy E^LN(x) is not included here for compatibility
        with the standard forward pass. Use layer_norm_energy(x) + energy(x)
        for the complete energy including LayerNorm contribution.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [..., N, D]

        Returns
        -------
        Tensor
            Scalar energy value E^ATT(g) + E^HN(g)
        """
        # g = EnergyLayerNorm(x): x ∈ ℝᴺˣᴰ → g ∈ ℝᴺˣᴷ
        g = self.layer_norm(x)

        # E^ATT = attention(g, mask)
        e_att = self.attention(g)

        # E^HN = hopfield(g)
        e_hn = self.hopfield(g)

        # E^TOTAL = E^ATT + E^HN
        # Explicitly ensure return type is Tensor
        total_energy: Tensor = e_att + e_hn
        return total_energy

    def layer_norm_energy(self, x: Tensor) -> Tensor:
        """Compute the energy contribution from LayerNorm.

        Note: This is included for compatibility with the reference implementation,
        though it's not used in the standard forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [..., N, D]

        Returns
        -------
        Tensor
            Scalar LayerNorm energy value
        """
        if hasattr(self.layer_norm, "compute_energy"):
            return self.layer_norm.compute_energy(x)
        # For standard LayerNorm without energy, return 0
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    def _compute_gradient(
        self,
        x: Tensor,
        detach_mode: bool,
        create_graph: bool,
        track_trajectory: bool,
        trajectory: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Compute energy and gradient at current position.

        This implementation computes gradients with respect to g (normalized
        representation) rather than x, following the Energy Transformer paper.

        Returns
        -------
        tuple[Tensor, Tensor]
            Energy value and gradient with respect to g
        """
        if not detach_mode:
            with force_enable_grad():
                # First normalize x to get g
                g = self.layer_norm(x)

                # Detach g and make it require gradients
                g_for_grad = g.detach().requires_grad_(True)

                # Compute energy components on g
                e_att = self.attention(g_for_grad)
                e_hn = self.hopfield(g_for_grad)
                energy = e_att + e_hn

                if track_trajectory:
                    trajectory.append(energy.detach().clone())

                # Compute gradient with respect to g, not x
                (grad,) = torch.autograd.grad(
                    energy,
                    g_for_grad,
                    create_graph=create_graph,
                )
        else:
            # In detach mode, compute energy without gradients
            with torch.no_grad():
                g = self.layer_norm(x)
                e_att = self.attention(g)
                e_hn = self.hopfield(g)
                energy = e_att + e_hn

                if track_trajectory:
                    trajectory.append(energy.detach().clone())

            # Compute gradient for update
            with force_enable_grad():
                g = self.layer_norm(x)
                g_for_grad = g.detach().requires_grad_(True)

                e_att = self.attention(g_for_grad)
                e_hn = self.hopfield(g_for_grad)
                energy_for_grad = e_att + e_hn

                (grad,) = torch.autograd.grad(
                    energy_for_grad,
                    g_for_grad,
                    create_graph=False,
                )

        return energy, grad

    def _compute_bb_step_size(
        self,
        x: Tensor,
        grad: Tensor,
        prev_x: Tensor | None,
        prev_grad: Tensor | None,
    ) -> Tensor | float:
        """Compute Barzilai-Borwein step size.

        Note: The step size is computed based on changes in g-space gradients,
        but applied to updates in x-space, following the original algorithm.
        """
        if prev_x is not None and prev_grad is not None:
            s = (x - prev_x).flatten()
            y = (grad - prev_grad).flatten()
            denom = torch.dot(s, y)
            return torch.dot(s, s) / denom.clamp_min(1e-8)
        return self.alpha

    def _armijo_line_search(
        self,
        x: Tensor,
        grad: Tensor,
        energy: Tensor,
        lr: Tensor | float,
        detach_mode: bool,
        armijo_gamma: float,
        armijo_max_iter: int,
    ) -> Tensor:
        """Perform Armijo backtracking line search.

        Returns
        -------
        Tensor
            Step to take
        """
        for _ in range(armijo_max_iter):
            new_x = x - lr * grad

            # Evaluate new energy
            if not detach_mode:
                with force_enable_grad():
                    new_x_grad = new_x.clone().requires_grad_(True)
                    new_energy = self.energy(new_x_grad)
            else:
                with torch.no_grad():
                    new_energy = self.energy(new_x.detach())

            if new_energy < energy:  # sufficient decrease
                return lr * grad

            lr *= armijo_gamma

        # If all backtracking fails, use minimal step
        return lr * grad

    def _descent(self, x: Tensor, steps: int) -> Tensor:
        """Perform gradient descent on the energy landscape.

        Parameters
        ----------
        x : Tensor
            Initial token configuration
        steps : int
            Number of gradient descent steps

        Returns
        -------
        Tensor
            Optimized tokens
        """
        # Reset optimizer state
        self.optimizer.reset()

        # Call pre-descent hooks
        for hook in self._pre_descent_hooks:
            hook(self, x)

        for t in range(steps):
            # Compute components separately for observability
            with torch.enable_grad():
                # Normalize to get g
                g = self.layer_norm(x.detach())
                g_grad = g.detach().requires_grad_(True)

                # Compute energy components separately
                e_att = self.attention(g_grad)
                e_hop = self.hopfield(g_grad)
                total_energy = e_att + e_hop

                # Compute gradient w.r.t. g
                (grad,) = torch.autograd.grad(total_energy, g_grad)

            # Get update from optimizer
            update, step_size = self.optimizer.step(grad, x, total_energy, t)

            # Create step info for hooks
            step_info = StepInfo(
                iteration=t,
                total_energy=total_energy.detach(),
                attention_energy=e_att.detach(),
                hopfield_energy=e_hop.detach(),
                grad_norm=grad.norm(p=2, dim=-1).detach(),
                step_size=step_size.detach() if step_size is not None else None,
                tokens=x.detach() if self._capture_tokens else None,
            )

            # Call step hooks
            for hook in self._step_hooks:
                hook(self, step_info)

            # Apply update
            x = x - update

        # Call post-descent hooks
        for hook in self._post_descent_hooks:
            hook(self, x)

        return x

    def register_step_hook(
        self, hook: Callable[[nn.Module, StepInfo], None]
    ) -> "RemovableHandle":
        """Register a hook called after each optimization step.

        Parameters
        ----------
        hook : callable
            Function with signature: hook(module, step_info)

        Returns
        -------
        RemovableHandle
            Handle that can be used to remove the hook

        Example
        -------
        >>> from energy_transformer.utils.observers import make_logger_hook
        >>> handle = model.register_step_hook(make_logger_hook())
        >>> output = model(x)
        >>> handle.remove()
        """
        self._step_hooks.append(hook)
        return RemovableHandle(self._step_hooks, hook)

    def register_pre_descent_hook(
        self, hook: Callable[[nn.Module, Tensor], None]
    ) -> "RemovableHandle":
        """Register a hook called before descent begins.

        Parameters
        ----------
        hook : callable
            Function with signature: hook(module, initial_tokens)
        """
        self._pre_descent_hooks.append(hook)
        return RemovableHandle(self._pre_descent_hooks, hook)

    def register_post_descent_hook(
        self, hook: Callable[[nn.Module, Tensor], None]
    ) -> "RemovableHandle":
        """Register a hook called after descent completes.

        Parameters
        ----------
        hook : callable
            Function with signature: hook(module, final_tokens)
        """
        self._post_descent_hooks.append(hook)
        return RemovableHandle(self._post_descent_hooks, hook)


class RemovableHandle:
    """Handle for removing hooks."""

    def __init__(self, hooks_list: list[Callable], hook: Callable):
        self.hooks_list = hooks_list
        self.hook = hook

    def remove(self) -> None:
        """Remove the associated hook."""
        from contextlib import suppress

        with suppress(ValueError):
            self.hooks_list.remove(self.hook)

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

    def forward(self, x: Tensor) -> Tensor:
        """Optimize token configuration via descent on energy landscape.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [..., N, D]

        Returns
        -------
        Tensor
            Optimized tokens

        Notes
        -----
        To access trajectory information, use hooks:

        >>> from energy_transformer.utils.observers import EnergyTracker
        >>> tracker = EnergyTracker()
        >>> handle = model.register_step_hook(lambda m, info: tracker.update(info))
        >>> output = model(x)
        >>> trajectory = tracker.get_trajectory()
        """
        # Clone to preserve input
        x = x.clone()

        # Clear token capture flag
        self._capture_tokens = False

        # Perform descent
        return self._descent(x, self.steps)


# Register the EnergyTransformer class in the registry
REALISER_REGISTRY["EnergyTransformer"] = EnergyTransformer
