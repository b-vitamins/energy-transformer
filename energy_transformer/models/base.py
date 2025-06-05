"""Base Energy Transformer."""

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Literal, NamedTuple

import torch
from torch import Tensor, nn

from energy_transformer.utils.observers import StepInfo
from energy_transformer.utils.optimizers import SGD, EnergyOptimizer

Track = Literal["none", "energy", "trajectory", "both"]
DescentMode = Literal["sgd", "bb"]


class ETOutput(NamedTuple):
    """Output container for EnergyTransformer forward pass."""

    tokens: Tensor
    final_energy: Tensor | None = None
    trajectory: Tensor | None = None


__all__ = ["DescentMode", "ETOutput", "EnergyTransformer", "Track"]

# Registry for model classes to enable lookups from realiser
REALISER_REGISTRY: dict[str, type[nn.Module]] = {}


def force_enable_grad() -> AbstractContextManager[None]:
    """Use torch.enable_grad() to temporarily enable gradient computation."""
    return torch.enable_grad()  # type: ignore[no-any-return, no-untyped-call]


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
        alpha: float | None = None,
    ) -> None:
        """Initialize the Energy Transformer with its energy components."""
        super().__init__()
        self.layer_norm = layer_norm
        self.attention = attention
        self.hopfield = hopfield
        self.steps = steps
        if optimizer is not None:
            self.optimizer = optimizer
            self.alpha = getattr(
                optimizer, "alpha", alpha if alpha is not None else 0.1
            )
        else:
            self.alpha = 0.1 if alpha is None else alpha
            self.optimizer = SGD(alpha=self.alpha)

        # Hook management
        self._step_hooks: list[Callable[[nn.Module, StepInfo], None]] = []
        self._pre_descent_hooks: list[Callable[[nn.Module, Tensor], None]] = []
        self._post_descent_hooks: list[Callable[[nn.Module, Tensor], None]] = []
        self._capture_tokens = False

    def _compute_energy(
        self,
        x: Tensor,
    ) -> Tensor:
        """Compute the composite energy.

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
        # Mathematical Note: This implementation follows the original Energy Transformer
        # algorithm which computes gradients with respect to g = LayerNorm(x) but updates
        # x directly. While this is mathematically inconsistent (mixing gradient spaces),
        # it maintains compatibility with the published algorithm and results.
        # Reset optimizer state
        self.optimizer.reset()

        # Call pre-descent hooks
        for pre_hook in self._pre_descent_hooks:
            pre_hook(self, x)

        for t in range(steps):
            # Compute components separately for observability
            with torch.enable_grad():  # type: ignore[no-untyped-call]
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
            for step_hook in self._step_hooks:
                step_hook(self, step_info)

            # Apply update
            x = x - update

        # Call post-descent hooks
        for post_hook in self._post_descent_hooks:
            post_hook(self, x)

        return x

    def _should_clone(
        self,
        training: bool,
        detach: bool,
        force_clone: bool | None,
    ) -> bool:
        """Determine whether to clone input tokens."""
        if force_clone is not None:
            return force_clone
        return training or detach

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

    def forward(
        self,
        x: Tensor,
        detach: bool = False,
        track: Track = "none",
        _mode: DescentMode = "bb",
        force_clone: bool | None = None,
        _armijo_gamma: float = 0.5,
        _armijo_max_iter: int = 4,
    ) -> Tensor | ETOutput:
        """Optimize token configuration via descent on energy landscape."""
        if torch.is_inference_mode_enabled() and not detach:
            raise RuntimeError(
                "EnergyTransformer requires gradient computation, "
                "which is not possible within torch.inference_mode()."
            )

        # Reset token capture flag
        self._capture_tokens = False

        if self._should_clone(self.training, detach, force_clone):
            x = x.clone()

        if detach:
            x = x.detach()

        x.requires_grad_(True)

        track_traj = track in ("trajectory", "both")
        return_energy = track in ("energy", "both")

        trajectory: list[Tensor] = []
        handle = None
        if track_traj:
            handle = self.register_step_hook(
                lambda _m, info: trajectory.append(info.total_energy)
            )

        out = self._descent(x, self.steps)

        if handle is not None:
            handle.remove()

        final_energy = None
        if return_energy:
            with torch.no_grad():
                final_energy = self._compute_energy(out)

        if detach:
            out = out.detach()
            if final_energy is not None:
                final_energy = final_energy.detach()

        if track == "none":
            return out

        traj_tensor = (
            torch.stack(trajectory) if track_traj and trajectory else None
        )
        return ETOutput(
            tokens=out, final_energy=final_energy, trajectory=traj_tensor
        )


class RemovableHandle:
    """Handle for removing hooks."""

    def __init__(
        self, hooks_list: list[Callable[..., None]], hook: Callable[..., None]
    ):
        self.hooks_list = hooks_list
        self.hook = hook

    def remove(self) -> None:
        """Remove the associated hook."""
        from contextlib import suppress

        with suppress(ValueError):
            self.hooks_list.remove(self.hook)


# Register the EnergyTransformer class in the registry
REALISER_REGISTRY["EnergyTransformer"] = EnergyTransformer
