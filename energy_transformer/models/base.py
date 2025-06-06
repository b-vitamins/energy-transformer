"""Base Energy Transformer."""

from collections.abc import Callable
from contextlib import AbstractContextManager

import torch
from torch import Tensor, nn

from energy_transformer.utils.observers import StepInfo
from energy_transformer.utils.optimizers import SGD, EnergyOptimizer

__all__ = ["EnergyTransformer"]

# Registry for model classes to enable lookups from realiser
REALISER_REGISTRY: dict[str, type[nn.Module]] = {}


def force_enable_grad() -> AbstractContextManager[None]:
    """Use torch.enable_grad() to temporarily enable gradient computation."""
    return torch.enable_grad()  # type: ignore[no-any-return, no-untyped-call]


class EnergyTransformer(nn.Module):  # type: ignore[misc]
    r"""Base Energy Transformer with gradient descent optimization.

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

    Notes
    -----
    Mathematical Foundation:
    The Energy Transformer implements a continuous-time dynamical system that minimizes
    a composite energy function through gradient descent:

    .. math::
        \tau \frac{dx_{iA}}{dt} = -\frac{\\partial E}{\\partial g_{iA}}, \\quad \text{where} \\quad E = E^{ATT} + E^{HN}

    Key aspects of this formulation:

    1. **Gradient Computation**: The gradient is computed with respect to g (normalized
       tokens) but applied to update x (raw tokens). This is mathematically valid because
       LayerNorm is defined as g = \\partial L/\\partial x for a Lagrangian L.

    2. **Energy Decrease Proof**: The energy decreases over time:

       .. math::
           \frac{dE}{dt} = -\frac{1}{\tau} \\sum_{i,j,A} \frac{\\partial E}{\\partial g_{iA}} M_{ij}^A \frac{\\partial E}{\\partial g_{jA}} \\leq 0

       where :math:`M_{ij}^A = \frac{\\partial g_{iA}}{\\partial x_{jA}} = \frac{\\partial^2 L}{\\partial x_i \\partial x_j}`
       is the Hessian of the LayerNorm Lagrangian, which is positive semi-definite.

    3. **Discrete Time Implementation**: For numerical computation, the continuous dynamics
       are discretized using Euler's method with step size \alpha:

       .. math::
           x^{(t+1)} = x^{(t)} - \alpha \\cdot \frac{\\partial E}{\\partial g^{(t)}}

    4. **Gradient Context**: The gradient computation requires autograd even during
       inference, which is why we use ``torch.enable_grad()``.

    5. **No Parameter Updates**: Gradients computed in :meth:`_descent` are used
       only to update the tokens. Model parameters (\(W^Q\), \(W^K\), \(\xi\),
       \(\gamma\), \(\delta\)) remain fixed during this inner optimization.

    Implementation Details:
    The use of ``torch.enable_grad()`` ensures gradients work even in inference
    mode. Detach operations prevent gradient accumulation across descent steps,
    so each step starts from the current token state without backpropagating
    through previous steps.

    Convergence Properties:
    The energy is bounded below (E^ATT is bounded below by entropy, E^HN by the
    activation function properties), and decreases monotonically, guaranteeing
    convergence to a fixed point.

    Examples
    --------
    >>> # Monitor convergence during training
    >>> model = EnergyTransformer(...)
    >>> hook_handle = model.register_step_hook(
    ...     lambda m, info: print(f"Step {info.iteration}: E={info.total_energy:.4f}")
    ... )
    >>> output = model(x)
    >>> hook_handle.remove()

    >>> # Early stopping based on convergence
    >>> from energy_transformer.utils.observers import make_convergence_hook
    >>> model.register_step_hook(
    ...     make_convergence_hook(lambda i: print(f"Converged at step {i}!"))
    ... )
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
        # Note: Following equation (6) from the paper, we compute gradients with respect
        # to g = LayerNorm(x) but update x directly. This is valid because LayerNorm
        # is defined as the gradient of a Lagrangian: g = \u2202L/\u2202x. The energy decrease
        # is guaranteed by the positive semi-definiteness of the Hessian M = \u2202\u00b2L/\u2202x\u00b2.
        # Reset optimizer state
        self.optimizer.reset()

        # Call pre-descent hooks
        for pre_hook in self._pre_descent_hooks:
            pre_hook(self, x)

        for t in range(steps):
            # Compute components separately for observability
            # IMPORTANT: This gradient computation is ONLY for token updates
            # during energy minimization. It does NOT accumulate gradients in
            # model parameters. Parameter gradients are computed during the
            # standard backward() pass when training the model.
            with torch.enable_grad():  # type: ignore[no-untyped-call]
                # Normalize to obtain g
                g = self.layer_norm(x.detach())  # shape: [B, N, D]
                g_grad = g.detach().requires_grad_(True)

                # Compute energy components
                e_att = self.attention(g_grad)
                e_hop = self.hopfield(g_grad)
                total_energy = e_att + e_hop

                # Compute gradient w.r.t. g
                (grad,) = torch.autograd.grad(total_energy, g_grad)

            # Get update from optimizer
            update, step_size = self.optimizer.step(
                grad, x, total_energy, t
            )  # shape: [B, N, D]

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
            x = x - update  # shape: [B, N, D]

        # Call post-descent hooks
        for post_hook in self._post_descent_hooks:
            post_hook(self, x)

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
        For monitoring optimization, use hooks:

        >>> from energy_transformer.utils.observers import EnergyTracker
        >>> tracker = EnergyTracker()
        >>> model.register_step_hook(lambda m, info: tracker.update(info))
        >>> output = model(x)
        >>> stats = tracker.get_batch_statistics()
        """
        if x.dim() < 2:  # noqa: PLR2004
            raise ValueError(
                f"Input must have at least 2 dimensions [*, D], got {x.dim()}"
            )
        expected_dim = getattr(self.layer_norm, "normalized_shape", None)
        if expected_dim is not None and x.size(-1) != expected_dim[-1]:
            raise ValueError(
                f"Input feature dimension ({x.size(-1)}) doesn't match "
                f"layer norm dimension ({expected_dim[-1]})"
            )

        # Clone to preserve input
        x = x.clone()

        # Clear token capture flag
        self._capture_tokens = False

        # Perform descent
        return self._descent(x, self.steps)


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
