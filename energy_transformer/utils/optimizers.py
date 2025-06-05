"""Energy landscape optimizers for Energy Transformer.

This module provides optimizers for descending the energy landscape
during the forward pass of Energy Transformer models.
"""

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class EnergyOptimizer(Protocol):
    """Protocol for energy landscape optimizers.

    These optimizers are used within the Energy Transformer forward pass
    to minimize the energy function through iterative updates.
    """

    def step(
        self,
        grad: Tensor,
        _x: Tensor,
        _energy: Tensor,
        _t: int,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute the update step.

        Parameters
        ----------
        grad : Tensor
            Gradient of energy with respect to g (normalized representation)
        x : Tensor
            Current token representation
        energy : Tensor
            Current energy value
        t : int
            Current iteration number

        Returns
        -------
        Tuple[Tensor, Optional[Tensor]]
            Update to apply to x, and optional step size for logging
        """
        ...

    def reset(self) -> None:
        """Reset any internal state of the optimizer."""
        ...


class SGD:
    """Simple gradient descent optimizer for energy landscape.

    Parameters
    ----------
    alpha : float
        Fixed step size for gradient descent
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def step(
        self,
        grad: Tensor,
        _x: Tensor,
        _energy: Tensor,
        _t: int,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute SGD update step."""
        return self.alpha * grad, torch.tensor(self.alpha)

    def reset(self) -> None:
        """SGD has no internal state to reset."""


class Momentum:
    """Momentum-based optimizer for energy landscape.

    Parameters
    ----------
    alpha : float
        Step size
    momentum : float
        Momentum coefficient (typically 0.9)
    """

    def __init__(self, alpha: float = 0.1, momentum: float = 0.9):
        self.alpha = alpha
        self.momentum = momentum
        self.velocity = None

    def step(
        self,
        grad: Tensor,
        _x: Tensor,
        _energy: Tensor,
        _t: int,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute momentum update step."""
        if self.velocity is None:
            self.velocity = torch.zeros_like(grad)

        self.velocity = self.momentum * self.velocity + self.alpha * grad
        return self.velocity, torch.tensor(self.alpha)

    def reset(self) -> None:
        """Reset velocity buffer."""
        self.velocity = None


class AdaptiveGD:
    """Adaptive gradient descent with norm-based step size.

    Scales step size based on gradient norm to prevent instability.

    Parameters
    ----------
    alpha : float
        Base step size
    eps : float
        Small constant for numerical stability
    """

    def __init__(self, alpha: float = 0.1, eps: float = 1e-8):
        self.alpha = alpha
        self.eps = eps

    def step(
        self,
        grad: Tensor,
        _x: Tensor,
        _energy: Tensor,
        _t: int,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute adaptive update step."""
        grad_norm = grad.norm(p=2, dim=-1, keepdim=True)
        adaptive_alpha = self.alpha / (grad_norm + self.eps)
        return adaptive_alpha * grad, adaptive_alpha.mean()

    def reset(self) -> None:
        """No internal state to reset."""
