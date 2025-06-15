from __future__ import annotations

from torch import Tensor, nn

__all__ = ["EnergyModule"]


class EnergyModule(nn.Module):
    """Base class for energy-based modules.

    Subclasses must implement :meth:`compute_energy` and
    :meth:`compute_grad` to define the energy function and its gradient.
    """

    def compute_energy(self, x: Tensor) -> Tensor:
        """Return scalar energy for input ``x``.

        Implementations should average the energy over the batch and
        sequence dimensions to keep scales comparable across modules.
        """
        raise NotImplementedError

    def compute_grad(self, x: Tensor) -> Tensor:
        """Return gradient of the energy with respect to ``x``."""
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Alias for :meth:`compute_energy` to integrate with ``nn.Module``."""
        return self.compute_energy(x)

    @property
    def num_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
