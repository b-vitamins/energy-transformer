from __future__ import annotations

from torch import Tensor, nn

__all__ = ["EnergyModule"]


class EnergyModule(nn.Module):
    """Base class for energy-based modules."""

    def compute_energy(self, x: Tensor) -> Tensor:
        """Compute scalar energy."""
        raise NotImplementedError

    def compute_grad(self, x: Tensor) -> Tensor:
        """Compute energy gradient."""
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Return energy."""
        return self.compute_energy(x)
