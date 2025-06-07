"""Base Energy Transformer."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from energy_transformer.layers.attention import MultiheadEnergyAttention
from energy_transformer.layers.constants import (
    DEFAULT_ALPHA,
    DEFAULT_SKIP_SCALE_INIT,
)
from energy_transformer.layers.hopfield import HopfieldNetwork
from energy_transformer.layers.simplicial import SimplicialHopfieldNetwork
from energy_transformer.layers.validation import validate_positive

__all__ = ["EnergyTransformer"]


class EnergyTransformer(nn.Module):
    """Energy Transformer core module.

    Parameters
    ----------
    layer_norm : nn.Module
        Normalization layer used before each energy component.
    attention : MultiheadEnergyAttention
        Energy-based attention module.
    hopfield : HopfieldNetwork | SimplicialHopfieldNetwork
        Associative memory module.
    steps : int, default=12
        Number of iterative refinement steps.
    """

    def __init__(
        self,
        layer_norm: nn.Module,
        attention: MultiheadEnergyAttention,
        hopfield: HopfieldNetwork | SimplicialHopfieldNetwork,
        steps: int = 12,
    ) -> None:
        super().__init__()

        validate_positive(steps, self.__class__.__name__, "steps")
        self.layer_norm = layer_norm
        self.attention = attention
        self.hopfield = hopfield
        self.steps = steps

        self.alpha = DEFAULT_ALPHA
        self.skip_scale = nn.Parameter(torch.ones(1) * DEFAULT_SKIP_SCALE_INIT)

    def _step(self, x: Tensor) -> Tensor:
        """Perform a single energy minimization step."""
        g = self.layer_norm(x)
        x = x - self.alpha * self.attention.compute_grad(g)
        g = self.layer_norm(x)
        return x - self.alpha * self.hopfield.compute_grad(g)

    def _compute_energies(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return attention and Hopfield energies without gradients."""
        g = self.layer_norm(x)
        e_att = self.attention.compute_energy(g)
        e_hop = self.hopfield.compute_energy(g)
        return e_att, e_hop

    def forward(
        self, x: Tensor, return_energies: bool = False
    ) -> Tensor | tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Run iterative energy minimization."""
        residual = x.clone()

        for _ in range(self.steps):
            x = self._step(x)

        out = x + self.skip_scale * residual

        if not return_energies:
            return out

        with torch.no_grad():
            e_att, e_hop = self._compute_energies(out)

        return out, [(e_att, e_hop)]
