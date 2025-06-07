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


class EnergyTransformer(nn.Module):
    """Energy Transformer with direct gradient descent."""

    def __init__(
        self,
        layer_norm: nn.Module,
        attention: MultiheadEnergyAttention,
        hopfield: HopfieldNetwork | SimplicialHopfieldNetwork,
        steps: int = 12,
    ) -> None:
        super().__init__()
        self.layer_norm = layer_norm
        self.attention = attention
        self.hopfield = hopfield
        self.steps = steps

        self.alpha = DEFAULT_ALPHA
        self.skip_scale = nn.Parameter(torch.ones(1) * DEFAULT_SKIP_SCALE_INIT)

    def forward(
        self, x: Tensor, return_energies: bool = False
    ) -> Tensor | tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Run iterative energy minimization."""
        residual = x.clone()

        for _ in range(self.steps):
            g = self.layer_norm(x)
            x = x - self.alpha * self.attention.compute_grad(g)

            g = self.layer_norm(x)
            x = x - self.alpha * self.hopfield.compute_grad(g)

        x = x + self.skip_scale * residual

        if not return_energies:
            return x

        with torch.no_grad():
            e_att = self.attention.compute_energy(self.layer_norm(x))
            e_hop = self.hopfield.compute_energy(self.layer_norm(x))
        return x, [(e_att, e_hop)]


REALISER_REGISTRY = {"EnergyTransformer": EnergyTransformer}
