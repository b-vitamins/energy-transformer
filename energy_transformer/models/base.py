"""Base Energy Transformer."""

import torch
from torch import Tensor, nn


class EnergyTransformer(nn.Module):
    """Energy Transformer with direct gradient descent."""

    def __init__(
        self,
        layer_norm: nn.Module,
        attention: nn.Module,
        hopfield: nn.Module,
        steps: int = 12,
        _optimizer: object | None = None,  # Ignored, kept for compatibility
    ) -> None:
        super().__init__()
        self.layer_norm = layer_norm
        self.attention = attention
        self.hopfield = hopfield
        self.steps = steps

        self.alpha = 0.125
        self.skip_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(
        self, x: Tensor, return_energies: bool = False
    ) -> Tensor | tuple[Tensor, list]:
        """Forward pass with direct gradient descent."""
        residual = x.clone()

        energies: list = []

        for i in range(self.steps):
            g = self.layer_norm(x)
            grad_attn = self.attention.compute_grad(g)
            x = x - self.alpha * grad_attn

            g = self.layer_norm(x)
            grad_hop = self.hopfield.compute_grad(g)
            x = x - self.alpha * grad_hop

            if return_energies and i == self.steps - 1:
                with torch.no_grad():
                    e_att = self.attention.compute_energy(self.layer_norm(x))
                    e_hop = self.hopfield.compute_energy(self.layer_norm(x))
                    energies.append((e_att, e_hop))

        x = x + self.skip_scale * residual

        if return_energies:
            return x, energies
        return x


REALISER_REGISTRY = {"EnergyTransformer": EnergyTransformer}
