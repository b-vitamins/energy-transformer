"""Hopfield Network layer for Energy Transformer."""

import torch
import torch.nn as nn
import torch.nn.functional as f


class HopfieldNetwork(nn.Module):
    """Modern Hopfield Network layer.

    This replaces the MLP in traditional transformers. It uses associative
    memory with ReLU activation and has an associated energy function.

    Args:
        d_model: Model dimension
        n_memories: Number of memory patterns
    """

    def __init__(self, d_model: int, n_memories: int):
        super().__init__()
        self.d_model = d_model
        self.n_memories = n_memories

        # Memory patterns
        self.xi = nn.Parameter(torch.randn(d_model, n_memories) * 0.02)

    def energy(self, g: torch.Tensor) -> torch.Tensor:
        """Compute Hopfield network energy.

        Args:
            g: Input tensor of shape (..., d_model)

        Returns
        -------
            Energy value (scalar)
        """
        # Compute hidden activations
        hidden = torch.matmul(g, self.xi)  # (..., n_memories)

        # Energy is negative sum of squared ReLU activations
        energy = -0.5 * (f.relu(hidden) ** 2).sum()

        return energy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computes gradient of energy.

        Note: This is typically not called directly in Energy Transformer.
        Instead, the gradient is computed via autograd during energy minimization.

        Args:
            x: Input tensor

        Returns
        -------
            Gradient of energy w.r.t. input
        """
        x.requires_grad_(True)
        energy = self.energy(x)
        grad = torch.autograd.grad(energy, x, create_graph=True)[0]
        return grad
