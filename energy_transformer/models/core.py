"""Core Energy Transformer model."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..config import ETConfig
from ..layers import EnergyAttention, EnergyLayerNorm, HopfieldNetwork


class EnergyTransformer(nn.Module):
    """Energy Transformer model.

    This model defines an energy function over sequences and performs
    inference by minimizing this energy via gradient descent.

    Parameters
    ----------
    config : ETConfig
        Model configuration.
    """

    def __init__(self, config: ETConfig) -> None:
        """Initialize the Energy Transformer components.

        Parameters
        ----------
        config : ETConfig
            Configuration containing dimensions, number of steps, etc.
        """
        super().__init__()
        self.config: ETConfig = config
        self.attention: EnergyAttention = EnergyAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_head=config.d_head,
            per_head_beta=config.per_head_beta,
        )
        self.hopfield: HopfieldNetwork = HopfieldNetwork(
            d_model=config.d_model,
            n_memories=config.n_memories,
        )
        self.layer_norm: EnergyLayerNorm = EnergyLayerNorm(
            d_model=config.d_model,
            use_bias=True,
        )

    def energy(self, x: Tensor) -> Tensor:
        """Compute total energy of the system.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., seq_len, d_model). If it does not
            already require gradients, `requires_grad` will be enabled.

        Returns
        -------
        Tensor
            Scalar energy value.
        """
        if not x.requires_grad:
            x = x.requires_grad_(True)

        ln_e = self.layer_norm.energy(x)
        g = self.layer_norm(x)
        attn_e = self.attention.energy(g)
        hop_e = self.hopfield.energy(g)
        return ln_e + attn_e + hop_e

    def forward(
        self,
        x: Tensor,
        n_steps: int | None = None,
        alpha: float | None = None,
        return_trajectory: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass via iterative energy minimization.

        Parameters
        ----------
        x : Tensor
            Initial state of shape (batch_size, seq_len, d_model).
        n_steps : int, optional
            Number of gradient‐descent steps (default: `config.n_steps`).
        alpha : float, optional
            Step size (default: `config.alpha`).
        return_trajectory : bool, optional
            If True, also return all intermediate states.

        Returns
        -------
        Tensor or (Tensor, Tensor)
            If `return_trajectory=False`, returns the final state tensor.
            Otherwise returns `(final_state, trajectory)`, where
            `trajectory` is a tensor of shape `(n_steps+1, batch_size, seq_len, d_model)`.
        """
        steps: int = n_steps if n_steps is not None else self.config.n_steps
        step_size: float = alpha if alpha is not None else self.config.alpha

        # Prepare state with correct grad setting
        if x.requires_grad:
            state: Tensor = x
        else:
            state = x.clone().detach().requires_grad_(True)

        trajectory: list[Tensor] = []
        if return_trajectory:
            trajectory.append(state.clone().detach())

        for _ in range(steps):
            e = self.energy(state)
            grad = torch.autograd.grad(e, state, create_graph=True)[0]
            state = state - step_size * grad
            if return_trajectory:
                trajectory.append(state.clone().detach())

        if return_trajectory:
            return state, torch.stack(trajectory)
        return state

    def compute_energy_trajectory(
        self,
        x: Tensor,
        n_steps: int | None = None,
        alpha: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute energy at each step of minimization.

        Parameters
        ----------
        x : Tensor
            Initial state tensor.
        n_steps : int, optional
            Number of gradient‐descent steps (default: `config.n_steps`).
        alpha : float, optional
            Step size (default: `config.alpha`).

        Returns
        -------
        (Tensor, Tensor)
            A tuple `(final_state, energy_values)` where
            `energy_values` is a 1D tensor of length `n_steps`.
        """
        steps: int = n_steps if n_steps is not None else self.config.n_steps
        step_size: float = alpha if alpha is not None else self.config.alpha

        state: Tensor = x.clone().detach().requires_grad_(True)
        energies: list[float] = []

        for _ in range(steps):
            e = self.energy(state)
            energies.append(e.item())
            grad = torch.autograd.grad(e, state)[0]
            state = state - step_size * grad
            state = state.detach().requires_grad_(True)

        return state.detach(), torch.tensor(energies)
