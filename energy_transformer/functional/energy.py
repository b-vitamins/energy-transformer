"""Functional utilities for energy computations."""

import math
from collections.abc import Callable
from typing import Literal, overload

import torch
import torch.nn.functional as f


def attention_energy(
    g: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    beta: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the attention energy functional.

    Parameters
    ----------
    g : torch.Tensor
        Input tensor of shape (..., seq_len, d_model).
    wq : torch.Tensor
        Query weights of shape (n_heads, d_head, d_model).
    wk : torch.Tensor
        Key weights of shape (n_heads, d_head, d_model).
    beta : float or torch.Tensor, optional
        Temperature parameter. Can be a scalar, per-head tensor of shape (n_heads,),
        or None (default: 1/sqrt(d_head)).

    Returns
    -------
    energy : torch.Tensor
        Scalar energy value.
    """
    n_heads, d_head, _ = wq.shape

    if beta is None:
        beta = 1.0 / math.sqrt(d_head)

    # Compute queries and keys
    q = torch.einsum("...qd,hzd->...qhz", g, wq)
    k = torch.einsum("...kd,hzd->...khz", g, wk)

    # Compute attention scores
    scores = torch.einsum("...qhz,...khz->...hqk", q, k)

    if isinstance(beta, torch.Tensor) and beta.shape == (n_heads,):
        beta_expanded = beta.view(n_heads, 1, 1)
        weighted_scores = beta_expanded * scores
        a = torch.logsumexp(weighted_scores, dim=-1)
        energy = -(1.0 / beta).unsqueeze(-1) * a
        energy = energy.sum()
    else:
        a = torch.logsumexp(beta * scores, dim=-1)
        energy = -1.0 / beta * a.sum()

    return energy


def hopfield_energy(g: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hopfield network energy functional.

    Parameters
    ----------
    g : torch.Tensor
        Input tensor of shape (..., d_model).
    xi : torch.Tensor
        Memory patterns of shape (d_model, n_memories).

    Returns
    -------
    energy : torch.Tensor
        Scalar energy value.
    """
    hidden = torch.matmul(g, xi)
    energy = -0.5 * (f.relu(hidden) ** 2).sum()
    return energy


def layer_norm_energy(
    x: torch.Tensor,
    gamma: torch.Tensor,
    delta: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Compute the layer normalization energy functional.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (..., d_model).
    gamma : torch.Tensor
        Scale parameter of shape (d_model,).
    delta : torch.Tensor, optional
        Bias parameter of shape (d_model,). Default is None.
    eps : float, optional
        Small epsilon for numerical stability. Default is 1e-5.

    Returns
    -------
    energy : torch.Tensor
        Scalar energy value.
    """
    dtype = x.dtype
    x_double = x.double()
    gamma_double = gamma.double()
    delta_double = None if delta is None else delta.double()

    x_centered = x_double - x_double.mean(dim=-1, keepdim=True)
    variance = (x_centered**2).mean(dim=-1, keepdim=True)
    normalized = gamma_double * x_centered / torch.sqrt(variance + eps)

    if delta_double is not None:
        normalized = normalized + delta_double

    # Lagrangian computation
    d = x_double.shape[-1]
    lagrangian = d * gamma_double * torch.sqrt(variance + eps).sum()
    if delta_double is not None:
        lagrangian = lagrangian + (delta_double * x_double).sum()

    energy = (normalized * x_double).sum() - lagrangian
    return energy.to(dtype)


def total_energy(
    x: torch.Tensor,
    layer_norm_params: dict[str, torch.Tensor],
    attention_params: dict[str, torch.Tensor],
    hopfield_params: dict[str, torch.Tensor],
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Compute the total Energy Transformer energy.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    layer_norm_params : dict[str, torch.Tensor]
        Dictionary containing 'gamma' and optionally 'delta' for layer normalization.
    attention_params : dict[str, torch.Tensor]
        Dictionary containing 'wq', 'wk', and optionally 'beta' for attention.
    hopfield_params : dict[str, torch.Tensor]
        Dictionary containing 'xi' memory patterns for Hopfield energy.
    eps : float, optional
        Small epsilon for numerical stability. Default is 1e-5.

    Returns
    -------
    total : torch.Tensor
        Scalar total energy.
    """
    gamma = layer_norm_params["gamma"]
    delta = layer_norm_params.get("delta")

    ln_e = layer_norm_energy(x, gamma, delta, eps)

    # Normalize input for attention and Hopfield
    x_centered = x - x.mean(dim=-1, keepdim=True)
    variance = (x_centered**2).mean(dim=-1, keepdim=True)
    g = gamma * x_centered / torch.sqrt(variance + eps)
    if delta is not None:
        g = g + delta

    wq = attention_params["wq"]
    wk = attention_params["wk"]
    beta = attention_params.get("beta")
    attn_e = attention_energy(g, wq, wk, beta)

    xi = hopfield_params["xi"]
    hop_e = hopfield_energy(g, xi)

    return ln_e + attn_e + hop_e


def energy_gradient(
    x: torch.Tensor,
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    create_graph: bool = True,
) -> torch.Tensor:
    """
    Compute the gradient of the energy with respect to the input.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor for which the gradient is computed.
    energy_fn : Callable[[torch.Tensor], torch.Tensor]
        Function that computes the energy given x.
    create_graph : bool, optional
        If True, create computational graph for higher-order gradients. Default is True.

    Returns
    -------
    grad : torch.Tensor
        Gradient of the energy with respect to x.
    """
    state = x.clone().requires_grad_(True)
    energy_val = energy_fn(state)
    (grad,) = torch.autograd.grad(energy_val, state, create_graph=create_graph)
    return grad


@overload
def minimize_energy(
    x: torch.Tensor,
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_steps: int = ...,  # noqa: FBT003
    alpha: float = ...,  # noqa: FBT003
    return_trajectory: Literal[False] = ...,  # noqa: FBT003
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def minimize_energy(
    x: torch.Tensor,
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_steps: int = ...,  # noqa: FBT003
    alpha: float = ...,  # noqa: FBT003
    return_trajectory: Literal[True],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


def minimize_energy(
    x: torch.Tensor,
    energy_fn: Callable[[torch.Tensor], torch.Tensor],
    n_steps: int = 10,
    alpha: float = 0.1,
    return_trajectory: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """
    Minimize the energy via gradient descent.

    Parameters
    ----------
    x : torch.Tensor
        Initial input state.
    energy_fn : Callable[[torch.Tensor], torch.Tensor]
        Function to compute energy from the input.
    n_steps : int, optional
        Number of gradient descent steps. Default is 10.
    alpha : float, optional
        Step size. Default is 0.1.
    return_trajectory : bool, optional
        If True, return the trajectory of states. Default is False.

    Returns
    -------
    final_state : torch.Tensor
        Final state after gradient descent.
    energies : torch.Tensor
        Tensor of energy values at each step.
    trajectory : torch.Tensor, optional
        Only returned if return_trajectory is True.
        Tensor of shape (n_steps+1, ...) containing intermediate states.
    """
    state = x.clone().requires_grad_(True)
    energies_list: list[float] = []
    if return_trajectory:
        trajectory_list = [state.clone().detach()]
        for _ in range(n_steps):
            energy_val = energy_fn(state)
            energies_list.append(energy_val.item())
            (grad,) = torch.autograd.grad(energy_val, state, create_graph=True)
            state = state - alpha * grad
            trajectory_list.append(state.clone().detach())

        energies = torch.tensor(energies_list)
        trajectory = torch.stack(trajectory_list)
        return state, energies, trajectory
    else:
        for _ in range(n_steps):
            energy_val = energy_fn(state)
            energies_list.append(energy_val.item())
            (grad,) = torch.autograd.grad(energy_val, state, create_graph=True)
            state = state - alpha * grad

        energies = torch.tensor(energies_list)
        return state, energies
