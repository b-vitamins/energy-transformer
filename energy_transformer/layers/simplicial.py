"""Simplicial Hopfield Network implementation."""

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .base import BaseHopfieldNetwork


class SimplicialHopfieldNetwork(BaseHopfieldNetwork):
    """Simplicial Hopfield Network with higher-order interactions.

    The energy function is defined as:
    E = -lse(β, ξᵀ g, Σ) + 0.5 * ||g||²

    where:
    - β is the inverse temperature parameter
    - ξ ∈ ℝᴷˣᴰ are the stored patterns
    - g ∈ ℝᴰ is the current state
    - Σ is the set of simplices defining higher-order interactions

    Parameters
    ----------
    in_dim : int
        Input dimension D of token vectors
    simplices : Sequence[Sequence[int]], optional
        Set Σ of simplices defining higher-order interactions between features.
        Each inner sequence represents a simplex σ.
        If None, simplices will be generated automatically.
    β : float, optional
        Inverse temperature parameter. Must be positive. Default: 1.0.
    hidden_dim : int, optional
        Number of memory patterns K. If None, set to int(in_dim * multiplier).
    multiplier : float, optional
        Multiplier for default hidden_dim when hidden_dim is None.
    c : float, optional
        Budget multiplier for automatic simplex generation.
        Default: 1.0.
    simplex_dim : int, optional
        Maximum simplex dimension for automatic generation. Default: 2.
    proportion : tuple[float, ...], optional
        Proportions for each simplex dimension from 1 to simplex_dim-1.
        Must satisfy len(proportion) == simplex_dim-1 & sum(proportion) < 1.0.
        Default: (0.5,) meaning 50% edges, 50% triangles.
    seed : int, optional
        Random seed for reproducible simplex generation. Default: None.
    """

    def __init__(
        self,
        in_dim: int,
        simplices: Sequence[Sequence[int]] | None = None,
        β: float = 1.0,
        hidden_dim: int | None = None,
        multiplier: float = 2.0,
        c: float = 1.0,
        simplex_dim: int = 2,
        proportion: tuple[float, ...] = (0.5,),
        seed: int | None = None,
    ):
        """Initialize the Simplicial Hopfield Network.

        Parameters
        ----------
        in_dim : int
            Input dimension D of token vectors
        simplices : Sequence[Sequence[int]], optional
            Set Σ of simplices defining higher-order feature interactions.
            If None, generates simplices automatically.
        β : float, optional
            Inverse temperature parameter (must be positive). Default: 1.0.
        hidden_dim : int, optional
            Number of memory patterns K.
        multiplier : float, optional
            Multiplier for default hidden_dim
        c : float, optional
            Budget multiplier for automatic simplex generation
        simplex_dim : int, optional
            Maximum simplex dimension for automatic generation
        proportion : tuple[float, ...], optional
            Relative proportions for simplex dimensions 1 to simplex_dim-1
        seed : int, optional
            Random seed for reproducible generation

        Raises
        ------
        ValueError
            If β <= 0,
            if simplex_dim-1 != len(proportion),
            if sum(proportion) >= 1.0,
            if any simplex σ ∈ Σ contains invalid feature indices, or
            if too many simplices are provided (> D²(D-1)/20).
        """
        super().__init__()

        if β <= 0:
            raise ValueError("β must be positive")

        if hidden_dim is None:
            hidden_dim = int(in_dim * multiplier)

        # Validate automatic generation parameters if needed
        if simplices is None:
            if simplex_dim < 1:
                raise ValueError("simplex_dim must be >= 1")
            if len(proportion) != simplex_dim - 1:
                raise ValueError(
                    f"len(proportion) must equal simplex_dim-1, "
                    f"got {len(proportion)} != {simplex_dim - 1}"
                )
            if sum(proportion) >= 1.0:
                raise ValueError(
                    f"sum(proportion) must be < 1.0, got {sum(proportion)}"
                )
            if any(p <= 0 for p in proportion):
                raise ValueError("All proportions must be positive")

            # Generate simplices automatically
            simplices = self._generate_simplices(
                in_dim, c, simplex_dim, proportion, seed
            )

        # Validate provided or generated simplices
        if simplices:
            max_idx = max(
                (max(simplex) if simplex else -1) for simplex in simplices
            )
            if max_idx >= in_dim:
                raise ValueError(
                    f"Simplex contains index {max_idx} but in_dim is {in_dim}"
                )

            # Check total simplex count doesn't exceed D²(D-1)/20
            max_simplices = (in_dim * in_dim * (in_dim - 1)) // 20
            if len(simplices) > max_simplices:
                raise ValueError(
                    f"Too many simplices: {len(simplices)} > {max_simplices}"
                )

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.simplices = (
            [list(simplex) for simplex in simplices] if simplices else []
        )
        self.β = β

        # Memory patterns ξ ∈ ℝᴷˣᴰ
        self.ξ = nn.Parameter(torch.empty(hidden_dim, in_dim))  # shape: [K, D]

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters."""
        std = 1.0 / (self.in_dim * self.hidden_dim) ** 0.25
        nn.init.normal_(self.ξ, std=std)

    def forward(self, g: Tensor) -> Tensor:
        """Forward pass - computes energy.

        Parameters
        ----------
        g : Tensor
            Input tensor of shape [..., N, D]

        Returns
        -------
        Tensor
            Scalar energy value
        """
        e_shn = self.energy(g)
        # Ensure the energy function returns a scalar
        assert e_shn.ndim == 0, "Energy function must return a scalar tensor"

        return e_shn

    def energy(self, g: Tensor) -> Tensor:
        """Compute simplicial Hopfield energy.

        For input g of shape [..., N, D], computes energy
        for each of the N tokens and returns the total scalar energy.

        Parameters
        ----------
        g : Tensor
            Input tensor of shape [..., N, D] where N is the number
            of tokens and D is the feature dimension

        Returns
        -------
        Tensor
            Scalar tensor
        """
        lse_term = self._lse(g)
        regularisation = 0.5 * (g * g).sum(dim=-1)  # (batch_size * N,)

        # Sum over all tokens and batch elements
        return (-lse_term + regularisation).sum()  # scalar

    def _lse(self, g: torch.Tensor) -> torch.Tensor:
        """Compute log-sum-exp term for simplicial Hopfield energy.

        Parameters
        ----------
        g : torch.Tensor
            Input tensor of shape [..., D] where D is the feature dimension.

        Returns
        -------
        torch.Tensor
            LSE values of shape [...] (one per batch element).

        Raises
        ------
        ValueError
            If β <= 0, invalid tensor dimensions, features or device mismatch.
        """
        if self.β <= 0:
            raise ValueError("β>0 required")
        if self.ξ.ndim != 2 or g.ndim < 1:
            raise ValueError("ξ(2-D) & g(≥1-D)")
        if self.ξ.shape[1] != g.shape[-1]:
            raise ValueError("feature mismatch")
        if self.ξ.device != g.device:
            raise ValueError("different devices")

        # Get dimensions
        d = g.shape[-1]
        batch_shape = g.shape[:-1]

        # Flatten batch dimensions
        g_flat = g.view(-1, d)  # (batch_size, D)
        batch_size = g_flat.shape[0]

        # Initialize accumulator for each batch element
        total_exp_sum = torch.zeros(batch_size, device=g.device, dtype=g.dtype)

        # Vectorize across patterns and batch elements for each simplex
        for σ in self.simplices:
            if len(σ) == 0:
                continue

            # Extract relevant parts
            ξ_sigma = self.ξ[:, σ]  # (P, |σ|) - all patterns for this simplex
            g_sigma = g_flat[
                :, σ
            ]  # (batch_size, |σ|) - all batch elements for this simplex

            # Compute all dot products at once: (batch_size, P)
            dot_products = torch.mm(g_sigma, ξ_sigma.t())

            # Sum exp(β * dot_products) across patterns for each batch element
            exp_terms = torch.exp(self.β * dot_products)  # (batch_size, P)
            total_exp_sum += exp_terms.sum(dim=1)  # (batch_size,)

        # Compute LSE
        lse_results = torch.where(
            total_exp_sum > 0,
            (1 / self.β) * torch.log(total_exp_sum),
            torch.zeros_like(total_exp_sum),
        )

        return lse_results.view(batch_shape)

    def update(self, g: Tensor) -> Tensor:
        """Compute simplicial Hopfield update rule.

        Implements the update rule:
        S⁽ᵗ⁾ = softmax(β * ∑_σ∈Σ Ξ_σᵀ S_σ⁽ᵗ⁻¹⁾) Ξ

        Parameters
        ----------
        g : Tensor
            Current state tensor of shape [..., N, D] where
            N is number of tokens and D is feature dimension.

        Returns
        -------
        Tensor
            Updated state tensor of same shape as input.

        Raises
        ------
        ValueError
            If β <= 0,
            if invalid tensor dimension, or
            if feature dimension mismatch.
        """
        if self.β <= 0:
            raise ValueError("β>0 required")
        if self.ξ.ndim != 2 or g.ndim < 1 or self.ξ.shape[1] != g.shape[-1]:
            raise ValueError("shape mismatch")

        # Get dimensions
        p, d = self.ξ.shape[0], g.shape[-1]

        # Flatten batch dimensions
        g_flat = g.view(-1, d)  # (batch_size, D)
        batch_size = g_flat.shape[0]

        # Initialize similarities accumulator: (batch_size, P)
        similarities = torch.zeros(
            batch_size, p, device=g.device, dtype=g.dtype
        )

        # Vectorize across patterns and batch elements for each simplex
        for σ in self.simplices:
            if len(σ) == 0:
                continue

            # Extract relevant parts
            ξ_sigma = self.ξ[:, σ]  # (P, |σ|) - all patterns for this simplex
            g_sigma = g_flat[
                :, σ
            ]  # (batch_size, |σ|) - all batch elements for this simplex

            # Compute all dot products and accumulate: (batch_size, P)
            similarities += torch.mm(g_sigma, ξ_sigma.t())

        # Temperature-scaled softmax across patterns: (batch_size, P)
        α = torch.softmax(similarities / self.β, dim=1)

        # Weighted combination of patterns: (batch_size, D)
        result = torch.mm(α, self.ξ)

        return result.view(g.shape)

    @staticmethod
    def unrank(r: int, d: int, p: int) -> tuple[int, ...]:
        """Convert combinatorial rank to k-element subset.

        Parameters
        ----------
        r : int
            Combinatorial rank in lexicographic order.
        d : int
            Size of the universe.
        p : int
            Size of the subset.

        Returns
        -------
        tuple[int, ...]
            p-element subset in lexicographic order.

        Raises
        ------
        ValueError
            If rank is out of range for comb(d, p).
        """
        if not (0 <= r < math.comb(d, p)):
            raise ValueError(f"Rank {r} out of range for comb({d}, {p})")
        if p == 0:
            return ()
        if p == 1:
            return (r,)
        if p == d:
            return tuple(range(d))
        result = []
        remaining = p
        budget = r
        for i in range(d):
            if remaining == 0:
                break
            ways_without = math.comb(d - i - 1, remaining - 1)
            if budget < ways_without:
                result.append(i)
                remaining -= 1
            else:
                budget -= ways_without
        return tuple(result)

    @staticmethod
    def select_random_simplices(
        d: int, n_simp: int, dim: int, generator: torch.Generator
    ) -> list[list[int]]:
        """Randomly choose n_simp distinct simplices of dimension dim.

        Parameters
        ----------
        d : int
            The size of the set from which to select simplices.
        n_simp : int
            The number of simplices to select.
        dim : int
            The dimension of the simplices.
        generator : torch.Generator
            A torch Generator for reproducibility.

        Returns
        -------
        list[list[int]]
            List of simplices, each represented as a list of sorted vertex ids.

        Raises
        ------
        ValueError
            If n_simp > total number of possible simplices.
        """
        p = dim + 1
        total = math.comb(d, p)
        if n_simp > total:
            raise ValueError(
                f"Requested {n_simp} but only {total} possible {p}-simplices."
            )

        # Use torch to generate random ranks without replacement
        chosen_ranks = torch.randperm(total, generator=generator)[:n_simp]

        simplices = []
        for rank in chosen_ranks:
            simplex = SimplicialHopfieldNetwork.unrank(rank.item(), d, p)
            simplices.append(list(simplex))
        return simplices

    @staticmethod
    def _generate_simplices(
        d: int,
        c: float,
        simplex_dim: int,
        proportion: tuple[float, ...],
        seed: int | None,
    ) -> list[list[int]]:
        """Generate simplices automatically based on budget and proportions.

        Parameters
        ----------
        d : int
            Feature dimension (size of universe).
        c : float
            Budget multiplier. Total budget is c * d * (d-1) / 2.
        simplex_dim : int
            Maximum simplex dimension.
        proportion : tuple[float, ...]
            Relative proportions for dimensions 1 to simplex_dim-1.
        seed : int, optional
            Random seed for reproducible generation.

        Returns
        -------
        list[list[int]]
            Generated simplices as list of vertex index lists.
        """
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        budget = int(c * d * (d - 1) / 2)
        used = 0

        gensimplex = SimplicialHopfieldNetwork.select_random_simplices
        simplices = []

        for i, fraction in enumerate(proportion):
            dim = i + 1  # dimension 1=edges, 2=triangles, etc.
            how_many = int(fraction * budget)
            if how_many > 0:
                bunch = gensimplex(d, how_many, dim, generator)
                simplices.extend(bunch)
                used += how_many

        left = budget - used
        if left > 0:
            last = gensimplex(d, left, simplex_dim, generator)
            simplices.extend(last)
        return simplices
