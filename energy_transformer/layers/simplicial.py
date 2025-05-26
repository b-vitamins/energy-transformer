"""Simplicial Hopfield Network."""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

try:
    import numpy as np
    from scipy.spatial import Delaunay, cKDTree

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .base import BaseHopfieldNetwork

__all__: Final = [
    "SimplicialHopfieldNetwork",
]


class SimplexValidator:
    """Validates and canonicalizes simplices, ensuring they meet requirements."""

    @staticmethod
    def validate_and_canonicalize(
        simplices: Sequence[Sequence[int]],
    ) -> tuple[dict[int, list[list[int]]], int]:
        """Validate and canonicalize simplices, grouping by size.

        Parameters
        ----------
        simplices : Sequence[Sequence[int]]
            Collection of simplices, each simplex is a sequence of vertex indices.

        Returns
        -------
        tuple[dict[int, list[list[int]]], int]
            Dictionary mapping simplex sizes to lists of canonical simplices,
            and the maximum vertex index found.

        Raises
        ------
        ValueError
            If simplices is empty, contains simplices of size < 2, has negative
            vertex indices, or contains duplicate vertices within a simplex.
        TypeError
            If vertex indices are not integers.
        """
        if not simplices:
            raise ValueError("'simplices' must contain at least one simplex.")

        buckets: dict[int, list[list[int]]] = defaultdict(list)
        max_vertex = -1
        seen: set[tuple[int, ...]] = set()

        for simplex in simplices:
            if len(simplex) < 2:
                raise ValueError(
                    "Simplices of size <2 are not allowed (no self-loops)."
                )

            canonical = SimplexValidator._canonicalize_simplex(simplex)
            tup = tuple(canonical)

            if tup in seen:
                continue
            seen.add(tup)

            buckets[len(tup)].append(list(tup))
            max_vertex = max(max_vertex, tup[-1])

        return buckets, max_vertex

    @staticmethod
    def _canonicalize_simplex(simplex: Sequence[int]) -> list[int]:
        """Convert a simplex to canonical form (sorted list of unique integers)."""
        canonical = []
        for v in simplex:
            try:
                v_int = int(v)
            except (ValueError, TypeError) as err:
                raise TypeError(
                    f"Vertex indices must be integers; got {type(v).__name__}"
                ) from err
            if v_int < 0:
                raise ValueError(
                    f"Vertex indices must be non-negative; got {v_int}"
                )
            canonical.append(v_int)

        if len(set(canonical)) != len(canonical):
            raise ValueError(f"Simplex {simplex} has duplicate vertices.")

        canonical.sort()
        return canonical


class SimplexBudgetManager:
    """Manages budget allocation for simplex generation."""

    @staticmethod
    def edge_units(m: int) -> int:
        """Return budget cost (number of internal edges) of an m-simplex.

        Parameters
        ----------
        m : int
            Size of the simplex (number of vertices).

        Returns
        -------
        int
            Number of edges within the simplex, computed as C(m, 2).
        """
        return math.comb(m, 2)

    @staticmethod
    def normalize_dimension_weights(
        dim_weights: dict[int, float] | None, max_dim: int
    ) -> dict[int, float]:
        """Normalize dimension weights to sum to 1.

        Parameters
        ----------
        dim_weights : dict[int, float] | None
            Raw dimension weights mapping dimension to weight.
        max_dim : int
            Maximum dimension to consider.

        Returns
        -------
        dict[int, float]
            Normalized weights summing to 1.

        Raises
        ------
        ValueError
            If no valid weights are provided.
        """
        if dim_weights is None:
            return {1: 1.0}  # Default: all budget to edges

        weights = {
            k: v for k, v in dim_weights.items() if 1 <= k <= max_dim and v > 0
        }

        if not weights:
            raise ValueError(
                "dim_weights must assign positive weight to at least one "
                f"dimension <= {max_dim}."
            )

        weight_sum = sum(weights.values())
        return {k: v / weight_sum for k, v in weights.items()}

    @staticmethod
    def allocate_budget(
        budget: int,
        dim_weights: dict[int, float],
        candidates_by_size: dict[int, list[tuple[int, ...]]],
    ) -> tuple[list[list[int]], int]:
        """Allocate budget across dimensions to generate simplices.

        Parameters
        ----------
        budget : int
            Total edge budget available.
        dim_weights : dict[int, float]
            Normalized dimension weights.
        candidates_by_size : dict[int, list[tuple[int, ...]]]
            Candidate simplices grouped by size.

        Returns
        -------
        tuple[list[list[int]], int]
            Generated simplices and remaining budget.
        """
        consumed: list[list[int]] = []
        remaining = budget

        for dim, weight in dim_weights.items():
            simplex_size = dim + 1
            allowance = min(int(budget * weight), remaining)
            cost_per_simplex = SimplexBudgetManager.edge_units(simplex_size)

            candidates = candidates_by_size.get(simplex_size)
            if not candidates:
                continue

            while allowance >= cost_per_simplex and candidates:
                simplex = candidates.pop()
                consumed.append(list(simplex))
                allowance -= cost_per_simplex
                remaining -= cost_per_simplex

        return consumed, remaining


class SimplexGenerator(ABC):
    """Abstract base class for simplex generation strategies."""

    def __init__(self, rng: random.Random | None = None):
        """Initialize generator with optional random number generator."""
        self.rng = rng or random.Random()

    @abstractmethod
    def generate(
        self,
        num_vertices: int,
        max_dim: int,
        budget: int,
        dim_weights: dict[int, float] | None = None,
    ) -> list[list[int]]:
        """Generate simplices according to the specific strategy.

        Parameters
        ----------
        num_vertices : int
            Number of vertices in the complex.
        max_dim : int
            Maximum simplex dimension to generate.
        budget : int
            Edge budget for generation.
        dim_weights : dict[int, float] | None
            Optional dimension weights.

        Returns
        -------
        list[list[int]]
            List of generated simplices.
        """
        pass


class RandomSimplexGenerator(SimplexGenerator):
    """Generates simplices randomly within budget constraints."""

    def generate(
        self,
        num_vertices: int,
        max_dim: int,
        budget: int,
        dim_weights: dict[int, float] | None = None,
    ) -> list[list[int]]:
        """Generate random simplices within budget.

        Uses a budget-based approach to randomly select simplices
        from all possible combinations, weighted by dimension.
        """
        # Normalize weights
        dim_weights = SimplexBudgetManager.normalize_dimension_weights(
            dim_weights, max_dim
        )

        # Generate all candidate simplices
        candidates_by_size = self._generate_candidates(
            num_vertices, max_dim, dim_weights
        )

        # Allocate budget to generate simplices
        simplices, remaining = SimplexBudgetManager.allocate_budget(
            budget, dim_weights, candidates_by_size
        )

        # Use remaining budget for edges if available
        if remaining >= SimplexBudgetManager.edge_units(2):
            edges = candidates_by_size.get(2, [])
            while remaining >= 1 and edges:
                simplex = edges.pop()
                simplices.append(list(simplex))
                remaining -= 1

        return simplices

    def _generate_candidates(
        self,
        num_vertices: int,
        max_dim: int,
        dim_weights: dict[int, float],
    ) -> dict[int, list[tuple[int, ...]]]:
        """Generate and shuffle candidate simplices by dimension."""
        candidates: dict[int, list[tuple[int, ...]]] = {}

        for simplex_size in range(2, max_dim + 2):
            if simplex_size - 1 not in dim_weights:
                continue

            all_combinations = list(
                combinations(range(num_vertices), simplex_size)
            )
            self.rng.shuffle(all_combinations)
            candidates[simplex_size] = all_combinations

        return candidates


class TopologyAwareSimplexGenerator(SimplexGenerator):
    """Generates simplices based on spatial topology using k-NN and Delaunay."""

    def __init__(
        self,
        coordinates: Sequence[Sequence[float]],
        k_neighbors: int | None = None,
        include_delaunay: bool = True,
        rng: random.Random | None = None,
    ):
        """Initialize with spatial coordinates and parameters.

        Parameters
        ----------
        coordinates : Sequence[Sequence[float]]
            Spatial coordinates for each vertex.
        k_neighbors : int | None
            Number of nearest neighbors. Auto-determined if None.
        include_delaunay : bool
            Whether to include Delaunay triangulation.
        rng : random.Random | None
            Random number generator for sampling.
        """
        super().__init__(rng)
        if not HAS_SCIPY:
            raise ImportError(
                "scipy is required for topology-aware simplex generation"
            )
        self.coordinates = np.array(coordinates)
        self.k_neighbors = k_neighbors
        self.include_delaunay = include_delaunay

    def generate(
        self,
        num_vertices: int,
        max_dim: int,
        budget: int,
        dim_weights: dict[int, float] | None = None,
    ) -> list[list[int]]:
        """Generate topology-aware simplices.

        Uses k-NN for edges and optionally Delaunay triangulation
        for higher-order simplices. The budget parameter is interpreted
        as a fraction for subsampling if needed.
        """
        if len(self.coordinates) != num_vertices:
            raise ValueError(
                f"Number of coordinates ({len(self.coordinates)}) must match "
                f"num_vertices ({num_vertices})."
            )

        simplices = []

        # Add k-NN edges
        if num_vertices > 1:
            simplices.extend(self._generate_knn_edges(num_vertices))

        # Add Delaunay simplices
        if self.include_delaunay and max_dim >= 2 and num_vertices >= 3:
            self._add_delaunay_simplices(simplices, max_dim)

        # Remove duplicates
        unique_simplices = self._remove_duplicates(simplices)

        # Apply budget as sampling fraction
        budget_fraction = budget  # Interpret as fraction for topology-aware
        if 0 < budget_fraction < 1.0 and len(unique_simplices) > 0:
            n_keep = max(1, int(len(unique_simplices) * budget_fraction))
            unique_simplices = self.rng.sample(unique_simplices, n_keep)

        return unique_simplices

    def _determine_k_neighbors(self, num_vertices: int) -> int:
        """Adaptively determine k based on dimensionality and size."""
        if self.k_neighbors is not None:
            return self.k_neighbors

        n_dims = self.coordinates.shape[1]
        return min(
            int(2 * n_dims + 3),  # Higher k for higher dimensions
            int(np.sqrt(num_vertices)),  # Scale with size
            num_vertices - 1,  # Can't exceed n-1
        )

    def _generate_knn_edges(self, num_vertices: int) -> list[list[int]]:
        """Generate edges using k-nearest neighbors."""
        k = self._determine_k_neighbors(num_vertices)
        tree = cKDTree(self.coordinates)
        _, indices = tree.query(self.coordinates, k=min(k + 1, num_vertices))

        edges = []
        edges_set = set()

        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # Skip self
                edge = tuple(sorted([i, j]))
                if edge not in edges_set:
                    edges_set.add(edge)
                    edges.append(list(edge))

        return edges

    def _add_delaunay_simplices(
        self, simplices: list[list[int]], max_dim: int
    ) -> None:
        """Add Delaunay triangulation simplices."""
        try:
            tri = Delaunay(self.coordinates)

            if self.coordinates.shape[1] == 2:  # 2D case
                for simplex in tri.simplices:
                    simplices.append(sorted(simplex.tolist()))

            elif self.coordinates.shape[1] >= 3 and max_dim >= 3:
                for simplex in tri.simplices:
                    # Add full simplex if within max_dim
                    if len(simplex) - 1 <= max_dim:
                        simplices.append(sorted(simplex.tolist()))

                    # Add all faces
                    for r in range(2, min(len(simplex), max_dim + 2)):
                        for subset in combinations(simplex, r):
                            simplices.append(sorted(subset))
        except Exception:
            # Delaunay might fail for degenerate configurations
            pass

    @staticmethod
    def _remove_duplicates(simplices: list[list[int]]) -> list[list[int]]:
        """Remove duplicate simplices while preserving order."""
        seen = set()
        unique = []

        for simplex in simplices:
            key = tuple(simplex)
            if key not in seen:
                seen.add(key)
                unique.append(simplex)

        return unique


class SimplexFactory:
    """Factory for creating appropriate simplex generators."""

    @staticmethod
    def create_generator(
        coordinates: Sequence[Sequence[float]] | None = None,
        rng: random.Random | None = None,
    ) -> SimplexGenerator:
        """Create appropriate generator based on available data.

        Parameters
        ----------
        coordinates : Sequence[Sequence[float]] | None
            Spatial coordinates for topology-aware generation.
        rng : random.Random | None
            Random number generator.

        Returns
        -------
        SimplexGenerator
            Either TopologyAwareSimplexGenerator or RandomSimplexGenerator.
        """
        if coordinates is not None and HAS_SCIPY:
            return TopologyAwareSimplexGenerator(
                coordinates, include_delaunay=True, rng=rng
            )
        return RandomSimplexGenerator(rng)


# Module-level functions for backward compatibility
def _validate_and_canonicalise(
    simplices: Sequence[Sequence[int]],
) -> tuple[dict[int, list[list[int]]], int]:
    """Backward compatibility wrapper for SimplexValidator."""
    return SimplexValidator.validate_and_canonicalize(simplices)


def _autogen_simps(
    num_vertices: int,
    *,
    max_dim: int,
    twiddle: float,
    dim_weights: dict[int, float] | None = None,
    coordinates: Sequence[Sequence[float]] | None = None,
    rng: random.Random | None = None,
) -> list[list[int]]:
    """Generate a simplicial complex with optional topology awareness.

    Parameters
    ----------
    num_vertices : int
        Number of vertices in the complex.
    max_dim : int
        Highest simplex dimension to include (>=1 for edges).
    twiddle : float
        Fraction of the full pairwise budget N(N-1)/2 to use.
    dim_weights : dict[int, float] | None
        Mapping from simplex size to weight. Missing sizes get weight 0.
    coordinates : Sequence[Sequence[float]] | None
        Optional spatial coordinates for topology-aware generation.
    rng : random.Random | None
        Random number generator for reproducibility.

    Returns
    -------
    list[list[int]]
        List of generated simplices.

    Raises
    ------
    ValueError
        If parameters are invalid.
    RuntimeError
        If generation fails.
    """
    if num_vertices < 2:
        raise ValueError("Need at least 2 vertices to build a complex.")
    if max_dim < 1:
        raise ValueError("max_dim must be >=1 (edges).")

    max_dim = min(max_dim, num_vertices - 1)

    # Create appropriate generator
    generator = SimplexFactory.create_generator(coordinates, rng)

    # For topology-aware, twiddle is used as budget fraction directly
    if isinstance(generator, TopologyAwareSimplexGenerator):
        simplices = generator.generate(num_vertices, max_dim, twiddle)
    else:
        # For random generation, convert twiddle to edge budget
        budget = int(twiddle * num_vertices * (num_vertices - 1) // 2)
        if budget < 1:
            raise ValueError("Budget too small – would generate empty complex.")
        simplices = generator.generate(
            num_vertices, max_dim, budget, dim_weights
        )

    if not simplices:
        raise RuntimeError("Auto-generation failed to produce simplices.")

    return simplices


class SimplicialHopfieldNetwork(BaseHopfieldNetwork):
    """Continuous Simplicial Hopfield Network with optional auto complex.

    This network extends the classical Hopfield model to operate on simplicial
    complexes, allowing for higher-order interactions between neurons through
    simplices of various dimensions.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    simplices : Sequence[Sequence[int]] | None
        Manual specification of simplicial complex. If None, auto-generates.
    num_vertices : int | None
        Number of vertices for auto-generation. Required when simplices=None.
    max_dim : int, default=1
        Maximum simplex dimension for auto-generation.
    budget : float, default=0.1
        Fraction of full edge budget to use for auto-generation.
    dim_weights : dict[int, float] | None
        Weight distribution across simplex dimensions for auto-generation.
    coordinates : Sequence[Sequence[float]] | None
        Spatial coordinates for topology-aware generation. If provided with
        scipy available, uses k-NN + Delaunay triangulation instead of random.
    hidden_dim : int | None
        Hidden dimension for memory matrix. Defaults to in_dim * multiplier.
    multiplier : int, default=4
        Multiplier for default hidden dimension.
    temperature : float, default=0.1
        Temperature parameter for softmax normalization.
    rng : random.Random | None
        Random number generator for reproducible auto-generation.

    Attributes
    ----------
    max_vertex : int
        Maximum vertex index in the simplicial complex.
    simps_by_size : dict[int, Tensor]
        Registered buffers containing simplices grouped by size.
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden dimension of memory matrix.
    T : float
        Temperature parameter.
    ξ : nn.Parameter
        Learnable memory matrix of shape (hidden_dim, in_dim).
    """

    def __init__(
        self,
        in_dim: int,
        simplices: Sequence[Sequence[int]] | None = None,
        *,
        num_vertices: int | None = None,
        max_dim: int = 1,
        budget: float = 0.1,
        dim_weights: dict[int, float] | None = None,
        coordinates: Sequence[Sequence[float]] | None = None,
        hidden_dim: int | None = None,
        multiplier: int = 4,
        temperature: float = 0.5,
        rng: random.Random | None = None,
    ) -> None:
        """Initialize the Simplicial Hopfield Network."""
        super().__init__()

        if simplices is None:
            if num_vertices is None:
                raise ValueError(
                    "Must specify num_vertices when simplices=None."
                )
            simplices = _autogen_simps(
                num_vertices,
                max_dim=max_dim,
                twiddle=budget,
                dim_weights=dim_weights,
                coordinates=coordinates,
                rng=rng,
            )

        buckets, max_vertex = _validate_and_canonicalise(simplices)
        self.max_vertex: int = max_vertex
        self.simps_by_size: dict[int, Tensor] = {}

        for m, bucket in buckets.items():
            tensor = torch.as_tensor(bucket, dtype=torch.long)
            self.register_buffer(f"simplices_m{m}", tensor, persistent=False)
            self.simps_by_size[m] = tensor

        if hidden_dim is None:
            hidden_dim = int(in_dim * multiplier)

        self.in_dim: int = int(in_dim)
        self.hidden_dim: int = hidden_dim
        self.T: float = float(temperature)

        self.ξ = nn.Parameter(torch.empty(hidden_dim, in_dim))
        self.reset_parameters()

    @staticmethod
    def _simplex_product(u: Tensor, idx: Tensor) -> Tensor:
        """Compute ϕ_i(σ)=∏_{v∈σ} g_i(v) for all simplices σ in ``idx``.

        This method computes the product without materialising an intermediate
        ``(..., k, sₘ, n_tokens)`` tensor.

        Parameters
        ----------
        u : Tensor
            Shape (..., k, n_tokens).
        idx : Tensor
            Long tensor of shape (sₘ, m) with vertex indices for m-simplices.

        Returns
        -------
        Tensor
            Shape (..., k, sₘ) containing the products per simplex.
        """
        prod = torch.index_select(u, -1, idx[:, 0])  # (..., k, sₘ)
        # m ≤ 3 in practice; a tiny Python loop is faster and lighter
        for j in range(1, idx.shape[1]):  # over remaining vertices
            slice_j = torch.index_select(u, -1, idx[:, j])  # (..., k, sₘ)
            prod = prod.mul_(slice_j)
        return prod

    def forward(self, g: Tensor) -> Tensor:
        """Compute the simplicial Hopfield energy.

        Parameters
        ----------
        g : Tensor
            Input tensor of shape (..., n_tokens, in_dim).

        Returns
        -------
        Tensor
            Scalar energy value summed across batch dimensions.

        Raises
        ------
        ValueError
            If input has fewer tokens than required by the simplicial complex.
        """
        num_tokens = g.shape[-2]
        if num_tokens <= self.max_vertex:
            raise ValueError(
                f"Input has {num_tokens} tokens but complex references "
                f"vertex {self.max_vertex}."
            )

        # Linear transformation: (..., n_tokens, hidden_dim)
        u = F.linear(g, self.ξ)
        # Transpose for simplicial operations: (..., hidden_dim, n_tokens)
        u = u.transpose(-1, -2)

        list_of_logits: list[Tensor] = []
        for _m, idx in self.simps_by_size.items():
            if idx.device != u.device:
                idx = idx.to(u.device, non_blocking=True)
            # Vectorised gather-and-product (memory-frugal)
            prod = self._simplex_product(u, idx)  # (..., k, sₘ)
            list_of_logits.append(prod)

        # Concatenate across all simplex sizes: (..., k, total_simplices)
        logits = torch.cat(list_of_logits, dim=-1) / self.T
        # Log-sum-exp over hidden and simplex dimensions: (...,)
        lse = torch.logsumexp(logits, dim=(-2, -1))
        e_shn = -self.T * lse

        # Quadratic regularization: (...,)
        quad = 0.5 * g.pow(2).sum(dim=(-2, -1))
        # Sum over batch dimensions: scalar
        return (e_shn + quad).sum()

    def reset_parameters(self) -> None:
        """Initialize learnable parameters using Xavier-like initialization."""
        std = 1.0 / (self.in_dim * self.hidden_dim) ** 0.25
        nn.init.normal_(self.ξ, std=std)
