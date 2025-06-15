"""Simplicial Hopfield Network with direct gradients.

This module implements a continuous Simplicial Hopfield Network that:
- Uses a single shared pattern matrix Ξ ∈ ℝᴰˣᴴ
- Supports diluted simplices with configurable edge/triangle mix
- Averages energy per simplex for consistent scale
- Provides exact gradients without autograd

The key innovation is treating edges and triangles as continuous
pattern-matching units with softmax attention over patterns.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from .types import Device, Dtype

__all__ = ["SimplicialHopfieldNetwork"]

# Minimum number of vertices to form a triangle
_MIN_TRIANGLE_VERTICES = 3


class SimplicialHopfieldNetwork(nn.Module):
    """Continuous Simplicial Hopfield Network with dilution.

    This implementation uses a shared pattern matrix and supports
    a mixture of 2-simplices (edges) and 3-simplices (triangles)
    with the composition controlled by triangle_fraction.

    Parameters
    ----------
    embed_dim : int
        Input embedding dimension.
    hidden_dim : int, optional
        Hidden dimension (number of patterns). Defaults to 4 * embed_dim.
    triangle_fraction : float, default=0.5
        Fraction of simplices that should be triangles vs edges.
        0.0 = all edges, 1.0 = all triangles, 0.5 = balanced mix.
    beta : float, default=0.1
        Inverse temperature parameter for softmax.
    init_std : float, default=0.02
        Standard deviation for weight initialization.
    device : Device, optional
        Device for parameters.
    dtype : Dtype, optional
        Data type for parameters.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        triangle_fraction: float = 0.5,
        beta: float = 0.1,
        init_std: float = 0.02,
        device: Device = None,
        dtype: Dtype = None,
    ) -> None:
        super().__init__()
        if not (0.0 <= triangle_fraction <= 1.0):
            raise ValueError(
                f"SimplicialHopfieldNetwork: triangle_fraction must be in [0, 1]. "
                f"Got {triangle_fraction}. "
                f"Hint: Use 0.0 for edges only, 1.0 for triangles only, "
                f"0.5 (default) for balanced mix."
            )

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or int(embed_dim * 4)
        self.triangle_fraction = triangle_fraction

        kwargs = {"device": device, "dtype": dtype}
        self.patterns = nn.Parameter(
            torch.randn(embed_dim, self.hidden_dim, **kwargs) * init_std  # type: ignore[arg-type]
        )  # Xi  (D x H)
        self.beta = nn.Parameter(torch.tensor(beta, **kwargs))  # type: ignore[arg-type]

        # Cache mapping sequence length → chosen simplices
        self._simplices_cache: dict[int, tuple[Tensor, Tensor]] = {}

    @staticmethod
    def _all_pairs(n: int, device: torch.device) -> Tensor:
        """Generate all possible pairs of indices."""
        idx = torch.arange(n, device=device)
        return torch.combinations(idx, r=2)

    @staticmethod
    def _all_triangles(n: int, device: torch.device) -> Tensor:
        """Generate all possible triangles of indices."""
        idx = torch.arange(n, device=device)
        return torch.combinations(idx, r=3)

    def _choose_simplices(
        self, n: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Select simplices based on triangle_fraction.

        Returns
        -------
        edges : Tensor
            Selected edge indices of shape (M₂, 2).
        triangles : Tensor
            Selected triangle indices of shape (M₃, 3).
        """
        if n in self._simplices_cache:
            return self._simplices_cache[n]

        total = n * (n - 1) // 2  # N choose 2
        edges_all = self._all_pairs(n, device)
        tris_all = (
            self._all_triangles(n, device)
            if n >= _MIN_TRIANGLE_VERTICES
            else torch.empty(0, 3, dtype=torch.long, device=device)
        )

        m_tri_desired = int(round(self.triangle_fraction * total))
        m_tri = min(m_tri_desired, tris_all.size(0))
        m_edge = min(total - m_tri, edges_all.size(0))

        if m_edge < edges_all.size(0):
            perm = torch.randperm(edges_all.size(0), device=device)[:m_edge]
            edges = edges_all[perm]
        else:
            edges = edges_all

        if m_tri > 0:
            if m_tri < tris_all.size(0):
                perm = torch.randperm(tris_all.size(0), device=device)[:m_tri]
                triangles = tris_all[perm]
            else:
                triangles = tris_all
        else:
            triangles = torch.empty(0, 3, dtype=torch.long, device=device)

        self._simplices_cache[n] = (edges, triangles)
        return edges, triangles

    def compute_energy(self, g: Tensor) -> Tensor:
        """Compute Hopfield energy with per-simplex averaging.

        Parameters
        ----------
        g : Tensor
            Input tensor of shape (B, N, D).

        Returns
        -------
        Tensor
            Scalar energy value averaged over batch, tokens, and simplices.
        """
        b, n = g.shape[:2]
        device = g.device
        edges, triangles = self._choose_simplices(n, device)
        num_simplices = edges.size(0) + triangles.size(0)

        if num_simplices == 0:
            raise RuntimeError(
                f"SimplicialHopfieldNetwork: No simplices selected for N={n}. "
                f"With triangle_fraction={self.triangle_fraction}, need N≥2 for edges "
                f"or N≥3 for triangles."
            )

        # h_vμ = Xi^T g_v  (B,N,H)
        h = torch.einsum("bnd,dk->bnk", g, self.patterns)
        beta = self.beta
        total_lse: Tensor = g.new_zeros(())

        if edges.numel() > 0:
            # Sum patterns for edge vertices
            h_e = h[:, edges[:, 0]] + h[:, edges[:, 1]]  # (B,M₂,H)
            total_lse += torch.logsumexp(beta * h_e, dim=-1).sum()

        if triangles.numel() > 0:
            # Sum patterns for triangle vertices
            h_t = (
                h[:, triangles[:, 0]]
                + h[:, triangles[:, 1]]
                + h[:, triangles[:, 2]]
            )  # (B,M₃,H)
            total_lse += torch.logsumexp(beta * h_t, dim=-1).sum()

        # Per-simplex average for consistent scale
        energy_patterns = -(1.0 / (beta * num_simplices)) * total_lse
        energy_reg = -2.0 * g.pow(2).sum()
        return cast(Tensor, (energy_patterns + energy_reg) / (b * n))

    def compute_grad(self, g: Tensor) -> Tensor:
        """Compute gradient ∂E/∂g with matching per-simplex scaling.

        Parameters
        ----------
        g : Tensor
            Input tensor of shape (B, N, D).

        Returns
        -------
        Tensor
            Gradient tensor of shape (B, N, D).
        """
        b, n, d = g.shape
        device = g.device
        edges, triangles = self._choose_simplices(n, device)
        num_simplices = edges.size(0) + triangles.size(0)

        if num_simplices == 0:
            raise RuntimeError(
                f"SimplicialHopfieldNetwork: No simplices selected for N={n}."
            )

        beta = self.beta

        h = torch.einsum("bnd,dk->bnk", g, self.patterns)  # (B,N,H)
        grad = torch.zeros_like(g)
        patterns = self.patterns  # (D,H)

        batch_offsets = torch.arange(b, device=device).view(b, 1, 1) * n
        flat_grad = grad.view(b * n, d)

        def _scatter(vertices: Tensor, a: Tensor) -> None:
            """Scatter gradient contributions to vertices."""
            contrib = -torch.einsum("dk,bmk->bmd", patterns, a)
            contrib = contrib * (1.0 / num_simplices)

            offsets = vertices.unsqueeze(0) + batch_offsets  # (B,M,K)
            contrib_exp = contrib.unsqueeze(2).expand(
                -1, -1, vertices.size(1), -1
            )
            flat_grad.index_add_(
                0,
                offsets.reshape(-1),
                contrib_exp.reshape(-1, d),
            )

        if edges.numel() > 0:
            h_e = h[:, edges[:, 0]] + h[:, edges[:, 1]]
            a_e = torch.softmax(beta * h_e, dim=-1)  # (B,M₂,H)
            _scatter(edges, a_e)

        if triangles.numel() > 0:
            h_t = (
                h[:, triangles[:, 0]]
                + h[:, triangles[:, 1]]
                + h[:, triangles[:, 2]]
            )
            a_t = torch.softmax(beta * h_t, dim=-1)  # (B,M₃,H)
            _scatter(triangles, a_t)

        # Add regularization gradient and average
        return (grad - 4 * g) / (b * n)

    def forward(self, g: Tensor) -> Tensor:
        """Forward pass returns energy for compatibility with EnergyTransformer."""
        return self.compute_energy(g)

    @property
    def cache_size(self) -> int:
        """Number of cached simplex configurations."""
        return len(self._simplices_cache)

    @property
    def device(self) -> torch.device:
        """Device of the module parameters."""
        return self.patterns.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the module parameters."""
        return self.patterns.dtype
