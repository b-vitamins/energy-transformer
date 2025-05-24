"""Simplicial Hopfield Network."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import NamedTuple

import torch

__all__ = [
    "QuerySpec",
    "SimplexGenerator",
    "UnionSimplexGenerator",
    "SimplexQuery",
    "SimplicialComplex",
    "canonical",
]

# Simple safety limits
MAX_FACET_SIZE = 50
MAX_TOTAL_SIMPLICES = 10_000_000


def canonical(simplex: Iterable[int]) -> tuple[int, ...]:
    """Convert simplex to canonical form.

    Parameters
    ----------
    simplex : Iterable[int]
        Input simplex with potentially unsorted or duplicate vertices.

    Returns
    -------
    tuple[int, ...]
        Sorted tuple of unique non-negative integers.
    """
    return tuple(
        sorted(set(v for v in simplex if isinstance(v, int) and v >= 0))
    )


def unrank(r: int, n: int, k: int) -> tuple[int, ...]:
    """Convert combinatorial rank to k-element subset.

    Parameters
    ----------
    r : int
        Combinatorial rank in lexicographic order.
    n : int
        Size of the universe.
    k : int
        Size of the subset.

    Returns
    -------
    tuple[int, ...]
        k-element subset in lexicographic order.

    Raises
    ------
    ValueError
        If rank is out of range for comb(n, k).
    """
    if not (0 <= r < math.comb(n, k)):
        raise ValueError(f"Rank {r} out of range for comb({n}, {k})")

    if k == 0:
        return ()
    if k == 1:
        return (r,)
    if k == n:
        return tuple(range(n))

    result = []
    remaining = k
    budget = r

    for i in range(n):
        if remaining == 0:
            break
        ways_without = math.comb(n - i - 1, remaining - 1)
        if budget < ways_without:
            result.append(i)
            remaining -= 1
        else:
            budget -= ways_without

    return tuple(result)


class SampleResult(NamedTuple):
    """Result of sampling operation.

    Attributes
    ----------
    data : list[frozenset[int]]
        Sampled simplices.
    complete : bool
        Whether the requested number of samples was fully satisfied.
    """

    data: list[frozenset[int]]
    complete: bool


@dataclass(frozen=True)
class SimplexGenerator:
    """Generate k-simplices from a single facet.

    Parameters
    ----------
    facet_vertices : tuple[int, ...]
        Vertices of the facet in canonical form.
    k : int
        Dimension of simplices to generate.
    """

    facet_vertices: tuple[int, ...]
    k: int

    def __post_init__(self) -> None:
        """Validate generator parameters."""
        if self.k < 0:
            raise ValueError("k must be non-negative")

        canonical_vertices = canonical(self.facet_vertices)
        if self.facet_vertices != canonical_vertices:
            raise ValueError("Facet vertices must be canonical")

    @property
    def n(self) -> int:
        """Size of the facet.

        Returns
        -------
        int
            Number of vertices in the facet.
        """
        return len(self.facet_vertices)

    @property
    def total_count(self) -> int:
        """Total number of k-simplices in this facet.

        Returns
        -------
        int
            Total count of k-simplices that can be generated.
        """
        if self.k + 1 > self.n:
            return 0
        return math.comb(self.n, self.k + 1)

    @property
    def computation_safe(self) -> bool:
        """Whether computation is within safety limits.

        Returns
        -------
        bool
            True if computation is within safety limits.
        """
        a = self.n <= MAX_FACET_SIZE
        b = self.total_count <= MAX_TOTAL_SIMPLICES
        return a and b

    def available(self) -> int:
        """k-simplices available for safe computation.

        Returns
        -------
        int
            Number of available k-simplices for safe computation.
        """
        return self.total_count if self.computation_safe else 0

    def _simplex_at_rank(self, rank: int) -> frozenset[int]:
        """Simplex at the given rank.

        Parameters
        ----------
        rank : int
            Combinatorial rank of the simplex.

        Returns
        -------
        frozenset[int]
            The simplex at the specified rank.
        """
        indices = unrank(rank, self.n, self.k + 1)
        return frozenset(self.facet_vertices[i] for i in indices)

    def sample(
        self, n: int, *, replacement: bool = False, strict: bool = True
    ) -> SampleResult:
        """Sample n k-simplices.

        Parameters
        ----------
        n : int
            Number of simplices to sample.
        replacement : bool, optional
            Whether to sample with replacement, by default False.
        strict : bool, optional
            Whether to raise errors for unsafe computation, by default True.

        Returns
        -------
        SampleResult
            Sampling result containing data and completion status.

        Raises
        ------
        ValueError
            If computation is unsafe (when strict=True) or insufficient
            simplices available (when strict=True and replacement=False).
        """
        if not self.computation_safe:
            if strict:
                raise ValueError("Computation unsafe")
            return SampleResult([], False)

        if n <= 0:
            return SampleResult([], True)

        available = self.total_count
        if available == 0:
            return SampleResult([], True)

        if replacement:
            ranks = [random.randrange(available) for _ in range(n)]
            result = [self._simplex_at_rank(r) for r in ranks]
            return SampleResult(result, True)
        else:
            actual_n = min(n, available)
            if n > available and strict:
                raise ValueError(
                    f"Requested {n} but only {available} available"
                )

            ranks = random.sample(range(available), actual_n)
            result = [self._simplex_at_rank(r) for r in ranks]
            return SampleResult(result, actual_n == n)

    def count(
        self, n: int, *, offset: int = 0, strict: bool = True
    ) -> SampleResult:
        """Get n k-simplices starting from offset.

        Parameters
        ----------
        n : int
            Number of simplices to retrieve.
        offset : int, optional
            Starting offset, by default 0.
        strict : bool, optional
            Whether to raise errors for unsafe computation, by default True.

        Returns
        -------
        SampleResult
            Result containing requested simplices and completion status.

        Raises
        ------
        ValueError
            If computation is unsafe (when strict=True) or range exceeds
            available simplices (when strict=True).
        """
        if not self.computation_safe:
            if strict:
                raise ValueError("Computation unsafe")
            return SampleResult([], False)

        if n <= 0:
            return SampleResult([], True)

        available = self.total_count
        end = min(offset + n, available)
        max(0, end - offset)

        if offset + n > available and strict:
            raise ValueError("Range exceeds available simplices")

        result = [self._simplex_at_rank(r) for r in range(offset, end)]
        return SampleResult(result, end - offset == n)

    def all(self) -> list[frozenset[int]]:
        """All k-simplices from this facet.

        Returns
        -------
        list[frozenset[int]]
            All k-simplices from this facet.

        Raises
        ------
        ValueError
            If computation is unsafe.
        """
        if not self.computation_safe:
            raise ValueError("Computation unsafe")
        return [self._simplex_at_rank(r) for r in range(self.total_count)]

    def __iter__(self) -> Iterator[frozenset[int]]:
        """Iterate over all k-simplices.

        Yields
        ------
        frozenset[int]
            Each k-simplex in the facet.

        Raises
        ------
        ValueError
            If computation is unsafe.
        """
        if not self.computation_safe:
            raise ValueError("Computation unsafe")
        for r in range(self.total_count):
            yield self._simplex_at_rank(r)

    def __len__(self) -> int:
        """k-simplices in the facet.

        Returns
        -------
        int
            Total number of k-simplices.

        Raises
        ------
        ValueError
            If computation is unsafe.
        """
        if not self.computation_safe:
            raise ValueError("Computation unsafe")
        return self.total_count

    def __bool__(self) -> bool:
        """Check if any k-simplices exist.

        Returns
        -------
        bool
            True if k-simplices exist and computation is safe.
        """
        return self.computation_safe and self.total_count > 0


class UnionSimplexGenerator:
    """Generate k-simplices from multiple facets with streaming.

    Parameters
    ----------
    facets : list[tuple[int, ...]]
        List of facets to generate simplices from.
    k : int
        Dimension of simplices to generate.
    """

    def __init__(self, facets: list[tuple[int, ...]], k: int) -> None:
        """Initialize the union generator."""
        self.facets = facets
        self.k = k
        self.generators = [SimplexGenerator(facet, k) for facet in facets]
        self._computation_safe = all(
            g.computation_safe for g in self.generators
        )
        self._upper_bound = sum(g.available() for g in self.generators)

        # Cached exact count (computed lazily)
        self._exact_count: int | None = None

    @property
    def computation_safe(self) -> bool:
        """Whether computation is within safety limits.

        Returns
        -------
        bool
            True if computation is within safety limits.
        """
        return self._computation_safe

    @property
    def dimension(self) -> int:
        """Dimension of the generated simplices.

        Returns
        -------
        int
            Dimension k of generated simplices.
        """
        return self.k

    def available(self) -> int:
        """Upper bound on available k-simplices.

        Returns
        -------
        int
            Upper bound that may overestimate due to overlaps between facets.
        """
        return self._upper_bound if self._computation_safe else 0

    def _stream_unique(self) -> Iterator[frozenset[int]]:
        """Stream unique simplices from all facets.

        Yields
        ------
        frozenset[int]
            Each unique simplex across all facets.
        """
        seen: set[frozenset[int]] = set()
        for generator in self.generators:
            for simplex in generator:
                if simplex not in seen:
                    seen.add(simplex)
                    yield simplex

    def _get_exact_count(self) -> int:
        """Exact count computed by streaming once (cached).

        Returns
        -------
        int
            Exact number of unique simplices.
        """
        if self._exact_count is None:
            count = 0
            for _ in self._stream_unique():
                count += 1
            self._exact_count = count
        return self._exact_count

    def sample(
        self, n: int, *, replacement: bool = False, strict: bool = True
    ) -> SampleResult:
        """Sample n k-simplices.

        Parameters
        ----------
        n : int
            Number of simplices to sample.
        replacement : bool, optional
            Whether to sample with replacement, by default False.
        strict : bool, optional
            Whether to raise errors for issues, by default True.

        Returns
        -------
        SampleResult
            Sampling result with data and completion status.

        Raises
        ------
        ValueError
            If computation is unsafe (when strict=True) or insufficient
            simplices available (when strict=True and replacement=False).
        """
        if not self._computation_safe:
            if strict:
                raise ValueError("Computation unsafe")
            return SampleResult([], False)

        if n <= 0:
            return SampleResult([], True)

        # For sampling, we need to materialize
        all_simplices = list(self._stream_unique())
        available = len(all_simplices)

        if available == 0:
            return SampleResult([], True)

        if replacement:
            result = [random.choice(all_simplices) for _ in range(n)]
            return SampleResult(result, True)
        else:
            actual_n = min(n, available)
            if n > available and strict:
                raise ValueError(
                    f"Requested {n} but only {available} available"
                )

            result = random.sample(all_simplices, actual_n)
            return SampleResult(result, actual_n == n)

    def count(
        self, n: int, *, offset: int = 0, strict: bool = True
    ) -> SampleResult:
        """Get n k-simplices starting from offset.

        Parameters
        ----------
        n : int
            Number of simplices to retrieve.
        offset : int, optional
            Starting offset, by default 0.
        strict : bool, optional
            Whether to raise errors for issues, by default True.

        Returns
        -------
        SampleResult
            Result with requested simplices and completion status.

        Raises
        ------
        ValueError
            If computation is unsafe (when strict=True) or insufficient
            simplices available (when strict=True).
        """
        if not self._computation_safe:
            if strict:
                raise ValueError("Computation unsafe")
            return SampleResult([], False)

        if n <= 0:
            return SampleResult([], True)

        # Stream until we have enough
        result = []
        current = 0

        for simplex in self._stream_unique():
            if current >= offset + n:
                break
            if current >= offset:
                result.append(simplex)
            current += 1

        complete = len(result) == n
        if len(result) < n and strict:
            raise ValueError("Not enough simplices available")

        return SampleResult(result, complete)

    def all(self) -> list[frozenset[int]]:
        """All unique k-simplices across facets.

        Returns
        -------
        list[frozenset[int]]
            All unique k-simplices across all facets.

        Raises
        ------
        ValueError
            If computation is unsafe.
        """
        if not self._computation_safe:
            raise ValueError("Computation unsafe")
        return list(self._stream_unique())

    def __iter__(self) -> Iterator[frozenset[int]]:
        """Iterate over unique k-simplices.

        Yields
        ------
        frozenset[int]
            Each unique k-simplex across all facets.

        Raises
        ------
        ValueError
            If computation is unsafe.
        """
        if not self._computation_safe:
            raise ValueError("Computation unsafe")
        yield from self._stream_unique()

    def __len__(self) -> int:
        """Exact number of unique k-simplices.

        Returns
        -------
        int
            Exact count of unique simplices.

        Raises
        ------
        ValueError
            If computation is unsafe.
        """
        if not self._computation_safe:
            raise ValueError("Computation unsafe")
        return self._get_exact_count()

    def __bool__(self) -> bool:
        """Check if any k-simplices exist.

        Returns
        -------
        bool
            True if k-simplices exist and computation is safe.
        """
        if not self._computation_safe:
            return False
        try:
            next(iter(self._stream_unique()))
            return True
        except StopIteration:
            return False


@dataclass(frozen=True)
class QuerySpec:
    """Specification for simplex queries.

    Parameters
    ----------
    dimensions : frozenset[int], optional
        Set of dimensions to query, by default empty.
    proportions : dict[int, float], optional
        Proportions for each dimension, by default empty.
    filters : tuple[Callable[[frozenset[int]], bool], ...], optional
        Filter predicates to apply, by default empty.
    replacement : bool, optional
        Whether to sample with replacement, by default False.
    strict : bool, optional
        Whether to use strict error handling, by default True.
    """

    dimensions: frozenset[int] = field(default_factory=frozenset)
    proportions: dict[int, float] = field(default_factory=dict)
    filters: tuple[Callable[[frozenset[int]], bool], ...] = field(
        default_factory=tuple
    )
    replacement: bool = False
    strict: bool = True

    def __post_init__(self) -> None:
        """Validate proportions sum to 1.0."""
        if self.proportions:
            total = sum(self.proportions.values())
            if not math.isclose(total, 1.0, rel_tol=1e-9):
                raise ValueError(f"Proportions must sum to 1.0, got {total}")


class SimplexQuery:
    """Fluent interface for simplex queries.

    Parameters
    ----------
    complex : SimplicialComplex
        The simplicial complex to query.
    spec : QuerySpec, optional
        Query specification, by default None.
    """

    def __init__(
        self, complex: SimplicialComplex, spec: QuerySpec | None = None
    ) -> None:
        """Initialize the query interface."""
        self._complex = complex
        self._spec = spec or QuerySpec()

    def simplices(self, k: int) -> SimplexQuery:
        """Query k-dimensional simplices.

        Parameters
        ----------
        k : int
            Dimension of simplices to query.

        Returns
        -------
        SimplexQuery
            New query instance with dimension added.
        """
        new_spec = QuerySpec(
            dimensions=self._spec.dimensions | {k},
            proportions=self._spec.proportions,
            filters=self._spec.filters,
            replacement=self._spec.replacement,
            strict=self._spec.strict,
        )
        return SimplexQuery(self._complex, new_spec)

    def dimensions(self, ks: Iterable[int]) -> SimplexQuery:
        """Query multiple dimensions.

        Parameters
        ----------
        ks : Iterable[int]
            Dimensions to query.

        Returns
        -------
        SimplexQuery
            New query instance with dimensions added.
        """
        new_dims = self._spec.dimensions | frozenset(ks)
        new_spec = QuerySpec(
            dimensions=new_dims,
            proportions=self._spec.proportions,
            filters=self._spec.filters,
            replacement=self._spec.replacement,
            strict=self._spec.strict,
        )
        return SimplexQuery(self._complex, new_spec)

    def where(
        self, predicate: Callable[[frozenset[int]], bool]
    ) -> SimplexQuery:
        """Add filter predicate.

        Parameters
        ----------
        predicate : Callable[[frozenset[int]], bool]
            Filter function to apply to simplices.

        Returns
        -------
        SimplexQuery
            New query instance with filter added.
        """
        new_spec = QuerySpec(
            dimensions=self._spec.dimensions,
            proportions=self._spec.proportions,
            filters=self._spec.filters + (predicate,),
            replacement=self._spec.replacement,
            strict=self._spec.strict,
        )
        return SimplexQuery(self._complex, new_spec)

    def containing(self, vertex: int) -> SimplexQuery:
        """Filter to simplices containing vertex.

        Parameters
        ----------
        vertex : int
            Vertex that simplices must contain.

        Returns
        -------
        SimplexQuery
            New query instance with containment filter.
        """
        return self.where(lambda s: vertex in s)

    def with_replacement(self, replacement: bool = True) -> SimplexQuery:
        """Set sampling with replacement.

        Parameters
        ----------
        replacement : bool, optional
            Whether to sample with replacement, by default True.

        Returns
        -------
        SimplexQuery
            New query instance with replacement setting.
        """
        new_spec = QuerySpec(
            dimensions=self._spec.dimensions,
            proportions=self._spec.proportions,
            filters=self._spec.filters,
            replacement=replacement,
            strict=self._spec.strict,
        )
        return SimplexQuery(self._complex, new_spec)

    def strict(self, strict: bool = True) -> SimplexQuery:
        """Set strict error handling.

        Parameters
        ----------
        strict : bool, optional
            Whether to use strict error handling, by default True.

        Returns
        -------
        SimplexQuery
            New query instance with strict setting.
        """
        new_spec = QuerySpec(
            dimensions=self._spec.dimensions,
            proportions=self._spec.proportions,
            filters=self._spec.filters,
            replacement=self._spec.replacement,
            strict=strict,
        )
        return SimplexQuery(self._complex, new_spec)

    def sample(self, n: int) -> dict[int, list[frozenset[int]]]:
        """Sample n simplices per dimension.

        Parameters
        ----------
        n : int
            Number of simplices to sample per dimension.

        Returns
        -------
        dict[int, list[frozenset[int]]]
            Mapping from dimension to sampled simplices.
        """
        result: dict[int, list[frozenset[int]]] = {}

        if not self._spec.dimensions:
            return result

        # Simple equal distribution
        per_dim = n // len(self._spec.dimensions)
        remainder = n % len(self._spec.dimensions)

        for i, k in enumerate(sorted(self._spec.dimensions)):
            count = per_dim + (1 if i < remainder else 0)
            if count > 0:
                generator = self._complex._get_generator_for_dimension(k)
                sample_result = generator.sample(
                    count,
                    replacement=self._spec.replacement,
                    strict=self._spec.strict,
                )
                filtered = self._apply_filters(sample_result.data)
                result[k] = filtered

        return result

    def collect(self) -> dict[int, list[frozenset[int]]]:
        """Collect all matching simplices.

        Returns
        -------
        dict[int, list[frozenset[int]]]
            Mapping from dimension to all matching simplices.
        """
        result: dict[int, list[frozenset[int]]] = {}

        for k in self._spec.dimensions:
            generator = self._complex._get_generator_for_dimension(k)
            try:
                all_simplices = generator.all()
                filtered = self._apply_filters(all_simplices)
                result[k] = filtered
            except ValueError:
                if self._spec.strict:
                    raise
                result[k] = []

        return result

    def count(self) -> dict[int, int]:
        """Count matching simplices.

        Returns
        -------
        dict[int, int]
            Mapping from dimension to count of matching simplices.
            Returns -1 for uncountable dimensions.
        """
        result: dict[int, int] = {}

        for k in self._spec.dimensions:
            generator = self._complex._get_generator_for_dimension(k)

            if not self._spec.filters:
                result[k] = generator.available()
            else:
                try:
                    all_simplices = generator.all()
                    filtered = self._apply_filters(all_simplices)
                    result[k] = len(filtered)
                except ValueError:
                    result[k] = -1  # Uncountable

        return result

    def _apply_filters(
        self, simplices: list[frozenset[int]]
    ) -> list[frozenset[int]]:
        """Apply all filters to simplices.

        Parameters
        ----------
        simplices : list[frozenset[int]]
            Input simplices to filter.

        Returns
        -------
        list[frozenset[int]]
            Filtered simplices.
        """
        result = simplices
        for predicate in self._spec.filters:
            result = [s for s in result if predicate(s)]
        return result


class SimplicialComplex:
    """Simplicial complex with fluent query interface.

    Parameters
    ----------
    facets : Iterable[Iterable[int]], optional
        Maximal simplices defining the complex, by default empty.
    """

    def __init__(self, facets: Iterable[Iterable[int]] = ()) -> None:
        """Initialize with facets."""
        # Canonicalize and deduplicate facets
        canonical_facets = []
        seen = set()

        for facet in facets:
            canonical_facet = canonical(facet)
            if canonical_facet and frozenset(canonical_facet) not in seen:
                seen.add(frozenset(canonical_facet))
                canonical_facets.append(canonical_facet)

        self._facets = canonical_facets
        self._dimension = max((len(f) - 1 for f in self._facets), default=-1)

    @property
    def facets(self) -> list[tuple[int, ...]]:
        """Facets defining the simplicial complex.

        Returns
        -------
        list[tuple[int, ...]]
            Copy of the facets list.
        """
        return self._facets.copy()

    @property
    def dimension(self) -> int:
        """Maximum dimension of the complex.

        Returns
        -------
        int
            Highest dimension of any simplex in the complex.
        """
        return self._dimension

    def simplices(self, k: int) -> SimplexQuery:
        """Start query for k-simplices.

        Parameters
        ----------
        k : int
            Dimension of simplices to query.

        Returns
        -------
        SimplexQuery
            Query interface for k-simplices.
        """
        return SimplexQuery(self).simplices(k)

    def dimensions(self, ks: Iterable[int]) -> SimplexQuery:
        """Start query for multiple dimensions.

        Parameters
        ----------
        ks : Iterable[int]
            Dimensions to query.

        Returns
        -------
        SimplexQuery
            Query interface for multiple dimensions.
        """
        return SimplexQuery(self).dimensions(ks)

    def all_dimensions(self) -> SimplexQuery:
        """Query all dimensions.

        Returns
        -------
        SimplexQuery
            Query interface for all dimensions in the complex.
        """
        return self.dimensions(range(self.dimension + 1))

    def _get_generator_for_dimension(
        self, k: int
    ) -> SimplexGenerator | UnionSimplexGenerator:
        """Generate k-simplices from compatible facets.

        Parameters
        ----------
        k : int
            Dimension of simplices to generate.

        Returns
        -------
        SimplexGenerator | UnionSimplexGenerator
            Generator for k-simplices from all compatible facets.
        """
        if k < 0:
            return SimplexGenerator((), k)

        compatible_facets = [f for f in self._facets if len(f) >= k + 1]

        if not compatible_facets:
            return SimplexGenerator((), k)
        elif len(compatible_facets) == 1:
            return SimplexGenerator(compatible_facets[0], k)
        else:
            return UnionSimplexGenerator(compatible_facets, k)

    def __contains__(self, simplex: Iterable[int]) -> bool:
        """Check if simplex is in the complex.

        Parameters
        ----------
        simplex : Iterable[int]
            Simplex to check for membership.

        Returns
        -------
        bool
            True if simplex is contained in the complex.
        """
        query_simplex = frozenset(canonical(simplex))
        if not query_simplex:
            return False

        # Check if it's a subset of any facet
        for facet in self._facets:
            if query_simplex.issubset(facet):
                return True

        return False

    def __repr__(self) -> str:
        """Return string representation.

        Returns
        -------
        str
            String representation of the complex.
        """
        if not self._facets:
            return "SimplicialComplex(∅)"
        return (
            f"SimplicialComplex(dim={self.dimension}, "
            f"facets={len(self._facets)})"
        )


@lru_cache(maxsize=128)
def membership(
    κ: tuple[tuple[int, ...], ...],
    n: int,
    device_str: str,
    dtype_str: str,
) -> torch.Tensor:
    """Build sparse membership matrix for simplicial complex.

    The membership matrix M has M[v,σ] = 1 if vertex v ∈ simplex σ, else 0.
    This enables computing all simplex-wise dot products via a single sparse
    matrix multiply: Ξ_σ^μ · S_σ = (Ξ * S) @ M.

    Parameters
    ----------
    κ : tuple[tuple[int, ...], ...]
        Simplicial complex as tuple of simplices (hashable for caching).
        Each simplex is a tuple of vertex indices in [0, n).
    n : int
        Number of vertices in the complex.
    device_str : str
        String representation of target device.
    dtype_str : str
        String representation of target data type.

    Returns
    -------
    torch.Tensor
        Sparse COO tensor representing the membership matrix.
        Shape: (n, |κ|). For CUDA + fp16/bf16, uses fp32 internally.

    Raises
    ------
    ValueError
        If any vertex index is outside [0, n) or if κ is empty.
    """
    # Reconstruct device and dtype from strings for cross-process caching
    device = torch.device(device_str)
    # Handle torch.float32 -> float32
    dtype = getattr(torch, dtype_str.split(".")[-1])

    if not κ:
        # Handle empty complex gracefully
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros(0, device=device, dtype=dtype),
            (n, 0),
            device=device,
            dtype=dtype,
        )

    rows, cols = [], []
    for col, σ in enumerate(κ):
        for v in σ:
            if not (0 <= v < n):
                raise ValueError(
                    f"Vertex index {v} outside valid range [0, {n})"
                )
            rows.append(v)
            cols.append(col)

    if not rows:  # All simplices are empty
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros(0, device=device, dtype=dtype),
            (n, len(κ)),
            device=device,
            dtype=dtype,
        )

    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)

    # Use fp32 for sparse operations on CUDA with half precision
    sparse_dtype = (
        torch.float32
        if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        else dtype
    )

    values = torch.ones(len(rows), device=device, dtype=sparse_dtype)

    return torch.sparse_coo_tensor(
        indices, values, (n, len(κ)), device=device, dtype=sparse_dtype
    )


def _get_membership(
    κ: Sequence[Sequence[int]], n: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Call cached membership with string keys.

    Parameters
    ----------
    κ : Sequence[Sequence[int]]
        Simplicial complex structure.
    n : int
        Number of vertices.
    device : torch.device
        Target device.
    dtype : torch.dtype
        Target data type.

    Returns
    -------
    torch.Tensor
        Membership matrix.
    """
    κ_hashable = tuple(tuple(σ) for σ in κ)
    return membership(κ_hashable, n, str(device), str(dtype))


def lse(
    β: float, ξ: torch.Tensor, g: torch.Tensor, κ: Sequence[Sequence[int]]
) -> torch.Tensor:
    """Compute log-sum-exp term using sparse operations.

    Implements: (1/β) * log(Σ_μ Σ_σ exp(β * Ξ_σ^μ · S_σ))
    where Ξ_σ^μ · S_σ = Σ_{v∈σ} ξ[μ,v] * g[v].

    Parameters
    ----------
    β : float
        Inverse temperature parameter (β = 1/T). Must be positive.
    ξ : torch.Tensor, shape (P, N)
        Pattern matrix with P patterns and N vertices.
    g : torch.Tensor, shape (N,)
        State vector on the vertices.
    κ : Sequence[Sequence[int]]
        Simplicial complex structure.

    Returns
    -------
    torch.Tensor
        Scalar log-sum-exp term.

    Raises
    ------
    ValueError
        If β ≤ 0 or tensor shapes are incompatible.
    """
    if β <= 0:
        raise ValueError(f"β must be positive, got {β}")

    if ξ.ndim != 2 or g.ndim != 1:
        raise ValueError(f"Expected ξ.ndim=2, g.ndim=1,got {ξ.ndim}, {g.ndim}")

    if ξ.shape[1] != g.shape[0]:
        raise ValueError(
            f"Dimension mismatch: ξ has {ξ.shape[1]} features, "
            f"g has {g.shape[0]}"
        )

    device, dtype = ξ.device, ξ.dtype

    if g.device != device:
        raise ValueError("ξ and g must be on the same device")

    μ = _get_membership(κ, g.shape[0], device, dtype)

    # Mixed precision handling for numerical stability
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        ξ_compute, g_compute = ξ.float(), g.float()
    else:
        ξ_compute, g_compute = ξ, g

    # Core computation: Y[μ,v] = ξ[μ,v] * g[v]
    y = ξ_compute * g_compute  # (P, N)

    # Sparse matrix multiply: dot[μ,σ] = Σ_{v∈σ} Y[μ,v] = Ξ_σ^μ · S_σ
    dot = torch.sparse.mm(μ.t(), y.t()).t()  # (P, |κ|)

    # Convert back to original precision if needed
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        dot = dot.to(dtype)

    # Numerically stable log-sum-exp over both dimensions
    return (1 / β) * torch.logsumexp(dot * β, dim=(0, 1))


def energy(
    β: float, ξ: torch.Tensor, g: torch.Tensor, κ: Sequence[Sequence[int]]
) -> torch.Tensor:
    """Compute the Simplicial Hopfield energy function.

    Implements: E = -lse(β, ξ, g, κ) + ½ ||g||²

    This combines the pattern-matching term (LSE) with quadratic
    regularization to prevent unbounded growth of the state vector.

    Parameters
    ----------
    β : float
        Inverse temperature parameter (β = 1/T).
    ξ : torch.Tensor, shape (P, N)
        Pattern matrix with stored memories.
    g : torch.Tensor, shape (N,)
        Current state vector.
    κ : Sequence[Sequence[int]]
        Simplicial complex defining higher-order interactions.

    Returns
    -------
    torch.Tensor
        Scalar energy value.
    """
    lse_term = lse(β, ξ, g, κ)
    regularization = 0.5 * torch.dot(g, g)
    return -lse_term + regularization


def update(
    β: float,
    ξ: torch.Tensor,
    g: torch.Tensor,
    κ: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Compute the Simplicial Hopfield update rule.

    Implements: g⁽ᵗ⁾ = softmax(1/β * Σ_σ Ξ_σᵀ g_σ⁽ᵗ⁻¹⁾) Ξ

    This computes pattern-wise similarities via the simplicial structure,
    applies temperature-scaled softmax attention, and returns the weighted
    combination of stored patterns.

    Complexity: O(P * nnz(κ)) with two GPU kernel launches.

    Parameters
    ----------
    β : float
        Inverse temperature parameter (β = 1/T).
    ξ : torch.Tensor, shape (P, N)
        Pattern matrix with P stored patterns.
    g : torch.Tensor, shape (N,)
        State vector at previous time step t-1.
    κ : Sequence[Sequence[int]]
        Simplicial complex structure.

    Returns
    -------
    torch.Tensor, shape (N,)
        Updated state vector at time step t. Returns real-valued vector;
        use .sign() for ±1 spins in binary-spin applications.

    Raises
    ------
    ValueError
        If inputs have incompatible shapes or are on different devices.
    """
    if β <= 0:
        raise ValueError(f"β must be positive, got {β}")

    if ξ.ndim != 2 or g.ndim != 1:
        raise ValueError(
            f"Expected ξ.ndim=2, g_prev.ndim=1, got {ξ.ndim}, {g.ndim}"
        )

    if ξ.shape[1] != g.shape[0]:
        raise ValueError(f"Shape mismatch: ξ ({ξ.shape}) vs g ({g.shape})")

    device, dtype = ξ.device, ξ.dtype

    if g.device != device:
        raise ValueError("ξ and g_prev must be on the same device")

    μ = _get_membership(κ, g.shape[0], device, dtype)

    # Mixed precision handling for numerical stability
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        ξ_compute, g_compute = ξ.float(), g.float()
    else:
        ξ_compute, g_compute = ξ, g

    # Element-wise multiply: Y[μ,v] = ξ[μ,v] * g_prev[v]
    y = ξ_compute * g_compute  # (P, N)

    # Compute simplex-wise similarities: dot[μ,σ] = Σ_{v∈σ} Y[μ,v]
    dot = torch.sparse.mm(μ.t(), y.t()).t()  # (P, |κ|)

    # Convert back to original precision if needed
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        dot = dot.to(dtype)

    # Sum over simplices to get pattern-wise similarities
    similarity = dot.sum(dim=1)  # (P,)

    # Temperature-scaled softmax attention over patterns
    # For extreme β values with fp16, compute in float32 for headroom
    if (
        device.type == "cuda"
        and dtype in (torch.float16, torch.bfloat16)
        and β > 100
    ):
        α = torch.softmax(similarity.float() / β, dim=0).to(dtype)
    else:
        α = torch.softmax(similarity / β, dim=0)  # (P,)

    # Weighted combination of patterns
    return α @ ξ  # (N,)
