"""Simplicial Hopfield Network."""

from __future__ import annotations

import math
import random
import warnings
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import NamedTuple, cast

__all__ = [
    "QuerySpec",
    "SimplexGenerator",
    "SimplexQuery",
    "SimplicialComplex",
    "unrank",
]

# Safety limits for combinatorial operations
MAX_N = 100_000_000_000_000  # 100 trillion  # noqa: N806
MAX_K = 100  # noqa: N806


class SampleResult(NamedTuple):
    """Result of sampling operation with completeness indicator."""

    data: list[frozenset[int]]
    complete: bool


@lru_cache(maxsize=128)
def _build_pascal_cache(n: int, max_k: int) -> list[list[int]]:
    """Build Pascal triangle cache for unrank operations."""
    cache = []
    for candidate in range(n):
        row = []
        for j in range(
            max_k + 1
        ):  # Fixed: need max_k + 1 entries (0 to max_k inclusive)
            numerator = n - candidate - 1
            if numerator >= j >= 0:
                row.append(math.comb(numerator, j))
            else:
                row.append(0)
        cache.append(row)
    return cache


def unrank(
    r: int, n: int, k: int, *, cache: list[list[int]] | None = None
) -> tuple[int, ...]:
    """
    Convert combinatorial rank to k-element subset in lexicographic order.

    Implements the combinatorial number system (Lehmer code) for efficient
    rank-to-combination conversion. Uses mathematical unranking to avoid
    generating all combinations.

    Parameters
    ----------
    r : int
        The rank of the combination. Must satisfy 0 <= r < comb(n, k).
        Rank 0 corresponds to the lexicographically first combination.
    n : int
        The size of the universal set {0, 1, ..., n-1}. Must be
        non-negative. Limited to n <= 100 trillion for safety against
        overflow.
    k : int
        The size of the subset to choose. Must satisfy 0 <= k <= n.
        Limited to k <= 100 for safety against overflow.
    cache : list of list of int, optional
        Pre-computed Pascal triangle for efficiency. If provided,
        cache[candidate][j] should contain comb(n-candidate-1, j).

    Returns
    -------
    tuple of int
        A tuple of k distinct integers in ascending order, representing
        the combination of rank r. All elements are in range [0, n-1].

    Raises
    ------
    ValueError
        If r is out of the valid range [0, comb(n, k)), or if n > 100
        trillion, or if k > 100, or if comb(n, k) > 10^12.

    Examples
    --------
    >>> unrank(1, 4, 2)  # Second 2-element subset of {0,1,2,3}
    (0, 2)

    Notes
    -----
    Time complexity: O(k) without cache, O(1) lookups with cache
    Space complexity: O(k)
    The function is optimized for repeated calls with the same n, k values
    but different ranks, making it ideal for systematic simplex generation.
    """
    # Early validation for negative inputs
    if r < 0:
        raise ValueError(f"Rank r={r} must be non-negative")
    if n < 0:
        raise ValueError(f"Universe size n={n} must be non-negative")
    if k < 0:
        raise ValueError(f"Subset size k={k} must be non-negative")

    # Guard against catastrophic combinations
    if n > MAX_N or k > MAX_K:
        raise ValueError(
            f"Combination comb({n}, {k}) too large for safe computation. "
            f"Limits: n <= {MAX_N:,}, k <= {MAX_K}"
        )

    total_combinations = math.comb(n, k)
    if total_combinations > 10**12:
        raise ValueError(
            f"Combination comb({n}, {k}) = {total_combinations:,} exceeds "
            f"safe limit (10^12)"
        )

    if not (0 <= r < total_combinations):
        raise ValueError(f"Rank r={r} out of range for comb(n={n}, k={k})")

    chosen: list[int] = []
    budget = r
    seats_remaining = k

    for candidate in range(n):
        if seats_remaining == 0:
            break

        # Use cache if available, otherwise compute directly
        if (
            cache is not None
            and candidate < len(cache)
            and seats_remaining - 1 < len(cache[candidate])
        ):
            ways_without = cache[candidate][seats_remaining - 1]
        else:
            ways_without = math.comb(n - candidate - 1, seats_remaining - 1)

        if budget < ways_without:
            chosen.append(candidate)
            seats_remaining -= 1
        else:
            budget -= ways_without

    return tuple(chosen)


@dataclass(slots=True, frozen=True)
class SimplexGenerator:
    """
    Lazy generator for :math:`k`-simplices from a single facet.

    Provides memory-efficient access to :math:`k`-dimensional faces of a
    simplex without materializing the entire face lattice. Supports sampling,
    counting, and iteration with built-in safety limits.

    Parameters
    ----------
    facet_vertices : tuple of int
        Sorted tuple of unique vertex indices defining the facet.
    k : int
        Dimension of simplices to generate (:math:`k`-simplex has
        :math:`k+1` vertices). Must be non-negative.

    Attributes
    ----------
    total_count : int
        Total number of :math:`k`-simplices available from this facet.
    n : int
        Number of vertices in the facet.
    dimension : int
        Alias for k (dimension of simplices being generated).
    available : callable
        Method returning number of :math:`k`-simplices available for
        generation.
    """

    facet_vertices: tuple[int, ...]
    k: int

    # Computed fields
    n: int = field(init=False)
    total_count: int = field(init=False)
    _computation_safe: bool = field(init=False)
    _cache: list[list[int]] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Initialize computed fields and validate inputs."""
        # Input validation
        if self.k < 0:
            raise ValueError("k must be non-negative")

        if self.facet_vertices != tuple(sorted(set(self.facet_vertices))):
            raise ValueError(
                "facet_vertices must be a sorted tuple of unique integers"
            )

        # Set computed fields (using object.__setattr__ due to frozen=True)
        object.__setattr__(self, "n", len(self.facet_vertices))

        # Align safety check with unrank function for consistency
        if self.n > MAX_N or self.k + 1 > MAX_K:
            object.__setattr__(self, "total_count", 0)
            object.__setattr__(self, "_computation_safe", False)
        elif self.k + 1 > self.n:
            object.__setattr__(self, "total_count", 0)
            object.__setattr__(self, "_computation_safe", True)
        else:
            try:
                total = math.comb(self.n, self.k + 1)
                # Use same limit as unrank for consistency
                if total > 10**12:
                    object.__setattr__(self, "total_count", 0)
                    object.__setattr__(self, "_computation_safe", False)
                else:
                    object.__setattr__(self, "total_count", total)
                    object.__setattr__(self, "_computation_safe", True)

                    # Build Pascal cache with correct max_k parameter
                    if (
                        self._computation_safe
                        and self.n <= 1000
                        and self.k >= 0
                    ):
                        cache = _build_pascal_cache(self.n, self.k)
                        object.__setattr__(self, "_cache", cache)

            except (ValueError, OverflowError):
                object.__setattr__(self, "total_count", 0)
                object.__setattr__(self, "_computation_safe", False)

    @property
    def dimension(self) -> int:
        """Get dimension of the simplices being generated (alias for k)."""
        return self.k

    def _simplex(self, rank: int) -> frozenset[int]:
        """Convert rank to simplex using cached Pascal table when available."""
        indices = unrank(rank, self.n, self.k + 1, cache=self._cache)
        return frozenset(self.facet_vertices[i] for i in indices)

    def available(self) -> int:
        """
        Return number of :math:`k`-simplices available for generation.

        Returns
        -------
        int
            Total count of :math:`k`-simplices that can be generated from
            this facet. Returns 0 if computation is unsafe due to size
            limits.
        """
        return self.total_count if self._computation_safe else 0

    def count(self, n: int, *, strict: bool = True) -> SampleResult:
        """
        Generate first n :math:`k`-simplices in lexicographic order.

        Parameters
        ----------
        n : int
            Number of :math:`k`-simplices to generate. Must be non-negative.
        strict : bool, default True
            If True, raises error when fewer than n simplices are available.
            If False, returns available simplices with warning.

        Returns
        -------
        SampleResult
            Named tuple with 'data' (list of :math:`k`-simplices as
            frozensets) and 'complete' (bool indicating if all requested
            simplices were returned).

        Raises
        ------
        ValueError
            If strict=True and insufficient simplices available, or if
            facet size makes computation unsafe.
        RuntimeWarning
            If strict=False and fewer than n simplices available.
        """
        if not self._computation_safe:
            if strict:
                raise ValueError(
                    f"Cannot safely generate {self.k}-simplices from "
                    f"facet size {self.n}"
                )
            return SampleResult([], False)

        if n <= 0:
            return SampleResult([], True)

        available = self.total_count
        complete = n <= available
        actual_n = min(n, available)

        if n > available:
            if strict:
                raise ValueError(
                    f"Requested {n} simplices but only {available} available"
                )
            warnings.warn(
                f"Requested {n} simplices but only {available} available, "
                f"returning {actual_n}",
                RuntimeWarning,
                stacklevel=2,
            )

        result = [self._simplex(rank) for rank in range(actual_n)]
        return SampleResult(result, complete)

    def sample(
        self, n: int, *, replacement: bool = False, strict: bool = True
    ) -> SampleResult:
        """
        Generate random sample of n :math:`k`-simplices.

        Parameters
        ----------
        n : int
            Number of :math:`k`-simplices to sample. Must be non-negative.
        replacement : bool, default False
            If True, allows sampling the same simplex multiple times.
            If False, each simplex appears at most once in result.
            With replacement=True, the 'complete' field is always True.
        strict : bool, default True
            If True, raises error when fewer than n unique simplices
            available for sampling without replacement.

        Returns
        -------
        SampleResult
            Named tuple with 'data' (list of randomly sampled
            :math:`k`-simplices) and 'complete' (bool indicating if all
            requested simplices were returned).

        Raises
        ------
        ValueError
            If strict=True and insufficient unique simplices for sampling
            without replacement, or if facet size makes computation unsafe.
        RuntimeWarning
            If strict=False and fewer than n unique simplices available
            for sampling without replacement.
        """
        if not self._computation_safe:
            if strict:
                raise ValueError(
                    f"Cannot safely sample {self.k}-simplices from "
                    f"facet size {self.n}"
                )
            return SampleResult([], False)

        if n <= 0:
            return SampleResult([], True)

        available = self.total_count
        if available == 0:
            return SampleResult([], True)

        if replacement:
            # Sample with replacement - can exceed available count
            # With replacement=True, complete is always True since we can
            # always generate the requested number of samples
            ranks = [random.randrange(available) for _ in range(n)]
            result = [self._simplex(rank) for rank in ranks]
            return SampleResult(result, True)
        else:
            # Sample without replacement - limited by available count
            complete = n <= available
            actual_n = min(n, available)

            if n > available:
                if strict:
                    raise ValueError(
                        f"Requested {n} unique samples but only "
                        f"{available} available"
                    )
                warnings.warn(
                    f"Requested {n} unique samples but only "
                    f"{available} available, returning {actual_n}",
                    RuntimeWarning,
                    stacklevel=2,
                )

            # Choose sampling strategy based on size and density
            ranks = self._choose_sampling_strategy(actual_n, available)

            result = [self._simplex(rank) for rank in ranks]
            return SampleResult(result, complete)

    @staticmethod
    def _choose_sampling_strategy(k: int, total: int) -> list[int]:
        """
        Choose optimal sampling strategy with better memory management.

        Parameters
        ----------
        k : int
            Number of samples to draw
        total : int
            Total population size

        Returns
        -------
        list of int
            List of k unique random indices from range(total)
        """
        # Fixed: Handle zero total early
        if total == 0:
            return []

        # Ensure k doesn't exceed total
        k = min(k, total)

        if k == 0:
            return []

        density = k / total

        # For very dense sampling, use shuffled range to maintain uniformity
        if density >= 0.9:
            result = list(range(total))
            random.shuffle(result)
            return result[:k]

        if k < 1000:
            # For small samples, standard library is fine
            return random.sample(range(total), k)
        elif total > 2e8 and density > 0.05:
            # Warn about potential performance issues
            warnings.warn(
                f"Large dense sampling (k={k:,}, total={total:,}) may "
                f"take several seconds",
                RuntimeWarning,
                stacklevel=3,
            )

        if density > 0.5 and total > 1e8:
            # Avoid memory explosion for very dense sampling
            return SimplexGenerator._reservoir_sample(k, total)
        elif density > 0.2:
            # Dense sampling: use numpy choice (O(k) memory)
            try:
                import numpy as np

                rng = np.random.Generator(np.random.PCG64())
                result_array = rng.choice(total, k, replace=False)
                return cast(list[int], result_array.tolist())
            except (ImportError, ValueError):
                # Fallback to reservoir sampling if numpy unavailable
                # or invalid choice
                return SimplexGenerator._reservoir_sample(k, total)
        elif density < 0.01 and total > 10**6:
            # Sparse sampling from large range: use reservoir sampling
            return SimplexGenerator._reservoir_sample(k, total)
        else:
            # Default case: standard library random.sample
            return random.sample(range(total), k)

    @staticmethod
    def _reservoir_sample(k: int, total: int) -> list[int]:
        """
        Use reservoir sampling algorithm for large ranges.

        Uses Algorithm R to select k items from range(total) using O(k)
        memory. Optimal for sparse sampling (k << total) from very large
        populations.

        Parameters
        ----------
        k : int
            Number of samples to draw (reservoir size)
        total : int
            Total population size. For very large values (>10^8), this may
            take several seconds as Algorithm R must iterate through all
            items.

        Returns
        -------
        list of int
            List of k unique random indices from range(total)

        Notes
        -----
        Time complexity: O(total) but with very light operations after k
        items Space complexity: O(k)

        References
        ----------
        Algorithm R from Knuth's "The Art of Computer Programming,
        Volume 2"
        """
        if k >= total:
            return list(range(total))

        # Lower warning threshold - even 100M takes noticeable time
        if total > 1e8:
            warnings.warn(
                f"Reservoir sampling with total={total:,} may take "
                f"several seconds",
                RuntimeWarning,
                stacklevel=3,
            )

        # Initialize reservoir with first k items
        reservoir = list(range(k))

        # Process remaining items
        for i in range(k, total):
            # Random index from 0 to i (inclusive)
            j = random.randrange(i + 1)

            # If j < k, replace reservoir[j] with item i
            if j < k:
                reservoir[j] = i

        return reservoir

    def all(self, *, max_count: int = 1000000) -> list[frozenset[int]]:
        """
        Generate all :math:`k`-simplices from the facet.

        Parameters
        ----------
        max_count : int, default 1000000
            Safety limit on total simplices to prevent memory overflow.
            Increase with caution for large facets.

        Returns
        -------
        list of frozenset of int
            Complete list of all :math:`k`-simplices from this facet.

        Raises
        ------
        ValueError
            If total simplices exceed max_count, or if facet size
            makes computation unsafe.

        Examples
        --------
        >>> complex_obj = SimplicialComplex([[0, 1, 2, 3]])
        >>> all_triangles = complex_obj.simplices(k=2).all()
        >>> len(all_triangles)
        4
        """
        if not self._computation_safe:
            raise ValueError(
                f"Cannot safely generate all {self.k}-simplices from "
                f"facet size {self.n}"
            )

        if self.total_count > max_count:
            raise ValueError(
                f"Too many {self.k}-simplices: {self.total_count:,} > "
                f"{max_count:,}. Use count() or sample() instead, or "
                f"increase max_count."
            )

        result = self.count(self.total_count, strict=True)
        return result.data

    def as_array(self, *, dtype=None, max_count: int = 1000000):  # type: ignore[no-untyped-def]
        """
        Convert all k-simplices to a NumPy array for scientific computing.

        Each row represents one :math:`k`-simplex with (k+1) vertex indices.
        Convenient for downstream NumPy/PyTorch workflows.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            Data type for the array. Defaults to numpy's default int type.
        max_count : int, default 1000000
            Safety limit on total simplices to prevent memory overflow.

        Returns
        -------
        numpy.ndarray
            2D array of shape (total_count, k+1) where each row contains
            the sorted vertex indices of one :math:`k`-simplex.

        Raises
        ------
        ImportError
            If NumPy is not available.
        ValueError
            If total simplices exceed max_count, or if facet size
            makes computation unsafe.

        Notes
        -----
        Performance: For >1M simplices, prefer `.sample()` or your own
        vectorized routine, as this method uses O(total_count * (k+1))
        Python calls and may take several seconds.

        Examples
        --------
        >>> gen = SimplexGenerator((0, 1, 2, 3), k=1)  # edges from tetrahedron
        >>> edges = gen.as_array()
        >>> edges.shape
        (6, 2)
        >>> edges[0]  # first edge
        array([0, 1])
        """
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError("NumPy is required for as_array() method") from e

        if not self._computation_safe:
            raise ValueError(
                f"Cannot safely generate array of {self.k}-simplices from "
                f"facet size {self.n}"
            )

        if self.total_count > max_count:
            raise ValueError(
                f"Too many {self.k}-simplices: {self.total_count:,} > "
                f"{max_count:,}. Use count() or sample() instead, or "
                f"increase max_count."
            )

        if dtype is None:
            dtype = int

        if self.total_count == 0:
            return np.empty((0, self.k + 1), dtype=dtype)

        # Memory-efficient approach: don't materialize all combinations
        result = np.empty((self.total_count, self.k + 1), dtype=dtype)
        facet_array = np.array(self.facet_vertices)

        # Use the existing unrank-based approach to avoid memory explosion
        for i in range(self.total_count):
            indices = unrank(i, self.n, self.k + 1, cache=self._cache)
            result[i] = facet_array[list(indices)]

        return result

    def __iter__(self) -> Iterator[frozenset[int]]:
        """
        Iterate through all :math:`k`-simplices.

        Note: If computation is unsafe due to facet size, this yields
        nothing silently, while len() raises ValueError. Use available()
        to check if simplices can be safely generated.

        Yields
        ------
        frozenset of int
            Each :math:`k`-simplex as frozenset of vertex indices, in
            lexicographic order.

        Examples
        --------
        >>> complex_obj = SimplicialComplex([[0, 1, 2]])
        >>> edges = list(complex_obj.simplices(k=1))
        >>> len(edges)
        3
        """
        if self._computation_safe:
            # Minor optimization: avoid attribute lookup in loop
            simplex_func = self._simplex
            yield from (simplex_func(rank) for rank in range(self.total_count))

    def __len__(self) -> int:
        """
        Return total number of available :math:`k`-simplices.

        Raises
        ------
        ValueError
            If facet size makes computation unsafe, indicating the length
            is undefined due to safety limits.
        """
        if not self._computation_safe:
            raise ValueError(
                f"Facet too large for safe computation (n={self.n}, "
                f"k={self.k}); len() undefined"
            )
        return self.total_count


@dataclass(frozen=True)
class QuerySpec:
    """Immutable specification for simplex generation."""

    dimensions: frozenset[int] = field(default_factory=frozenset)
    proportions: dict[int, float] = field(default_factory=dict)
    filters: tuple[Callable[[frozenset[int]], bool], ...] = field(
        default_factory=tuple
    )
    replacement: bool = False
    strict: bool = False

    def with_dimension(self, k: int) -> QuerySpec:
        """Add a dimension to the query."""
        return QuerySpec(
            dimensions=self.dimensions | {k},
            proportions=self.proportions,
            filters=self.filters,
            replacement=self.replacement,
            strict=self.strict,
        )

    def with_proportions(self, props: dict[int, float]) -> QuerySpec:
        """Set proportions for mixed sampling."""
        return QuerySpec(
            dimensions=self.dimensions | frozenset(props.keys()),
            proportions=props,
            filters=self.filters,
            replacement=self.replacement,
            strict=self.strict,
        )

    def with_filter(
        self, predicate: Callable[[frozenset[int]], bool]
    ) -> QuerySpec:
        """Add a filter predicate."""
        return QuerySpec(
            dimensions=self.dimensions,
            proportions=self.proportions,
            filters=self.filters + (predicate,),
            replacement=self.replacement,
            strict=self.strict,
        )

    def with_replacement(self, replacement: bool = True) -> QuerySpec:
        """Set replacement strategy."""
        return QuerySpec(
            dimensions=self.dimensions,
            proportions=self.proportions,
            filters=self.filters,
            replacement=replacement,
            strict=self.strict,
        )

    def with_strict(self, strict: bool = True) -> QuerySpec:
        """Set strict mode."""
        return QuerySpec(
            dimensions=self.dimensions,
            proportions=self.proportions,
            filters=self.filters,
            replacement=self.replacement,
            strict=strict,
        )


class SimplexQuery:
    """
    Fluent interface for composable simplex operations.

    This class provides a beautiful combinator-style API for building
    complex simplex generation queries. Operations are lazy and composable,
    executing only when materialized.

    Examples
    --------
    >>> # Pure functional style
    >>> sc.simplices(k=1).union(sc.simplices(k=2)).sample(100)

    >>> # Mixed proportional sampling
    >>> sc.mixed({1: 0.5, 2: 0.3, 3: 0.2}).total(1000)

    >>> # Complex chaining
    >>> (sc.simplices(k=2)
    ...   .where(lambda s: 0 in s)  # triangles containing vertex 0
    ...   .union(sc.simplices(k=3))  # plus all tetrahedra
    ...   .with_replacement()
    ...   .sample(500))

    >>> # Statistical sampling
    >>> (sc.dimensions([1, 2, 3])
    ...   .proportional({1: 0.4, 2: 0.4, 3: 0.2})
    ...   .total(10000))
    """

    def __init__(
        self,
        simplicial_complex: SimplicialComplex,
        spec: QuerySpec | None = None,
    ) -> None:
        """Initialize SimplexQuery with complex and specification."""
        self._complex = simplicial_complex
        self._spec = spec or QuerySpec()

    def simplices(self, k: int) -> SimplexQuery:
        """
        Focus on :math:`k`-dimensional simplices.

        Parameters
        ----------
        k : int
            Dimension of simplices to include in the query.

        Returns
        -------
        SimplexQuery
            New query including :math:`k`-simplices.

        Examples
        --------
        >>> sc.simplices(k=1)  # edges only
        >>> sc.simplices(k=2).simplices(k=3)  # triangles and tetrahedra
        """
        return SimplexQuery(self._complex, self._spec.with_dimension(k))

    def dimensions(self, ks: Iterable[int]) -> SimplexQuery:
        """
        Focus on multiple dimensions at once.

        Parameters
        ----------
        ks : iterable of int
            Dimensions to include.

        Returns
        -------
        SimplexQuery
            New query including all specified dimensions.
        """
        spec = self._spec
        for k in ks:
            spec = spec.with_dimension(k)
        return SimplexQuery(self._complex, spec)

    def mixed(self, proportions: dict[int, float]) -> SimplexQuery:
        """
        Set proportional mixing across dimensions.

        Parameters
        ----------
        proportions : dict of {int: float}
            Mapping from dimension to proportion. Must sum to 1.0.

        Returns
        -------
        SimplexQuery
            New query with proportional sampling configured.

        Examples
        --------
        >>> sc.mixed({1: 0.6, 2: 0.4})  # 60% edges, 40% triangles
        """
        return SimplexQuery(
            self._complex, self._spec.with_proportions(proportions)
        )

    def proportional(self, proportions: dict[int, float]) -> SimplexQuery:
        """Alias for mixed() - more readable in some contexts."""
        return self.mixed(proportions)

    def where(
        self, predicate: Callable[[frozenset[int]], bool]
    ) -> SimplexQuery:
        """
        Filter simplices by predicate.

        Parameters
        ----------
        predicate : callable
            Function that takes a simplex (frozenset) and returns bool.

        Returns
        -------
        SimplexQuery
            New query with additional filter applied.

        Examples
        --------
        >>> # triangles containing vertex 0
        >>> sc.simplices(k=2).where(lambda tri: 0 in tri)
        >>> # edges touching {0,1,2}
        >>> sc.simplices(k=1).where(lambda edge: len(edge & {0, 1, 2}) >= 1)
        """
        return SimplexQuery(self._complex, self._spec.with_filter(predicate))

    def filter(
        self, predicate: Callable[[frozenset[int]], bool]
    ) -> SimplexQuery:
        """Alias for where() - familiar to functional programmers."""
        return self.where(predicate)

    def containing(self, vertex: int) -> SimplexQuery:
        """
        Filter to simplices containing a specific vertex.

        Parameters
        ----------
        vertex : int
            Vertex that must be present in filtered simplices.

        Returns
        -------
        SimplexQuery
            New query filtered to simplices containing the vertex.
        """
        return self.where(lambda s: vertex in s)

    def intersecting(self, vertices: set[int]) -> SimplexQuery:
        """
        Filter to simplices that intersect with given vertex set.

        Parameters
        ----------
        vertices : set of int
            Vertex set that filtered simplices must intersect.

        Returns
        -------
        SimplexQuery
            New query filtered to intersecting simplices.
        """
        return self.where(lambda s: bool(s & vertices))

    def with_replacement(self, replacement: bool = True) -> SimplexQuery:
        """
        Configure sampling with/without replacement.

        Parameters
        ----------
        replacement : bool, default True
            Whether to allow sampling the same simplex multiple times.

        Returns
        -------
        SimplexQuery
            New query with replacement strategy configured.
        """
        return SimplexQuery(
            self._complex, self._spec.with_replacement(replacement)
        )

    def without_replacement(self) -> SimplexQuery:
        """Configure sampling without replacement (convenience method)."""
        return self.with_replacement(False)

    def strict(self, strict: bool = True) -> SimplexQuery:
        """
        Configure strict mode for error handling.

        Parameters
        ----------
        strict : bool, default True
            Whether to raise errors when insufficient simplices available.

        Returns
        -------
        SimplexQuery
            New query with strict mode configured.
        """
        return SimplexQuery(self._complex, self._spec.with_strict(strict))

    def lenient(self) -> SimplexQuery:
        """Configure lenient mode (convenience method)."""
        return self.strict(False)

    def union(self, other: SimplexQuery) -> SimplexQuery:
        """
        Combine with another query (union of dimensions).

        Parameters
        ----------
        other : SimplexQuery
            Another query to combine with this one.

        Returns
        -------
        SimplexQuery
            New query combining both dimension sets.

        Examples
        --------
        >>> edges = sc.simplices(k=1)
        >>> triangles = sc.simplices(k=2)
        >>> both = edges.union(triangles)
        """
        if other._complex != self._complex:
            raise ValueError("Cannot union queries from different complexes")

        combined_dims = self._spec.dimensions | other._spec.dimensions
        combined_props = {**self._spec.proportions, **other._spec.proportions}
        combined_filters = self._spec.filters + other._spec.filters

        new_spec = QuerySpec(
            dimensions=combined_dims,
            proportions=combined_props,
            filters=combined_filters,
            replacement=self._spec.replacement or other._spec.replacement,
            strict=self._spec.strict or other._spec.strict,  # Fix: OR not AND
        )

        return SimplexQuery(self._complex, new_spec)

    def __or__(self, other: SimplexQuery) -> SimplexQuery:
        """Operator overload for union: query1 | query2."""
        return self.union(other)

    # Materialization methods (terminal operations)

    def sample(self, n: int) -> dict[int, list[frozenset[int]]]:
        """
        Materialize query by sampling n simplices.

        Parameters
        ----------
        n : int
            Total number of simplices to sample.

        Returns
        -------
        dict of {int: list of frozenset of int}
            Sampled simplices grouped by dimension.
        """
        return self._materialize_with_count(n, is_sample=True)

    def total(self, n: int) -> dict[int, list[frozenset[int]]]:
        """
        Materialize query with total count constraint.

        Parameters
        ----------
        n : int
            Total number of simplices to generate.

        Returns
        -------
        dict of {int: list of frozenset of int}
            Generated simplices grouped by dimension.
        """
        return self._materialize_with_count(n, is_sample=False)

    def collect(
        self, max_per_dim: int = 1000000
    ) -> dict[int, list[frozenset[int]]]:
        """
        Materialize all simplices matching the query.

        Parameters
        ----------
        max_per_dim : int, default 1000000
            Safety limit per dimension.

        Returns
        -------
        dict of {int: list of frozenset of int}
            All matching simplices grouped by dimension.
        """
        result: dict[int, list[frozenset[int]]] = {}

        for k in self._spec.dimensions:
            generator = self._complex._get_generator_for_dimension(k)
            if generator.available() == 0:
                result[k] = []
                continue

            # Get all simplices for this dimension
            try:
                all_simplices = generator.all(max_count=max_per_dim)
            except ValueError:
                if self._spec.strict:
                    raise
                warnings.warn(
                    f"Too many {k}-simplices to collect, skipping "
                    f"dimension {k}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                all_simplices = []

            # Apply filters
            filtered = self._apply_filters(all_simplices)
            result[k] = filtered

        return result

    def count(self) -> dict[int, int]:
        """
        Count simplices matching the query without materializing them.

        Returns
        -------
        dict of {int: int}
            Count of simplices per dimension.
        """
        result = {}

        for k in self._spec.dimensions:
            generator = self._complex._get_generator_for_dimension(k)

            if not self._spec.filters:
                # No filters, can use efficient counting
                result[k] = generator.available()
            else:
                # Have filters, need to materialize and count
                try:
                    all_simplices = generator.all(max_count=100000)
                    filtered = self._apply_filters(all_simplices)
                    result[k] = len(filtered)
                except ValueError:
                    result[k] = -1  # Indicate uncountable

        return result

    def first(self, n: int = 1) -> dict[int, list[frozenset[int]]]:
        """
        Get first n simplices per dimension in lexicographic order.

        Parameters
        ----------
        n : int, default 1
            Number of simplices to get per dimension.

        Returns
        -------
        dict of {int: list of frozenset of int}
            First n simplices per dimension.
        """
        result = {}

        for k in self._spec.dimensions:
            generator = self._complex._get_generator_for_dimension(k)
            count_result = generator.count(n, strict=self._spec.strict)
            filtered = self._apply_filters(count_result.data)
            result[k] = filtered[:n]  # Take first n after filtering

        return result

    def exists(self) -> dict[int, bool]:
        """
        Check if any simplices exist for each dimension in query.

        Returns
        -------
        dict of {int: bool}
            Whether simplices exist for each dimension.
        """
        result = {}

        for k in self._spec.dimensions:
            generator = self._complex._get_generator_for_dimension(k)
            avail = generator.available()  # Store once to avoid changes

            if avail == 0:
                result[k] = False
                continue

            if not self._spec.filters:
                result[k] = avail > 0
            else:
                # Batch processing to handle sparse filters better
                found = False
                batch_size = 1000
                processed = 0
                max_to_check = min(10000, avail)  # Cap total checks

                while not found and processed < max_to_check:
                    remaining = avail - processed
                    if remaining <= 0:
                        break

                    batch_count = min(batch_size, remaining)
                    try:
                        batch = generator.count(batch_count, strict=False)
                        filtered = self._apply_filters(batch.data)
                        if filtered:
                            found = True
                    except ValueError:
                        break
                    processed += batch_count

                result[k] = found

        return result

    # Helper methods

    def _materialize_with_count(
        self, n: int, is_sample: bool
    ) -> dict[int, list[frozenset[int]]]:
        """Implement sample() and total() with remainder distribution."""
        result = {}

        if self._spec.proportions:
            # Proportional sampling with proper remainder distribution
            if abs(sum(self._spec.proportions.values()) - 1.0) > 1e-10:
                raise ValueError(
                    f"Proportions must sum to 1.0, got "
                    f"{sum(self._spec.proportions.values())}"
                )

            # Calculate base counts and track remainder
            base_counts = {}
            total_allocated = 0

            for k, proportion in self._spec.proportions.items():
                base_count = int(proportion * n)
                base_counts[k] = base_count
                total_allocated += base_count

            # Distribute remainder cyclically among all dimensions
            remainder = n - total_allocated
            if remainder > 0:
                dims = sorted(
                    self._spec.proportions.keys(),
                    key=lambda x: self._spec.proportions[x],
                    reverse=True,
                )
                for i in range(remainder):
                    k = dims[i % len(dims)]
                    base_counts[k] += 1

            for k, count in base_counts.items():
                if count > 0:
                    result[k] = self._sample_dimension(k, count, is_sample)
        else:
            # Equal distribution across dimensions
            if not self._spec.dimensions:
                raise ValueError("No dimensions specified in query")

            per_dim = n // len(self._spec.dimensions)
            remainder = n % len(self._spec.dimensions)

            for i, k in enumerate(sorted(self._spec.dimensions)):
                count = per_dim + (1 if i < remainder else 0)
                if count > 0:
                    result[k] = self._sample_dimension(k, count, is_sample)

        # In strict mode, verify we got the requested total count
        if self._spec.strict:
            total_returned = sum(
                len(simplices) for simplices in result.values()
            )
            if total_returned < n:
                raise ValueError(
                    f"Requested {n} simplices but only {total_returned} "
                    f"available after filtering"
                )

        return result

    def _sample_dimension(
        self, k: int, count: int, is_sample: bool
    ) -> list[frozenset[int]]:
        """Sample or count from a specific dimension."""
        generator = self._complex._get_generator_for_dimension(k)

        if is_sample:
            sample_result = generator.sample(
                count,
                replacement=self._spec.replacement,
                strict=self._spec.strict,
            )
            candidates = sample_result.data
        else:
            count_result = generator.count(count, strict=self._spec.strict)
            candidates = count_result.data

        return self._apply_filters(candidates)

    def _apply_filters(
        self, simplices: list[frozenset[int]]
    ) -> list[frozenset[int]]:
        """Apply all filters to a list of simplices with early exit."""
        if not self._spec.filters:
            return simplices

        result = simplices
        for predicate in self._spec.filters:
            if not result:  # Early exit if empty
                break
            result = [s for s in result if predicate(s)]

        return result

    def __repr__(self) -> str:
        """Return string representation showing query specification."""
        parts = []

        if self._spec.dimensions:
            dims = sorted(self._spec.dimensions)
            parts.append(f"dims={dims}")

        if self._spec.proportions:
            parts.append(f"proportions={self._spec.proportions}")

        if self._spec.filters:
            parts.append(f"filters={len(self._spec.filters)}")

        if self._spec.replacement:
            parts.append("replacement=True")

        if self._spec.strict:
            parts.append("strict=True")

        spec_str = ", ".join(parts) if parts else "empty"
        return f"SimplexQuery({spec_str})"


class SimplicialComplex:
    """
    Fluent simplicial complex with combinator-style API.

    Provides an elegant, composable interface for simplex generation
    that feels natural to functional programmers. All operations are
    lazy and chainable until materialized.

    Examples
    --------
    >>> # Create a complex
    >>> sc = SimplicialComplex([[0, 1, 2, 3, 4]])  # 4-simplex

    >>> # Simple queries
    >>> edges = sc.simplices(k=1).sample(10)
    >>> triangles = sc.simplices(k=2).first(5)

    >>> # Mixed sampling
    >>> mixed = sc.mixed({1: 0.5, 2: 0.3, 3: 0.2}).total(1000)

    >>> # Complex chaining
    >>> filtered = (sc.simplices(k=2)
    ...            .where(lambda tri: 0 in tri)
    ...            .union(sc.simplices(k=3))
    ...            .with_replacement()
    ...            .sample(100))

    >>> # Statistical analysis
    >>> counts = sc.dimensions([1, 2, 3]).count()
    >>> exists = sc.simplices(k=4).containing(0).exists()
    """

    def __init__(self, facets: Iterable[Iterable[int]] = ()) -> None:
        """Initialize simplicial complex from facets with optimized storage."""
        self._facets: list[tuple[int, ...]] = []

        # Collect facets first
        facet_list = []
        for facet in facets:
            vertices = tuple(
                sorted(set(v for v in facet if isinstance(v, int) and v >= 0))
            )
            if vertices:
                facet_list.append(vertices)

        self._facets = facet_list

        # Use set for deduplication if many facets, otherwise stick to
        # list for memory
        if len(self._facets) > 1000:
            self._facet_sets: set[frozenset[int]] | list[frozenset[int]] = set(
                frozenset(f) for f in self._facets
            )
        else:
            self._facet_sets = [frozenset(f) for f in self._facets]

    @property
    def dimension(self) -> int:
        """Get maximum dimension across all facets."""
        return max((len(facet) - 1 for facet in self._facets), default=-1)

    @property
    def facets(self) -> list[tuple[int, ...]]:
        """Return copy of facets list."""
        return self._facets.copy()

    # Fluent API entry points

    def simplices(self, k: int) -> SimplexQuery:
        """
        Start a query for :math:`k`-dimensional simplices.

        This is the main entry point for fluent simplex queries.

        Parameters
        ----------
        k : int
            Dimension of simplices to query.

        Returns
        -------
        SimplexQuery
            Fluent query object for further chaining.

        Examples
        --------
        >>> sc.simplices(k=1).sample(100)  # 100 random edges
        >>> # all triangles containing vertex 0
        >>> sc.simplices(k=2).where(lambda tri: 0 in tri).collect()
        """
        return SimplexQuery(self).simplices(k)

    def dimensions(self, ks: Iterable[int]) -> SimplexQuery:
        """
        Start a query for multiple dimensions.

        Parameters
        ----------
        ks : iterable of int
            Dimensions to include in query.

        Returns
        -------
        SimplexQuery
            Fluent query object for further chaining.
        """
        return SimplexQuery(self).dimensions(ks)

    def mixed(self, proportions: dict[int, float]) -> SimplexQuery:
        """
        Start a proportional mixed-dimension query.

        Parameters
        ----------
        proportions : dict of {int: float}
            Mapping from dimension to proportion.

        Returns
        -------
        SimplexQuery
            Fluent query object for further chaining.
        """
        return SimplexQuery(self).mixed(proportions)

    def all_dimensions(self) -> SimplexQuery:
        """
        Start a query including all possible dimensions.

        Returns
        -------
        SimplexQuery
            Query covering dimensions 0 through self.dimension.
        """
        return self.dimensions(range(self.dimension + 1))

    # Internal helper methods

    def _get_generator_for_dimension(self, k: int) -> SimplexGenerator:
        """Get generator for k-simplices from largest compatible facet."""
        if k < 0:
            return SimplexGenerator((), k)

        compatible_facets = [f for f in self._facets if len(f) >= k + 1]
        if not compatible_facets:
            return SimplexGenerator((), k)

        largest_facet = max(compatible_facets, key=len)
        return SimplexGenerator(largest_facet, k)

    # Legacy methods for backward compatibility

    def total_simplices(self) -> int:
        """Count total simplices across all dimensions (legacy method)."""
        return sum(self.all_dimensions().count().values())

    def count_k_simplices(self, k: int) -> int:
        """Count :math:`k`-simplices (legacy method)."""
        counts = self.simplices(k).count()
        return counts.get(k, 0)

    def __contains__(self, simplex: Iterable[int]) -> bool:
        """
        Check if simplex exists in the complex.

        For large complexes (>1000 facets), uses optimized set-based
        checking with a fast path for facet-equality queries, but falls
        back to comprehensive subset checking when needed.

        Parameters
        ----------
        simplex : iterable of int
            The simplex to check for membership.

        Returns
        -------
        bool
            True if the simplex exists in the complex.

        Notes
        -----
        For large complexes, the direct membership check
        `query in self._facet_sets` is optimized for cases where the
        query simplex is itself a facet (same size).
        """
        query = frozenset(simplex)
        if not query:
            return False

        # Use cached frozensets for faster subset checking
        if isinstance(self._facet_sets, set):
            # For large complexes, check if query is directly in set first
            # (facet equality)
            if query in self._facet_sets:
                return True
            # Then check subset relationships
            return any(
                query.issubset(facet_set) for facet_set in self._facet_sets
            )
        else:
            return any(
                query.issubset(facet_set) for facet_set in self._facet_sets
            )

    def __repr__(self) -> str:
        """Return concise string representation."""
        if not self._facets:
            return "SimplicialComplex(∅)"
        return (
            f"SimplicialComplex(dim={self.dimension}, "
            f"facets={len(self._facets)})"
        )
