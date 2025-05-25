import itertools
from collections.abc import Iterable
from typing import Any, cast

import pytest
import torch

from energy_transformer.layers.simplicial import (
    QuerySpec,
    SimplexGenerator,
    SimplicialComplex,
    UnionSimplexGenerator,
    _get_membership,
    canonical,
    energy,
    lse,
    membership,
    unrank,
    update,
)


def test_canonical_filters_and_sorts() -> None:
    simplex: list[Any] = [3, 2, 2, -1, "a", 1]
    assert canonical(cast(Iterable[int], simplex)) == (1, 2, 3)
    assert canonical([]) == ()


def test_unrank_matches_combinations() -> None:
    n, k, r = 5, 3, 3
    combos = list(itertools.combinations(range(n), k))
    assert unrank(r, n, k) == combos[r]


def test_unrank_out_of_range() -> None:
    with pytest.raises(ValueError):
        unrank(10, 4, 2)


def test_simplex_generator_basics() -> None:
    gen = SimplexGenerator((0, 1, 2, 3), k=1)
    assert gen.n == 4
    assert gen.total_count == 6
    assert bool(gen)
    assert len(gen) == 6
    expected = [
        frozenset({0, 1}),
        frozenset({0, 2}),
        frozenset({0, 3}),
        frozenset({1, 2}),
        frozenset({1, 3}),
        frozenset({2, 3}),
    ]
    assert gen.all() == expected


def test_simplex_generator_sampling_rules() -> None:
    gen = SimplexGenerator((0, 1, 2), k=1)
    with pytest.raises(ValueError):
        gen.sample(4, replacement=False, strict=True)

    res = gen.sample(4, replacement=False, strict=False)
    assert len(res.data) == 3
    assert not res.complete

    res = gen.sample(5, replacement=True)
    assert len(res.data) == 5
    assert res.complete


def test_union_simplex_generator() -> None:
    facets: list[tuple[int, ...]] = [(0, 1, 2), (1, 2, 3)]
    gen = UnionSimplexGenerator(facets, k=1)
    assert gen.computation_safe
    assert len(gen) == 5
    all_simplices = gen.all()
    expected = {
        frozenset({0, 1}),
        frozenset({0, 2}),
        frozenset({1, 2}),
        frozenset({1, 3}),
        frozenset({2, 3}),
    }
    assert set(all_simplices) == expected


def test_simplicial_complex_properties_and_query() -> None:
    complex = SimplicialComplex([(0, 1, 2), (1, 3)])
    assert complex.dimension == 2
    assert (0, 1, 2) in complex.facets
    assert {0, 1} in complex
    assert {0, 3} not in complex

    result = complex.simplices(1).collect()
    assert set(result[1]) == {
        frozenset({0, 1}),
        frozenset({0, 2}),
        frozenset({1, 2}),
        frozenset({1, 3}),
    }
    count = complex.simplices(1).count()
    assert count == {1: 4}


def test_membership_matrix() -> None:
    κ = [(0, 1), (1, 2)]
    μ = _get_membership(κ, 3, torch.device("cpu"), torch.float32).coalesce()
    assert μ.shape == torch.Size([3, 2])
    indices = μ.indices().tolist()
    assert indices == [[0, 1, 1, 2], [0, 0, 1, 1]]


def test_energy_matches_lse_relationship() -> None:
    torch.manual_seed(0)
    ξ = torch.randn(2, 3)
    g = torch.randn(3)
    κ = [(0, 1), (2,)]
    β = 1.0
    e = energy(β, ξ, g, κ)
    lse_val = lse(β, ξ, g, κ)
    expected = -lse_val + 0.5 * torch.dot(g, g)
    assert torch.allclose(e, expected)


def test_query_spec_proportions_validation() -> None:
    with pytest.raises(ValueError):
        QuerySpec(proportions={0: 0.6, 1: 0.5})


def test_membership_invalid_vertex() -> None:
    with pytest.raises(ValueError):
        membership(((0, 1), (3,)), 3, "cpu", "torch.float32")


def test_membership_empty_complex_returns_empty_matrix() -> None:
    μ = membership((), 4, "cpu", "torch.float32").coalesce()
    assert μ.shape == torch.Size([4, 0])
    assert μ._nnz() == 0


def test_simplex_query_filtering_and_count() -> None:
    complex = SimplicialComplex([(0, 1, 2), (1, 2, 3)])
    query = complex.simplices(1).containing(0)
    collected = query.collect()
    assert set(collected[1]) == {frozenset({0, 1}), frozenset({0, 2})}
    count = query.count()
    assert count == {1: 2}


def _manual_update(
    β: float, ξ: torch.Tensor, g: torch.Tensor, κ: list[list[int]]
) -> torch.Tensor:
    μ = _get_membership(κ, g.shape[0], torch.device("cpu"), ξ.dtype)
    y = ξ * g  # (P, N)
    dot = torch.sparse.mm(μ.t(), y.t()).t()  # (P, |κ|)
    α = torch.softmax(dot.sum(dim=1) / β, dim=0)
    return α @ ξ


def test_update_matches_manual_computation() -> None:
    torch.manual_seed(0)
    ξ = torch.randn(2, 3)
    g = torch.randn(3)
    κ = [(0, 1), (2,)]
    β = 0.5
    out = update(β, ξ, g, κ)
    expected = _manual_update(β, ξ, g, κ)
    assert torch.allclose(out, expected)


def test_update_raises_for_shape_mismatch() -> None:
    ξ = torch.randn(1, 2)
    g = torch.randn(3)
    with pytest.raises(ValueError):
        update(1.0, ξ, g, [(0,)])
