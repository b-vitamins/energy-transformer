import math
import random

import pytest
import torch

from energy_transformer.layers import simplicial


def test_simplex_validator_success() -> None:
    # Input simplices with duplicates and different orderings
    simplices = [
        [1, 0],
        (2, 0, 1),
        [1, 0, 2],
        [3, 2, 1],
        [0, 2],
    ]  # Added [0, 2] edge
    buckets, max_v = simplicial.SimplexValidator.validate_and_canonicalize(
        simplices,
    )
    # Now the test expectations match what we actually provide
    assert buckets[2] == [[0, 1], [0, 2]]
    assert buckets[3] == [[0, 1, 2], [1, 2, 3]]
    assert max_v == 3

    # Alternative: Fix the test to match the actual input
    simplices2 = [[1, 0], (2, 0, 1), [1, 0, 2], [3, 2, 1]]
    buckets2, max_v2 = simplicial.SimplexValidator.validate_and_canonicalize(
        simplices2,
    )
    assert buckets2[2] == [[0, 1]]  # Only one edge in input
    assert buckets2[3] == [[0, 1, 2], [1, 2, 3]]  # Two unique triangles
    assert max_v2 == 3


@pytest.mark.parametrize(
    ("simplices", "exc"),
    [
        ([], ValueError),
        ([[0]], ValueError),
        ([[-1, 1]], ValueError),
        ([[0, 0]], ValueError),
        ([[0, "a"]], TypeError),
    ],
)
def test_simplex_validator_errors(simplices, exc) -> None:
    with pytest.raises(exc):
        simplicial.SimplexValidator.validate_and_canonicalize(simplices)


def test_simplex_budget_manager_edge_units() -> None:
    assert simplicial.SimplexBudgetManager.edge_units(3) == 3


def test_simplex_budget_manager_normalize_dimension_weights() -> None:
    assert simplicial.SimplexBudgetManager.normalize_dimension_weights(
        None,
        2,
    ) == {1: 1.0}

    weights = simplicial.SimplexBudgetManager.normalize_dimension_weights(
        {1: 1, 2: 1},
        2,
    )
    assert weights == {1: 0.5, 2: 0.5}

    with pytest.raises(ValueError, match="dim_weights"):
        simplicial.SimplexBudgetManager.normalize_dimension_weights({3: -1}, 2)


def test_allocate_budget_consumes_and_returns_remaining() -> None:
    candidates = {2: [(0, 1), (0, 2), (1, 2)], 3: [(0, 1, 2)]}
    weights = {1: 0.5, 2: 0.5}
    simplices, remaining = simplicial.SimplexBudgetManager.allocate_budget(
        6,
        weights,
        candidates,
    )
    # Order of generated simplices depends on RNG; just check counts
    assert len(simplices) == 4
    assert remaining == 0


def test_random_simplex_generator_generate() -> None:
    gen = simplicial.RandomSimplexGenerator(random.Random(0))  # noqa: S311
    simplices = gen.generate(4, 2, budget=1.0)
    # Should generate some simplices respecting budget
    assert len(simplices) > 0

    with pytest.raises(ValueError, match="Budget too small"):
        gen.generate(3, 1, budget=0.0)


class FakeArray:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self._shape = (len(self.data), len(self.data[0]) if self.data else 0)

    @property
    def shape(self):
        return self._shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class FakeNP:
    def array(self, data):
        return FakeArray(data)

    def sqrt(self, x):
        return math.sqrt(x)


class FakeKDTree:
    def __init__(self, coords):
        self.coords = coords

    def query(self, coords, k):
        n = len(coords)
        indices = []
        for i in range(n):
            # first element is self
            row = [i] + [(i + j + 1) % n for j in range(k - 1)]
            indices.append(row)
        return None, indices


class FakeSimplex(list):
    def tolist(self):
        return list(self)


class FakeDelaunay:
    def __init__(self, coords):
        dim = coords.shape[1]
        if dim == 2:
            self.simplices = [FakeSimplex([0, 1, 2])]
        else:
            self.simplices = [FakeSimplex([0, 1, 2, 3])]


@pytest.fixture(autouse=True)
def fake_scipy(monkeypatch):
    monkeypatch.setattr(simplicial, "HAS_SCIPY", True)
    monkeypatch.setattr(simplicial, "np", FakeNP())
    monkeypatch.setattr(simplicial, "cKDTree", FakeKDTree)
    monkeypatch.setattr(simplicial, "Delaunay", FakeDelaunay)


def test_topology_aware_generator_2d() -> None:
    coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    gen = simplicial.TopologyAwareSimplexGenerator(
        coords,
        k_neighbors=1,
        rng=random.Random(0),  # noqa: S311
    )
    simplices = gen.generate(3, 2, budget=1.0)
    assert [0, 1] in simplices
    assert any(len(s) == 3 for s in simplices)


def test_topology_aware_generator_3d_with_sampling() -> None:
    coords = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    gen = simplicial.TopologyAwareSimplexGenerator(coords, rng=random.Random(0))  # noqa: S311
    simplices = gen.generate(4, 3, budget=0.5)
    assert len(simplices) >= 1


def test_simplex_factory_returns_correct_generator() -> None:
    g1 = simplicial.SimplexFactory.create_generator(None)
    assert isinstance(g1, simplicial.RandomSimplexGenerator)

    coords = [(0.0, 0.0), (1.0, 0.0)]
    g2 = simplicial.SimplexFactory.create_generator(coords)
    assert isinstance(g2, simplicial.TopologyAwareSimplexGenerator)


def test_autogen_simps_uses_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyGen(simplicial.RandomSimplexGenerator):
        def __init__(self):
            pass

        def generate(self, *a, **k):
            return [[0, 1]]

    monkeypatch.setattr(
        simplicial.SimplexFactory,
        "create_generator",
        lambda *a, **k: DummyGen(),
    )
    simplices = simplicial._autogen_simps(2, max_dim=1, twiddle=1.0)
    assert simplices == [[0, 1]]

    with pytest.raises(ValueError, match="at least 2"):
        simplicial._autogen_simps(1, max_dim=1, twiddle=1.0)
    with pytest.raises(ValueError, match="max_dim"):
        simplicial._autogen_simps(3, max_dim=0, twiddle=1.0)

    monkeypatch.setattr(DummyGen, "generate", lambda *a, **k: [])
    with pytest.raises(RuntimeError, match="Auto-generation"):
        simplicial._autogen_simps(3, max_dim=1, twiddle=1.0)


def test_simplex_product_and_forward() -> None:
    simp_net = simplicial.SimplicialHopfieldNetwork(
        in_dim=1,
        simplices=[[0, 1], [1, 2, 3]],
        hidden_dim=2,
        temperature=1.0,
    )
    g = torch.ones(1, 4, 1)
    out = simp_net(g)
    assert out.shape == torch.Size([])

    u = torch.ones(1, 2, 4)
    idx = torch.tensor([[0, 1], [1, 2]])
    prod = simplicial.SimplicialHopfieldNetwork._simplex_product(u, idx)
    assert prod.shape == (1, 2, 2)


def test_shn_forward_raises_for_few_tokens() -> None:
    simp_net = simplicial.SimplicialHopfieldNetwork(
        in_dim=1,
        simplices=[[0, 1]],
        hidden_dim=1,
    )
    with pytest.raises(ValueError, match="Input has"):
        simp_net(torch.zeros(1, 1, 1))


def test_shn_autogeneration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(simplicial, "_autogen_simps", lambda *a, **k: [[0, 1]])
    net = simplicial.SimplicialHopfieldNetwork(
        in_dim=1,
        simplices=None,
        num_vertices=2,
    )
    assert [0, 1] in net.simps_by_size[2].tolist()
