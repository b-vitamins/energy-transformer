import pytest
from torch import nn

from energy_transformer.models.vision import viset

pytestmark = pytest.mark.unit


class DummyEnergyTransformer(nn.Module):
    def __init__(self, layer_norm, attention, hopfield, steps=0, alpha=0.0):
        super().__init__()
        self.layer_norm = layer_norm
        self.attention = attention
        self.hopfield = hopfield
        self.steps = steps
        self.alpha = alpha


class DummyLayerNorm(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim


class DummyAttention(nn.Module):
    def __init__(self, in_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.head_dim = head_dim


class DummySimplicialHopfieldNetwork(nn.Module):
    def __init__(
        self,
        in_dim,
        simplices,
        num_vertices,
        coordinates,
        max_dim,
        budget,
        dim_weights,
        hidden_dim,
    ):
        super().__init__()
        self.params = {
            "in_dim": in_dim,
            "simplices": simplices,
            "num_vertices": num_vertices,
            "coordinates": coordinates,
            "max_dim": max_dim,
            "budget": budget,
            "dim_weights": dim_weights,
            "hidden_dim": hidden_dim,
        }


@pytest.fixture(autouse=True)
def patch_components(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(viset, "EnergyTransformer", DummyEnergyTransformer)
    monkeypatch.setattr(viset, "LayerNorm", DummyLayerNorm)
    monkeypatch.setattr(viset, "MultiheadEnergyAttention", DummyAttention)
    monkeypatch.setattr(
        viset,
        "SimplicialHopfieldNetwork",
        DummySimplicialHopfieldNetwork,
    )


def test_viset_initializes_with_topology() -> None:
    model = viset.VisionSimplicialEnergyTransformer(
        img_size=4,
        patch_size=2,
        in_chans=3,
        num_classes=2,
        embed_dim=8,
        depth=2,
        num_heads=2,
        _head_dim=4,
        hopfield_hidden_dim=16,
        et_steps=3,
        et_alpha=0.1,
        use_topology=True,
        simplex_budget=0.2,
        simplex_max_dim=2,
    )
    assert len(model.et_blocks) == 2
    block = model.et_blocks[0]
    params = block.hopfield.params
    assert params["coordinates"] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert params["dim_weights"] == {1: 0.5, 2: 0.5}
    assert params["budget"] == 0.2
    assert params["max_dim"] == 2


def test_viset_without_topology_custom_weights() -> None:
    model = viset.VisionSimplicialEnergyTransformer(
        img_size=4,
        patch_size=2,
        in_chans=3,
        num_classes=2,
        embed_dim=8,
        depth=1,
        num_heads=2,
        _head_dim=4,
        hopfield_hidden_dim=16,
        et_steps=1,
        et_alpha=0.2,
        use_topology=False,
        simplex_budget=0.3,
        simplex_max_dim=3,
        simplex_dim_weights={1: 0.4, 3: 0.6},
    )
    block = model.et_blocks[0]
    params = block.hopfield.params
    assert params["coordinates"] is None
    assert params["dim_weights"] == {1: 0.4, 3: 0.6}


def test_factory_functions_return_models() -> None:
    factories = [
        viset.viset_tiny,
        viset.viset_small,
        viset.viset_base,
        viset.viset_2l_e50_t50_cifar,
        viset.viset_2l_e100_cifar,
        viset.viset_2l_t100_cifar,
        viset.viset_2l_random_cifar,
        viset.viset_4l_e50_t50_cifar,
        viset.viset_6l_e50_t50_cifar,
        viset.viset_2l_e40_t40_tet20_cifar,
    ]
    for fn in factories:
        if "cifar" in fn.__name__:
            model = fn(num_classes=2)
        else:
            model = fn(img_size=4, patch_size=2, in_chans=3, num_classes=2)
        assert isinstance(model, viset.VisionSimplicialEnergyTransformer)
    random_model = viset.viset_2l_random_cifar(num_classes=2)
    block = random_model.et_blocks[0]
    assert block.hopfield.params["coordinates"] is None


def test_get_viset_name() -> None:
    assert viset.get_viset_name(2, {1: 0.5, 2: 0.5}) == "ViSET-2L-E50-T50"
    assert viset.get_viset_name(3, {2: 1.0}) == "ViSET-3L-T100"
    assert (
        viset.get_viset_name(2, {1: 1.0}, use_topology=False)
        == "ViSET-2L-Random"
    )
