import pytest
import torch

from energy_transformer.layers.attention import MultiheadEnergyAttention
from energy_transformer.layers.hopfield import HopfieldNetwork
from energy_transformer.layers.layer_norm import EnergyLayerNorm
from energy_transformer.models.base import EnergyTransformer

pytestmark = pytest.mark.unit


def test_forward_returns_tensor() -> None:
    model = EnergyTransformer(
        layer_norm=EnergyLayerNorm(4),
        attention=MultiheadEnergyAttention(4, num_heads=1),
        hopfield=HopfieldNetwork(4, hidden_dim=4),
        steps=2,
    )
    x = torch.randn(1, 3, 4)
    out = model(x)
    assert out.shape == x.shape


def test_return_energies() -> None:
    model = EnergyTransformer(
        layer_norm=EnergyLayerNorm(2),
        attention=MultiheadEnergyAttention(2, num_heads=1),
        hopfield=HopfieldNetwork(2, hidden_dim=2),
        steps=1,
    )
    x = torch.randn(1, 2, 2)
    out, energies = model(x, return_energies=True)
    assert isinstance(out, torch.Tensor)
    assert len(energies) == 1
