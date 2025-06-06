import pytest
import torch

from energy_transformer.layers.attention import MultiheadEnergyAttention

pytestmark = pytest.mark.unit


def test_compute_energy_scalar() -> None:
    attn = MultiheadEnergyAttention(embed_dim=4, num_heads=2)
    g = torch.randn(2, 3, 4)
    energy = attn(g)
    assert energy.ndim == 0
