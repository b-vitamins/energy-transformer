import pytest
import torch

from energy_transformer.layers.attention import MultiheadEnergyAttention

pytestmark = pytest.mark.unit


def test_compute_energy_scalar() -> None:
    attn = MultiheadEnergyAttention(embed_dim=4, num_heads=2)
    g = torch.randn(2, 3, 4)
    energy = attn(g)
    assert energy.ndim == 0


def test_invalid_embed_dim_raises() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiheadEnergyAttention(embed_dim=5, num_heads=2)


def test_custom_beta_tensor() -> None:
    beta = torch.tensor([0.5, 0.5])
    attn = MultiheadEnergyAttention(embed_dim=4, num_heads=2, beta=beta)
    assert torch.allclose(attn.betas, beta)
