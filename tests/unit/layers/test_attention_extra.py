import pytest
import torch

from energy_transformer.layers.attention import MultiheadEnergyAttention

pytestmark = pytest.mark.unit


def test_compute_grad_matches_autograd() -> None:
    attn = MultiheadEnergyAttention(embed_dim=2, num_heads=1, beta=1.0)
    with torch.no_grad():
        attn.k_proj_weight.fill_(1.0)
        attn.q_proj_weight.fill_(1.0)
    x = torch.randn(1, 2, 2, requires_grad=True)
    grad = attn.compute_grad(x)
    energy = attn(x)
    energy.backward()
    assert torch.allclose(x.grad, grad, atol=1e-6)


def test_compute_energy_matches_forward_mean() -> None:
    attn = MultiheadEnergyAttention(embed_dim=2, num_heads=1, beta=1.0)
    with torch.no_grad():
        attn.k_proj_weight.zero_()
        attn.q_proj_weight.zero_()
    x = torch.randn(3, 2, 2)
    energy = attn.compute_energy(x)
    expected = attn(x) / x.shape[0]
    assert torch.allclose(energy, expected)


def test_extra_repr_non_default() -> None:
    attn = MultiheadEnergyAttention(
        embed_dim=4, num_heads=2, beta=0.5, batch_first=False
    )
    rep = attn.extra_repr()
    assert "4, 2" in rep
    assert "beta=0.5000" in rep
    assert "batch_first=False" in rep
