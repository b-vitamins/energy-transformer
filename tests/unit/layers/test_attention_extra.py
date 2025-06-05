import pathlib

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
    assert "embed_dim=4" in rep
    assert "num_heads=2" in rep
    assert "beta=0.5000" in rep
    assert "batch_first=False" in rep


def test_attention_properties() -> None:
    attn = MultiheadEnergyAttention(embed_dim=8, num_heads=2)
    assert attn.head_dim == 4
    expected_params = 2 * attn.num_heads * attn.head_dim * attn.embed_dim
    assert attn.total_params == expected_params
    assert attn.requires_grad_
    assert isinstance(attn.device, torch.device)
    assert isinstance(attn.dtype, torch.dtype)
    assert not attn.is_mixed_precision


def test_memory_efficient_matches_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "energy_transformer.layers.attention.MEMORY_EFFICIENT_SEQ_THRESHOLD", 2
    )
    attn = MultiheadEnergyAttention(embed_dim=3, num_heads=1)
    g = torch.randn(1, 5, 3)
    full = attn(g)
    efficient = attn(g, use_memory_efficient=True)
    assert torch.allclose(full, efficient, atol=1e-6)


def test_memory_efficient_with_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "energy_transformer.layers.attention.MEMORY_EFFICIENT_SEQ_THRESHOLD", 2
    )
    attn = MultiheadEnergyAttention(embed_dim=2, num_heads=1)
    g = torch.randn(1, 4, 2)
    mask = torch.zeros(4, 4)
    mask[0, 1] = float("-inf")
    full = attn(g, attn_mask=mask)
    efficient = attn(g, attn_mask=mask, use_memory_efficient=True)
    assert torch.allclose(full, efficient, atol=1e-6)


def test_memory_efficient_causal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "energy_transformer.layers.attention.MEMORY_EFFICIENT_SEQ_THRESHOLD", 2
    )
    attn = MultiheadEnergyAttention(embed_dim=2, num_heads=1)
    g = torch.randn(1, 4, 2)
    full = attn(g, is_causal=True)
    efficient = attn(g, is_causal=True, use_memory_efficient=True)
    assert torch.allclose(full, efficient, atol=1e-6)


def test_state_dict_includes_metadata(tmp_path: pathlib.Path) -> None:
    attn = MultiheadEnergyAttention(embed_dim=4, num_heads=2)
    path = tmp_path / "model.pth"
    torch.save(attn.state_dict(), path)
    state = torch.load(path)
    assert state["_metadata.version"] == "1.0"
    cfg = state["_metadata.config"]
    assert cfg["embed_dim"] == 4
    assert cfg["num_heads"] == 2
