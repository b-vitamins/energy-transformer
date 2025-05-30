import math

import pytest
import torch

from energy_transformer.layers.attention import MultiHeadEnergyAttention


def _manual_energy(
    g: torch.Tensor,
    w_k: torch.Tensor,
    w_q: torch.Tensor,
    beta: float = 1.0,
    attn_mask: torch.Tensor | None = None,
    include_diag: bool = True,
) -> torch.Tensor:
    k = torch.einsum("...nd,hyd->...nhy", g, w_k)
    q = torch.einsum("...nd,hyd->...nhy", g, w_q)
    a = torch.einsum("...nhy,...mhy->...hnm", k, q)
    if not include_diag:
        diag = torch.eye(g.shape[-2], device=g.device, dtype=torch.bool)
        diag = diag[None, None]
        a = a.masked_fill(diag, float("-inf"))
    if attn_mask is not None:
        a = a + attn_mask
    beta_a = beta * a
    lse = torch.logsumexp(beta_a, dim=-2)
    return -(1.0 / beta * lse).sum()


def test_attention_energy_matches_manual() -> None:
    attn = MultiHeadEnergyAttention(
        in_dim=2, num_heads=1, head_dim=2, beta=1.0, bias=False
    )
    with torch.no_grad():
        attn.w_k.copy_(torch.eye(2).unsqueeze(0))
        attn.w_q.copy_(torch.eye(2).unsqueeze(0))
    g = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    energy = attn(g)
    expected = _manual_energy(g, attn.w_k, attn.w_q)
    assert torch.allclose(energy, expected, atol=1e-6)


def test_attention_excludes_diagonal() -> None:
    attn = MultiHeadEnergyAttention(
        in_dim=1, num_heads=1, head_dim=1, beta=1.0, bias=False
    )
    with torch.no_grad():
        attn.w_k.fill_(1.0)
        attn.w_q.fill_(1.0)
    g = torch.ones(1, 2, 1)
    e1 = attn(g, include_diag=True)
    e2 = attn(g, include_diag=False)
    exp1 = _manual_energy(g, attn.w_k, attn.w_q, include_diag=True)
    exp2 = _manual_energy(g, attn.w_k, attn.w_q, include_diag=False)
    assert torch.allclose(e1, exp1, atol=1e-6)
    assert torch.allclose(e2, exp2, atol=1e-6)


def test_attention_applies_mask() -> None:
    attn = MultiHeadEnergyAttention(
        in_dim=1, num_heads=1, head_dim=1, beta=1.0, bias=False
    )
    with torch.no_grad():
        attn.w_k.fill_(1.0)
        attn.w_q.fill_(1.0)
    g = torch.ones(1, 2, 1)
    mask = torch.tensor([[[[0.0, float("-inf")], [0.0, 0.0]]]])
    energy = attn(g, attn_mask=mask)
    expected = _manual_energy(g, attn.w_k, attn.w_q, attn_mask=mask)
    assert torch.allclose(energy, expected, atol=1e-6)


def test_attention_single_token_zero() -> None:
    attn = MultiHeadEnergyAttention(in_dim=3, num_heads=1, head_dim=2)
    g = torch.randn(1, 1, 3)
    energy = attn(g)
    assert torch.allclose(energy, torch.tensor(0.0))


def test_attention_parameter_shapes_and_bias() -> None:
    attn_no_bias = MultiHeadEnergyAttention(
        in_dim=3, num_heads=2, head_dim=4, bias=False
    )
    assert attn_no_bias.w_k.shape == (2, 4, 3)
    assert attn_no_bias.w_q.shape == (2, 4, 3)
    assert attn_no_bias.b_k is None
    assert attn_no_bias.b_q is None

    attn_bias = MultiHeadEnergyAttention(
        in_dim=3, num_heads=2, head_dim=4, bias=True
    )
    assert attn_bias.b_k.shape == (4,)
    assert attn_bias.b_q.shape == (4,)


def test_attention_default_beta_matches_manual() -> None:
    head_dim = 4
    attn = MultiHeadEnergyAttention(
        in_dim=2, num_heads=1, head_dim=head_dim, beta=None, bias=False
    )
    assert attn.β == pytest.approx(1.0 / math.sqrt(head_dim))
    with torch.no_grad():
        attn.w_k.zero_()
        attn.w_q.zero_()
        attn.w_k[0, :2].copy_(torch.eye(2))
        attn.w_q[0, :2].copy_(torch.eye(2))
    g = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    energy = attn(g)
    expected = _manual_energy(g, attn.w_k, attn.w_q, beta=attn.β)
    assert torch.allclose(energy, expected, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_attention_mixed_precision(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    attn = MultiHeadEnergyAttention(
        in_dim=1, num_heads=1, head_dim=1, beta=1.0, bias=False
    )
    with torch.no_grad():
        attn.w_k.fill_(1.0)
        attn.w_q.fill_(1.0)
    g = torch.ones(1, 2, 1, dtype=dtype)
    energy = attn(g)
    expected = _manual_energy(g, attn.w_k.to(dtype), attn.w_q.to(dtype))
    assert energy.dtype == dtype

    # bfloat16 has lower precision than float16
    atol = 5e-3 if dtype == torch.bfloat16 else 1e-3
    assert torch.allclose(energy, expected, atol=atol)


def test_attention_mask_broadcast() -> None:
    attn = MultiHeadEnergyAttention(
        in_dim=1, num_heads=2, head_dim=1, beta=1.0, bias=False
    )
    with torch.no_grad():
        attn.w_k.fill_(1.0)
        attn.w_q.fill_(1.0)
    g = torch.ones(1, 3, 1)
    mask = torch.tensor(
        [[0.0, float("-inf"), 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    mask = mask.unsqueeze(0).unsqueeze(0)  # shape [1, 1, N, N]
    energy = attn(g, attn_mask=mask)
    expanded_mask = mask.expand(1, 2, 3, 3)
    expected = _manual_energy(g, attn.w_k, attn.w_q, attn_mask=expanded_mask)
    assert torch.allclose(energy, expected, atol=1e-6)
