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
