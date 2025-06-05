import math

import pytest
import torch

from energy_transformer.layers.attention import MultiheadEnergyAttention

pytestmark = pytest.mark.unit


def _manual_energy(
    g: torch.Tensor,
    w_k: torch.Tensor,
    w_q: torch.Tensor,
    beta: float | torch.Tensor = 1.0,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    if isinstance(beta, torch.Tensor):
        beta_tensor = beta
    else:
        beta_tensor = torch.full(
            (w_k.shape[0],), beta, device=g.device, dtype=g.dtype
        )
    compute_dtype = (
        torch.float32 if g.dtype in {torch.float16, torch.bfloat16} else g.dtype
    )
    k = torch.einsum(
        "bse,hde->bshd", g.to(compute_dtype), w_k.to(compute_dtype)
    )
    q = torch.einsum(
        "bse,hde->bshd", g.to(compute_dtype), w_q.to(compute_dtype)
    )
    scores = torch.einsum(
        "bshd,bthd,h->bhst", q, k, beta_tensor.to(compute_dtype)
    )
    if g.shape[1] > 1:
        diag = torch.eye(g.shape[1], device=g.device, dtype=torch.bool)
        scores = scores.masked_fill(
            diag.unsqueeze(0).unsqueeze(0), float("-inf")
        )
    if is_causal:
        seq_len = scores.shape[-1]
        causal = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=g.device), 1
        )
        scores = scores + causal
    if attn_mask is not None:
        scores = scores + attn_mask
    lse = torch.logsumexp(scores, dim=-1)
    beta_view = (
        beta_tensor.view(1, -1, 1) if isinstance(beta, torch.Tensor) else beta
    )
    return (-(lse / beta_view).sum()).to(g.dtype)


def test_attention_energy_matches_manual() -> None:
    attn = MultiheadEnergyAttention(
        embed_dim=4,
        num_heads=1,
        beta=1.0,
    )
    with torch.no_grad():
        attn.k_proj_weight.copy_(torch.eye(4).unsqueeze(0))
        attn.q_proj_weight.copy_(torch.eye(4).unsqueeze(0))
    g = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    energy = attn(g)
    expected = _manual_energy(g, attn.k_proj_weight, attn.q_proj_weight)
    assert torch.allclose(energy, expected, atol=1e-6)


def test_attention_excludes_diagonal() -> None:
    attn = MultiheadEnergyAttention(
        embed_dim=1,
        num_heads=1,
        beta=1.0,
    )
    with torch.no_grad():
        attn.k_proj_weight.fill_(1.0)
        attn.q_proj_weight.fill_(1.0)
    g = torch.ones(1, 2, 1)
    mask = torch.tensor([[[0.0, float("-inf")], [0.0, 0.0]]])
    e1 = attn(g)
    e2 = attn(g, attn_mask=mask)
    exp1 = _manual_energy(g, attn.k_proj_weight, attn.q_proj_weight)
    exp2 = _manual_energy(
        g, attn.k_proj_weight, attn.q_proj_weight, attn_mask=mask
    )
    assert torch.allclose(e1, exp1, atol=1e-6)
    assert torch.allclose(e2, exp2, atol=1e-6)


def test_attention_applies_mask() -> None:
    attn = MultiheadEnergyAttention(
        embed_dim=1,
        num_heads=1,
        beta=1.0,
    )
    with torch.no_grad():
        attn.k_proj_weight.fill_(1.0)
        attn.q_proj_weight.fill_(1.0)
    g = torch.ones(1, 2, 1)
    mask = torch.tensor([[[0.0, float("-inf")], [0.0, 0.0]]])
    energy = attn(g, attn_mask=mask)
    expected = _manual_energy(
        g, attn.k_proj_weight, attn.q_proj_weight, attn_mask=mask
    )
    assert torch.allclose(energy, expected, atol=1e-6)


def test_attention_single_token_zero() -> None:
    attn = MultiheadEnergyAttention(embed_dim=3, num_heads=1)
    g = torch.randn(1, 1, 3)
    energy = attn(g)
    assert torch.allclose(energy, torch.tensor(0.0))


def test_attention_parameter_shapes() -> None:
    attn = MultiheadEnergyAttention(
        embed_dim=4,
        num_heads=2,
    )
    assert attn.k_proj_weight.shape == (2, 2, 4)
    assert attn.q_proj_weight.shape == (2, 2, 4)


def test_attention_default_beta_matches_manual() -> None:
    head_dim = 4
    attn = MultiheadEnergyAttention(
        embed_dim=4,
        num_heads=1,
        beta=None,
    )
    assert attn.beta == pytest.approx(1.0 / math.sqrt(head_dim))
    with torch.no_grad():
        attn.k_proj_weight.zero_()
        attn.q_proj_weight.zero_()
        attn.k_proj_weight[0, :2, :2].copy_(torch.eye(2))
        attn.q_proj_weight[0, :2, :2].copy_(torch.eye(2))
    g = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    energy = attn(g)
    expected = _manual_energy(
        g, attn.k_proj_weight, attn.q_proj_weight, beta=attn.beta
    )
    assert torch.allclose(energy, expected, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_attention_mixed_precision(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    attn = MultiheadEnergyAttention(
        embed_dim=1,
        num_heads=1,
        beta=1.0,
    )
    with torch.no_grad():
        attn.k_proj_weight.fill_(1.0)
        attn.q_proj_weight.fill_(1.0)
    g = torch.ones(1, 2, 1, dtype=dtype)
    energy = attn(g)
    expected = _manual_energy(
        g, attn.k_proj_weight.to(dtype), attn.q_proj_weight.to(dtype)
    )
    assert energy.dtype == dtype

    # bfloat16 has lower precision than float16
    atol = 5e-3 if dtype == torch.bfloat16 else 1e-3
    assert torch.allclose(energy, expected, atol=atol)


def test_attention_mask_broadcast() -> None:
    attn = MultiheadEnergyAttention(
        embed_dim=2,
        num_heads=2,
        beta=1.0,
    )
    with torch.no_grad():
        attn.k_proj_weight.fill_(1.0)
        attn.q_proj_weight.fill_(1.0)
    g = torch.ones(1, 3, 1)
    mask = torch.tensor(
        [[0.0, float("-inf"), 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    )
    mask = mask.unsqueeze(0)  # shape [1, N, N]
    energy = attn(g, attn_mask=mask)
    expanded_mask = mask.expand(2, 3, 3)
    expected = _manual_energy(
        g, attn.k_proj_weight, attn.q_proj_weight, attn_mask=expanded_mask
    )
    assert torch.allclose(energy, expected, atol=1e-6)
