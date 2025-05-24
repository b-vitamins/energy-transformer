import pytest
import torch

from energy_transformer.layers.attention import (
    MultiHeadEnergyAttention,
    _chunked_logsumexp,
    _get_diag_mask,
    clear_diag_cache,
)


def manual_energy(g, w_k, w_q, beta, include_diag=True, attn_mask=None):
    k = torch.einsum("nd,hyd->nhy", g, w_k)
    q = torch.einsum("nd,hyd->nhy", g, w_q)
    a = torch.einsum("nhy,mhy->hnm", k, q)
    if not include_diag:
        diag = torch.eye(a.size(-1), dtype=torch.bool)
        a = a.masked_fill(diag, float("-inf"))
    if attn_mask is not None:
        a = a + attn_mask
    lse = torch.logsumexp(beta * a, dim=-2)
    return -(1 / beta) * lse.sum()


def test_get_diag_mask_caches_by_length():
    clear_diag_cache()
    mask1 = _get_diag_mask(torch.device("cpu"), 4)
    mask2 = _get_diag_mask(torch.device("cpu"), 4)
    assert mask1 is mask2
    assert mask1.shape == (4, 4)
    mask3 = _get_diag_mask(torch.device("cpu"), 5)
    assert mask3 is not mask1
    assert mask3.shape == (5, 5)


@pytest.mark.parametrize(
    "shape,chunk",
    [((2, 5), 10), ((3, 2048), 512)],
)
def test_chunked_logsumexp_matches_torch(shape, chunk):
    torch.manual_seed(0)
    logits = torch.randn(*shape)
    expected = torch.logsumexp(logits, dim=-1)
    result = _chunked_logsumexp(logits, dim=-1, chunk_size=chunk)
    assert torch.allclose(result, expected, atol=1e-6)


def test_attention_energy_matches_manual():
    g = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    attn = MultiHeadEnergyAttention(
        in_dim=2,
        num_heads=1,
        head_dim=1,
        beta=1.0,
        bias=False,
        chunk_size=1024,
    )
    with torch.no_grad():
        attn.w_k.fill_(1.0)
        attn.w_q.fill_(1.0)
    expected = manual_energy(g, attn.w_k, attn.w_q, attn.β)
    result = attn(g)
    assert torch.allclose(result, expected)


def test_attention_single_token_returns_zero():
    g = torch.randn(1, 2)
    attn = MultiHeadEnergyAttention(
        in_dim=2,
        num_heads=1,
        head_dim=1,
        beta=1.0,
        bias=False,
        chunk_size=1024,
    )
    energy = attn(g)
    assert energy.item() == pytest.approx(0.0)
