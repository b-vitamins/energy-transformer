import pytest
import torch
from torch.nn import functional as F  # noqa: N812

from energy_transformer.layers.layer_norm import EnergyLayerNorm

pytestmark = pytest.mark.unit


def test_layernorm_matches_torch() -> None:
    torch.manual_seed(0)
    ln = EnergyLayerNorm(3)
    x = torch.randn(2, 3)
    out = ln(x)

    ref = torch.nn.LayerNorm(3, eps=ln.eps)
    with torch.no_grad():
        ref.weight.fill_(F.softplus(ln.log_gamma).item())
        ref.bias.copy_(ln.delta)
    expected = ref(x)
    assert torch.allclose(out, expected, atol=1e-6)


def test_energy_gradient_equals_output() -> None:
    ln = EnergyLayerNorm(4)
    x = torch.randn(2, 4, requires_grad=True)
    g = ln(x)
    energy = ln.compute_energy(x).sum()
    energy.backward()
    assert torch.allclose(x.grad, g, atol=1e-6)


def test_parameter_initialization() -> None:
    ln = EnergyLayerNorm(4)
    gamma = F.softplus(ln.log_gamma).item()
    assert gamma == pytest.approx(1.0)
    assert torch.all(ln.delta == 0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mixed_precision_matches_float32(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    ln = EnergyLayerNorm(3)
    x = torch.randn(2, 3)
    out_fp32 = ln(x)

    x_mp = x.to(dtype)
    out_mp = ln(x_mp)
    assert out_mp.dtype == torch.float32

    # bfloat16 has lower precision (8-bit mantissa) than float16 (10-bit)
    atol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    assert torch.allclose(out_mp, out_fp32, atol=atol)


def test_energy_lagrangian_manual_computation() -> None:
    ln = EnergyLayerNorm(4)
    with torch.no_grad():
        ln.log_gamma.fill_(0.7)
        ln.delta.copy_(torch.tensor([0.1, -0.2, 0.3, 0.4]))

    x = torch.randn(3, 4)
    energy = ln.compute_energy(x)

    gamma = F.softplus(ln.log_gamma)
    var = torch.var(x, dim=-1, unbiased=False)
    expected = ln.D * gamma * torch.sqrt(var + ln.eps) + (ln.delta * x).sum(
        dim=-1
    )
    assert torch.allclose(energy, expected, atol=1e-6)


def test_layernorm_additional_dims_matches_torch() -> None:
    torch.manual_seed(0)
    ln = EnergyLayerNorm(3)
    x = torch.randn(2, 4, 3)
    out = ln(x)

    ref = torch.nn.LayerNorm(3, eps=ln.eps)
    with torch.no_grad():
        ref.weight.fill_(F.softplus(ln.log_gamma).item())
        ref.bias.copy_(ln.delta)
    expected = ref(x)
    assert torch.allclose(out, expected, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_energy_lagrangian_mixed_precision(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    ln = EnergyLayerNorm(2)
    x = torch.randn(4, 2)
    energy_fp32 = ln.compute_energy(x)

    x_mp = x.to(dtype)
    energy_mp = ln.compute_energy(x_mp)
    assert energy_mp.dtype == torch.float32

    atol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    assert torch.allclose(energy_mp, energy_fp32, atol=atol)
