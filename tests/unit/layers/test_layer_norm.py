import torch
import torch.nn.functional as functional

from energy_transformer.layers.layer_norm import LayerNorm


def test_layernorm_matches_torch() -> None:
    torch.manual_seed(0)
    ln = LayerNorm(in_dim=3)
    x = torch.randn(2, 3)
    out = ln(x)

    ref = torch.nn.LayerNorm(3, eps=ln.eps)
    with torch.no_grad():
        ref.weight.fill_(functional.softplus(ln.logγ).item())
        ref.bias.copy_(ln.δ)
    expected = ref(x)
    assert torch.allclose(out, expected, atol=1e-6)


def test_energy_gradient_equals_output() -> None:
    ln = LayerNorm(in_dim=4)
    x = torch.randn(2, 4, requires_grad=True)
    g = ln(x)
    energy = ln.get_energy_lagrangian(x).sum()
    energy.backward()
    assert torch.allclose(x.grad, g, atol=1e-6)


def test_export_standard_layernorm() -> None:
    ln = LayerNorm(in_dim=2)
    ref = ln.export_standard_layernorm()
    x = torch.randn(3, 2)
    out1 = ln(x)
    out2 = ref(x)
    assert torch.allclose(out1, out2, atol=1e-6)
