"""Unit tests for Simplicial Hopfield Networks."""

import pytest
import torch

from energy_transformer.layers.simplicial import (
    SHNReLU,
    SHNSoftmax,
    SimplicialHopfieldNetwork,
)

pytestmark = pytest.mark.unit


def test_simplicial_hopfield_relu_energy_matches_manual() -> None:
    """Test that ReLU energy computation matches manual calculation."""
    net = SimplicialHopfieldNetwork(2, hidden_dim=2, order=3, activation="relu")
    with torch.no_grad():
        net.kernel[0].copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        net.kernel[1].copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        net.kernel[2].copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    g = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).unsqueeze(0)
    energy = net(g)

    h = torch.einsum("...nd,vdk->...vnk", g, net.kernel)
    a_v = torch.relu(h)
    sum_a = a_v.sum(dim=1)
    expected = -0.5 / 3 * (sum_a**2).sum()

    assert torch.allclose(energy, expected, atol=1e-6)


def test_simplicial_hopfield_softmax_energy_matches_manual() -> None:
    """Test that softmax energy computation matches manual calculation."""
    beta = 0.5
    net = SimplicialHopfieldNetwork(
        2, hidden_dim=2, order=3, activation="softmax", beta=beta
    )
    with torch.no_grad():
        net.kernel[0].copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        net.kernel[1].copy_(torch.tensor([[0.5, 0.5], [0.5, 0.5]]))
        net.kernel[2].copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    g = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).unsqueeze(0)
    energy = net(g)

    h = torch.einsum("...nd,vdk->...vnk", g, net.kernel)
    lse = torch.logsumexp(beta * h, dim=-1)
    expected = -(1.0 / (3 * beta)) * lse.sum()

    assert torch.allclose(energy, expected, atol=1e-6)


def test_simplicial_hopfield_order_validation() -> None:
    """Test that order < 2 raises ValueError."""
    with pytest.raises(ValueError, match="order must be >= 2"):
        SimplicialHopfieldNetwork(2, order=1)


def test_simplicial_hopfield_invalid_activation() -> None:
    """Test that invalid activation raises ValueError."""
    with pytest.raises(ValueError, match="activation must be"):
        SimplicialHopfieldNetwork(2, activation="tanh")


def test_simplicial_hopfield_default_hidden_dim() -> None:
    """Test default hidden dimension calculation."""
    net = SimplicialHopfieldNetwork(3, hidden_dim=None, hidden_ratio=2.5)
    assert net.hidden_dim == int(3 * 2.5)


def test_simplicial_hopfield_energy_is_scalar() -> None:
    """Test that energy output is a scalar."""
    net = SimplicialHopfieldNetwork(2, hidden_dim=3, order=4)
    g = torch.randn(2, 5, 2)
    energy = net(g)
    assert energy.shape == torch.Size([])


def test_simplicial_hopfield_gradients_flow() -> None:
    """Test that gradients flow through the network."""
    net = SimplicialHopfieldNetwork(2, hidden_dim=2, order=3)
    g = torch.randn(1, 3, 2, requires_grad=True)
    energy = net(g)
    energy.backward()
    assert g.grad is not None
    assert net.kernel.grad is not None


def test_simplicial_hopfield_compute_grad_relu() -> None:
    """Test gradient computation for ReLU activation."""
    net = SimplicialHopfieldNetwork(2, hidden_dim=2, order=3, activation="relu")
    with torch.no_grad():
        net.kernel.fill_(1.0)

    g = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).unsqueeze(0)
    grad = net.compute_grad(g)
    assert grad.shape == g.shape

    g_auto = g.clone().requires_grad_(True)
    energy = net(g_auto)
    energy.backward()
    assert torch.allclose(grad, g_auto.grad, atol=1e-6)


def test_simplicial_hopfield_compute_grad_softmax() -> None:
    """Test gradient computation for softmax activation."""
    beta = 0.2
    net = SimplicialHopfieldNetwork(
        2, hidden_dim=2, order=3, activation="softmax", beta=beta
    )
    with torch.no_grad():
        net.kernel.fill_(0.5)

    g = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).unsqueeze(0)
    grad = net.compute_grad(g)
    assert grad.shape == g.shape

    g_auto = g.clone().requires_grad_(True)
    energy = net(g_auto)
    energy.backward()
    assert torch.allclose(grad, g_auto.grad, atol=1e-6)


def test_simplicial_hopfield_with_bias() -> None:
    """Test SimplicalHopfieldNetwork with bias terms."""
    net = SimplicialHopfieldNetwork(
        2, hidden_dim=3, order=3, bias=True, activation="relu"
    )
    assert net.bias is not None
    assert net.bias.shape == (3, 1, 1, 3)

    g = torch.randn(2, 4, 2)
    energy = net(g)
    assert torch.isfinite(energy)


def test_simplicial_hopfield_beta_preserved_after_reset() -> None:
    """Test that beta value is preserved after parameter reset."""
    net = SimplicialHopfieldNetwork(3, order=3, activation="softmax", beta=0.5)
    before = net.beta.clone()
    net._reset_parameters()
    assert torch.allclose(net.beta, before)


def test_simplicial_hopfield_properties() -> None:
    """Test network properties."""
    net = SimplicialHopfieldNetwork(
        4, hidden_dim=6, order=4, activation="softmax", beta=0.2, bias=True
    )
    assert net.memory_dim == 6
    assert net.input_dim == 4
    assert net.simplex_order == 4
    assert net.activation_type == "softmax"
    assert not net.is_classical
    assert net.is_modern
    assert net.temperature == pytest.approx(0.2)
    expected_params = 4 * 4 * 6 + 4 * 6 + 1
    assert net.total_params == expected_params
    assert isinstance(net.device, torch.device)


def test_simplicial_hopfield_extra_repr() -> None:
    """Test string representation."""
    net = SimplicialHopfieldNetwork(
        2, hidden_dim=4, order=3, activation="softmax", beta=0.1, bias=True
    )
    rep = net.extra_repr()
    assert "embed_dim=2" in rep
    assert "hidden_dim=4" in rep
    assert "order=3" in rep
    assert "activation='softmax'" in rep
    assert "beta=0.100" in rep
    assert "bias=True" in rep


def test_simplicial_hopfield_different_orders() -> None:
    """Test network with different simplicial orders."""
    for order in [2, 3, 4, 5]:
        net = SimplicialHopfieldNetwork(3, order=order)
        g = torch.randn(2, 4, 3)
        energy = net(g)
        assert torch.isfinite(energy)
        assert net.kernel.shape == (order, 3, net.hidden_dim)


def test_simplicial_hopfield_numerical_stability() -> None:
    """Test numerical stability with extreme values."""
    net = SimplicialHopfieldNetwork(2, order=3, activation="softmax", beta=0.1)

    g_small = torch.full((1, 3, 2), 1e-8)
    energy_small = net(g_small)
    assert torch.isfinite(energy_small)

    g_large = torch.full((1, 3, 2), 100.0)
    energy_large = net(g_large)
    assert torch.isfinite(energy_large)


def test_shn_relu_factory() -> None:
    """Test SHNReLU factory class."""
    net = SHNReLU(4, order=3, hidden_ratio=2.0)
    assert net.activation == "relu"
    assert net.order == 3
    assert net.hidden_dim == 8
    assert net.beta is None

    g = torch.randn(2, 5, 4)
    energy = net(g)
    assert torch.isfinite(energy)


def test_shn_softmax_factory() -> None:
    """Test SHNSoftmax factory class."""
    net = SHNSoftmax(4, order=4, hidden_ratio=3.0, beta=0.2)
    assert net.activation == "softmax"
    assert net.order == 4
    assert net.hidden_dim == 12
    assert net.temperature == pytest.approx(0.2)

    g = torch.randn(2, 5, 4)
    energy = net(g)
    assert torch.isfinite(energy)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_simplicial_hopfield_mixed_precision(dtype: torch.dtype) -> None:
    """Test mixed precision support."""
    torch.manual_seed(42)
    net = SimplicialHopfieldNetwork(2, hidden_dim=2, order=3, activation="relu")

    g = torch.randn(1, 3, 2, dtype=dtype)
    energy = net(g)
    assert energy.dtype == dtype

    grad = net.compute_grad(g)
    assert grad.dtype == dtype


def test_simplicial_hopfield_energy_decreases() -> None:
    """Test that energy decreases with proper optimization."""
    from energy_transformer.layers import (
        EnergyLayerNorm,
        MultiheadEnergyAttention,
    )
    from energy_transformer.models import EnergyTransformer
    from energy_transformer.testing import assert_energy_decreases
    from energy_transformer.utils import SGD

    et_block = EnergyTransformer(
        layer_norm=EnergyLayerNorm(8),
        attention=MultiheadEnergyAttention(8, num_heads=2),
        hopfield=SimplicialHopfieldNetwork(8, hidden_dim=16, order=3),
        steps=10,
        optimizer=SGD(alpha=0.1),
    )

    x = torch.randn(2, 4, 8)
    assert_energy_decreases(et_block, x, tolerance=1e-6)
