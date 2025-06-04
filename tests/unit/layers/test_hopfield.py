import pytest
import torch

from energy_transformer.layers.hopfield import HopfieldNetwork

pytestmark = pytest.mark.unit


def test_hopfield_energy_matches_manual() -> None:
    net = HopfieldNetwork(2, hidden_dim=2)
    with torch.no_grad():
        net.kernel.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    g = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    energy = net(g)
    h = torch.matmul(g, net.kernel)
    expected = -0.5 * (torch.relu(h) ** 2).sum()
    assert torch.allclose(energy, expected, atol=1e-6)


def test_hopfield_softmax_energy_matches_manual() -> None:
    beta = 0.5
    net = HopfieldNetwork(2, hidden_dim=2, activation="softmax", beta=beta)
    with torch.no_grad():
        net.kernel.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    g = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    energy = net(g)
    h = torch.matmul(g, net.kernel)
    expected = -(1.0 / beta) * torch.logsumexp(beta * h, dim=-1).sum()
    assert torch.allclose(energy, expected, atol=1e-6)


def test_hopfield_default_hidden_dim() -> None:
    net = HopfieldNetwork(3, hidden_dim=None, hidden_ratio=2.5)
    assert net.hidden_dim == int(3 * 2.5)


def test_hopfield_energy_is_scalar() -> None:
    net = HopfieldNetwork(2, hidden_dim=2)
    g = torch.randn(2, 2)
    energy = net(g)
    assert energy.shape == torch.Size([])


def test_hopfield_reset_parameters_std() -> None:
    torch.manual_seed(0)
    init_std = 0.03
    net = HopfieldNetwork(4, hidden_dim=6, init_std=init_std)
    assert net.kernel.mean().abs() < 0.1
    assert torch.isclose(net.kernel.std(), torch.tensor(init_std), atol=5e-2)


def test_hopfield_forward_with_batch_dims() -> None:
    net = HopfieldNetwork(2, hidden_dim=3)
    with torch.no_grad():
        net.kernel.copy_(torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]]))
    g = torch.tensor(
        [
            [[1.0, 2.0], [0.0, -1.0], [3.0, 0.5]],
            [[-1.0, 1.0], [2.0, 0.0], [0.5, 0.5]],
        ],
    )
    energy = net(g)
    h = torch.matmul(g, net.kernel)
    expected = -0.5 * (torch.relu(h) ** 2).sum()
    assert torch.allclose(energy, expected, atol=1e-6)


def test_hopfield_gradients_flow() -> None:
    net = HopfieldNetwork(2, hidden_dim=2)
    g = torch.randn(3, 2, requires_grad=True)
    energy = net(g)
    energy.backward()
    assert g.grad is not None
    assert net.kernel.grad is not None


def test_hopfield_zero_weights_energy_zero() -> None:
    net = HopfieldNetwork(3, hidden_dim=2)
    with torch.no_grad():
        net.kernel.zero_()
    g = torch.randn(5, 3)
    energy = net(g)
    assert torch.allclose(energy, torch.tensor(0.0))


def test_hopfield_negative_activations() -> None:
    net = HopfieldNetwork(2, hidden_dim=2)
    with torch.no_grad():
        net.kernel.copy_(torch.tensor([[1.0, -1.0], [0.5, 0.5]]))
    g = torch.tensor([[1.0, 2.0], [-1.0, 1.0]])
    energy = net(g)
    h = torch.matmul(g, net.kernel)
    expected = -0.5 * (torch.relu(h) ** 2).sum()
    assert torch.allclose(energy, expected, atol=1e-6)
    assert torch.all(torch.relu(h[h < 0]) == 0)


def test_hopfield_hidden_dim_override() -> None:
    net = HopfieldNetwork(2, hidden_dim=5, hidden_ratio=0.1)
    assert net.hidden_dim == 5
    assert net.kernel.shape == (2, 5)
