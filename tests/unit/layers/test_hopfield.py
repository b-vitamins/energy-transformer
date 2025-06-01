import pytest
pytestmark = pytest.mark.unit
import torch

from energy_transformer.layers.hopfield import HopfieldNetwork


def test_hopfield_energy_matches_manual() -> None:
    net = HopfieldNetwork(in_dim=2, hidden_dim=2, multiplier=1.0)
    with torch.no_grad():
        net.ξ.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    g = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    energy = net(g)
    h = torch.matmul(g, net.ξ.t())
    expected = -0.5 * (torch.relu(h) ** 2).sum()
    assert torch.allclose(energy, expected, atol=1e-6)


def test_hopfield_custom_energy_function() -> None:
    def custom_fn(h: torch.Tensor) -> torch.Tensor:
        return h.sum()

    net = HopfieldNetwork(
        in_dim=2,
        hidden_dim=2,
        multiplier=1.0,
        energy_fn=custom_fn,
    )
    with torch.no_grad():
        net.ξ.fill_(1.0)
    g = torch.ones(2, 2)
    energy = net(g)
    expected = custom_fn(torch.matmul(g, net.ξ.t()))
    assert torch.allclose(energy, expected, atol=1e-6)


def test_hopfield_default_hidden_dim() -> None:
    net = HopfieldNetwork(in_dim=3, hidden_dim=None, multiplier=2.5)
    assert net.hidden_dim == int(3 * 2.5)


def test_hopfield_energy_is_scalar() -> None:
    net = HopfieldNetwork(in_dim=2, hidden_dim=2)
    g = torch.randn(2, 2)
    energy = net(g)
    assert energy.shape == torch.Size([])


def test_hopfield_reset_parameters_std() -> None:
    torch.manual_seed(0)
    net = HopfieldNetwork(in_dim=4, hidden_dim=6)
    # Parameter statistics after initialization
    std_expected = 1.0 / (net.in_dim * net.hidden_dim) ** 0.25
    assert net.ξ.mean().abs() < 0.1
    assert torch.isclose(net.ξ.std(), torch.tensor(std_expected), atol=5e-2)


def test_hopfield_forward_with_batch_dims() -> None:
    net = HopfieldNetwork(in_dim=2, hidden_dim=3, multiplier=1.0)
    with torch.no_grad():
        net.ξ.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]))
    g = torch.tensor(
        [
            [[1.0, 2.0], [0.0, -1.0], [3.0, 0.5]],
            [[-1.0, 1.0], [2.0, 0.0], [0.5, 0.5]],
        ],
    )
    energy = net(g)
    h = torch.matmul(g, net.ξ.t())
    expected = -0.5 * (torch.relu(h) ** 2).sum()
    assert torch.allclose(energy, expected, atol=1e-6)


def test_hopfield_energy_function_must_return_scalar() -> None:
    def bad_fn(h: torch.Tensor) -> torch.Tensor:
        return h.mean(dim=-1)  # returns tensor of shape [..., N]

    net = HopfieldNetwork(in_dim=2, hidden_dim=2, energy_fn=bad_fn)
    g = torch.randn(2, 2)
    with pytest.raises(AssertionError):
        net(g)


def test_hopfield_gradients_flow() -> None:
    net = HopfieldNetwork(in_dim=2, hidden_dim=2)
    g = torch.randn(3, 2, requires_grad=True)
    energy = net(g)
    energy.backward()
    assert g.grad is not None
    assert net.ξ.grad is not None


def test_hopfield_zero_weights_energy_zero() -> None:
    net = HopfieldNetwork(in_dim=3, hidden_dim=2)
    with torch.no_grad():
        net.ξ.zero_()
    g = torch.randn(5, 3)
    energy = net(g)
    assert torch.allclose(energy, torch.tensor(0.0))


def test_hopfield_negative_activations() -> None:
    net = HopfieldNetwork(in_dim=2, hidden_dim=2)
    with torch.no_grad():
        net.ξ.copy_(torch.tensor([[1.0, -1.0], [0.5, 0.5]]))
    g = torch.tensor([[1.0, 2.0], [-1.0, 1.0]])
    energy = net(g)
    h = torch.matmul(g, net.ξ.t())
    expected = -0.5 * (torch.relu(h) ** 2).sum()
    assert torch.allclose(energy, expected, atol=1e-6)
    assert torch.all(torch.relu(h[h < 0]) == 0)


def test_hopfield_hidden_dim_override() -> None:
    net = HopfieldNetwork(in_dim=2, hidden_dim=5, multiplier=0.1)
    assert net.hidden_dim == 5
    assert net.ξ.shape == (5, 2)
