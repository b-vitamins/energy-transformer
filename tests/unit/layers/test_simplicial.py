import pytest
import torch

from energy_transformer.layers.simplicial import SimplicialHopfieldNetwork

pytestmark = pytest.mark.unit


def test_energy_relu_scalar() -> None:
    net = SimplicialHopfieldNetwork(2, hidden_dim=2, order=3)
    g = torch.randn(1, 4, 2)
    energy = net(g)
    assert energy.ndim == 0


def test_compute_grad_matches_autograd() -> None:
    net = SimplicialHopfieldNetwork(2, hidden_dim=2, order=3, activation="relu")
    x = torch.randn(1, 3, 2, requires_grad=True)
    grad = net.compute_grad(x)
    energy = net(x)
    energy.backward()
    assert torch.allclose(x.grad, grad, atol=1e-6)


def test_init_invalid_order() -> None:
    with pytest.raises(ValueError, match="order must be >= 2"):
        SimplicialHopfieldNetwork(2, order=1)


def test_energy_softmax_scalar() -> None:
    net = SimplicialHopfieldNetwork(
        2, hidden_dim=2, order=3, activation="softmax", beta=0.5
    )
    g = torch.randn(1, 3, 2)
    energy = net(g)
    assert energy.ndim == 0


def test_compute_grad_softmax_matches_autograd() -> None:
    net = SimplicialHopfieldNetwork(
        2, hidden_dim=2, order=3, activation="softmax", beta=0.5
    )
    x = torch.randn(1, 3, 2, requires_grad=True)
    grad = net.compute_grad(x)
    energy = net(x)
    energy.backward()
    assert torch.allclose(x.grad, grad, atol=1e-6)
