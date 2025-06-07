import pytest
import torch

from energy_transformer.layers.hopfield import HopfieldNetwork

pytestmark = pytest.mark.unit


def test_energy_relu_matches_manual() -> None:
    net = HopfieldNetwork(2, hidden_dim=2)
    with torch.no_grad():
        net.kernel.copy_(torch.eye(2))
    g = torch.tensor([[1.0, 2.0]]).unsqueeze(0)
    energy = net(g)
    h = torch.einsum("bnd,dk->bnk", g, net.kernel)
    expected = -0.5 * (torch.relu(h) ** 2).sum()
    assert torch.allclose(energy, expected)


def test_energy_softmax_matches_manual() -> None:
    net = HopfieldNetwork(2, hidden_dim=2, activation="softmax", beta=0.5)
    with torch.no_grad():
        net.kernel.copy_(torch.eye(2))
    g = torch.tensor([[0.5, 1.0]]).unsqueeze(0)
    energy = net(g)
    h = torch.einsum("bnd,dk->bnk", g, net.kernel) * net.beta
    lse = torch.logsumexp(h, dim=-1)
    expected = -(1.0 / net.beta) * lse.sum() / (g.size(0) * g.size(1))
    assert torch.allclose(energy, expected)


def test_compute_grad_relu_matches_autograd() -> None:
    net = HopfieldNetwork(2, hidden_dim=2)
    g = torch.tensor([[1.0, 2.0]], requires_grad=True).unsqueeze(0)
    energy = net(g)
    grad_auto = torch.autograd.grad(energy, g)[0]
    grad_manual = net.compute_grad(g.detach())
    torch.testing.assert_close(grad_manual, grad_auto)


def test_compute_grad_softmax_matches_autograd() -> None:
    net = HopfieldNetwork(2, hidden_dim=2, activation="softmax", beta=0.5)
    g = torch.tensor([[0.5, 1.0]], requires_grad=True).unsqueeze(0)
    energy = net(g)
    grad_auto = torch.autograd.grad(energy, g)[0]
    grad_manual = net.compute_grad(g.detach())
    torch.testing.assert_close(grad_manual, grad_auto)


def test_energy_no_bias() -> None:
    net = HopfieldNetwork(2, hidden_dim=2, bias=False)
    g = torch.randn(1, 1, 2)
    energy = net(g)
    assert energy.ndim == 0
