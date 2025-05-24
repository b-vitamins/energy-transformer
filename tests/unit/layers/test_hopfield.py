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
        in_dim=2, hidden_dim=2, multiplier=1.0, energy_fn=custom_fn
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
