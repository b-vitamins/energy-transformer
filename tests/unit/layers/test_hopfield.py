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
