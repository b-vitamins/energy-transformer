import pytest
import torch

from energy_transformer.layers.base import EnergyModule

pytestmark = pytest.mark.unit


class DummyEnergy(EnergyModule):
    def compute_energy(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()

    def compute_grad(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)


def test_forward_returns_energy() -> None:
    module = DummyEnergy()
    x = torch.tensor([1.0, 2.0])
    out = module(x)
    assert out.item() == 3.0
