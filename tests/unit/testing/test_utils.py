import pytest
import torch

from energy_transformer.models.base import EnergyTransformer
from energy_transformer.testing import assert_energy_decreases
from energy_transformer.utils.optimizers import SGD


class DummyLayerNorm(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x * 2.0

    def reset_parameters(self) -> None:  # type: ignore[override]
        pass


class DummyEnergyAttention(torch.nn.Module):
    def forward(self, g: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.sum(g)

    def reset_parameters(self) -> None:  # type: ignore[override]
        pass


class DummyHopfieldNetwork(torch.nn.Module):
    def forward(self, g: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.mean(g)

    def reset_parameters(self) -> None:  # type: ignore[override]
        pass


@pytest.mark.unit
def test_assert_energy_decreases() -> None:
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=2,
        optimizer=SGD(alpha=1.0),
    )
    x = torch.randn(1, 2, 2)
    assert_energy_decreases(model, x)
