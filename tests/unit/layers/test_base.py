import pytest
import torch
from torch import nn

from energy_transformer.layers.base import _validate_scalar_energy

pytestmark = pytest.mark.unit


class DummyLayerNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x * 2

    def reset_parameters(self) -> None:  # type: ignore[override]
        pass


class DummyEnergyAttention(nn.Module):
    def forward(self, g: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return g.sum()

    def reset_parameters(self) -> None:  # type: ignore[override]
        pass


class DummyHopfieldNetwork(nn.Module):
    def forward(self, g: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return g.mean()

    def reset_parameters(self) -> None:  # type: ignore[override]
        pass


def test_validate_scalar_energy_accepts_scalar() -> None:
    energy = torch.tensor(3.14)
    # Should not raise when energy is scalar
    _validate_scalar_energy(energy, "test")


def test_validate_scalar_energy_rejects_tensor() -> None:
    energy = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="must return scalar energy tensor"):
        _validate_scalar_energy(energy, "mycomp")


def test_default_reset_parameters_return_none() -> None:
    assert DummyLayerNorm().reset_parameters() is None
    assert DummyEnergyAttention().reset_parameters() is None
    assert DummyHopfieldNetwork().reset_parameters() is None
