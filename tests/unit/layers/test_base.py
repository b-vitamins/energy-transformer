import pytest
import torch

from energy_transformer.layers.base import _validate_scalar_energy, BaseLayerNorm, BaseEnergyAttention, BaseHopfieldNetwork


class DummyLayerNorm(BaseLayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


class DummyEnergyAttention(BaseEnergyAttention):
    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return g.sum()


class DummyHopfieldNetwork(BaseHopfieldNetwork):
    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return g.mean()


def test_validate_scalar_energy_accepts_scalar():
    energy = torch.tensor(3.14)
    # Should not raise when energy is scalar
    _validate_scalar_energy(energy, "test")


def test_validate_scalar_energy_rejects_tensor():
    energy = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="must return scalar energy tensor"):
        _validate_scalar_energy(energy, "mycomp")


def test_default_reset_parameters_return_none():
    assert DummyLayerNorm().reset_parameters() is None
    assert DummyEnergyAttention().reset_parameters() is None
    assert DummyHopfieldNetwork().reset_parameters() is None


