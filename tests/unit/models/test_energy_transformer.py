import pytest
import torch

from energy_transformer.layers.base import (
    BaseEnergyAttention,
    BaseHopfieldNetwork,
    BaseLayerNorm,
)
from energy_transformer.models.base import EnergyTransformer, ETOutput


class DummyLayerNorm(BaseLayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


class DummyEnergyAttention(BaseEnergyAttention):
    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return g.sum()


class DummyHopfieldNetwork(BaseHopfieldNetwork):
    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return g.mean()


def test_energy_combines_components() -> None:
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=1,
        α=1.0,
    )
    x = torch.ones(2, 2)
    energy = model.energy(x)
    # g = 2 * x -> all twos; attention energy = 2*4=8, hopfield energy = 2
    assert energy.item() == pytest.approx(10.0)


def test_forward_returns_optimized_tokens_and_energy() -> None:
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=1,
        α=1.0,
    )
    x = torch.ones(1, 2, 2)
    out = model(x.clone(), return_energy=True)
    assert isinstance(out, ETOutput)
    # Gradient of energy w.r.t x is 2.5 for each element with α=1.0
    expected_tokens = x - 2.5
    assert torch.allclose(out.tokens, expected_tokens)
    assert out.final_energy is not None


def test_forward_detach_disables_grad() -> None:
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=1,
        α=1.0,
    )
    x = torch.ones(1, 2, 2)
    out = model(x, detach=True)
    assert isinstance(out, torch.Tensor)
    assert not out.requires_grad
