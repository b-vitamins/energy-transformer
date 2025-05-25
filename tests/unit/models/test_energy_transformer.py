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


class MaskAwareEnergyAttention(BaseEnergyAttention):
    def forward(self, g: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Energy is sum of masked tokens; verifies mask is forwarded correctly
        assert attn_mask.shape == g.shape
        return (g * attn_mask).sum()


def test_energy_uses_mask_in_attention() -> None:
    model = EnergyTransformer(
        DummyLayerNorm(),
        MaskAwareEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=1,
        α=1.0,
    )
    x = torch.ones(1, 2, 2)
    mask = torch.full_like(x, 0.5)
    energy = model.energy(x, energy_mask=mask)
    # g = 2 * x -> all twos; attention energy = 2 * 0.5 * 4 = 4
    # hopfield energy = 2
    assert energy.item() == pytest.approx(6.0)


def test_forward_returns_energy_and_trajectory() -> None:
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=2,
        α=1.0,
    )
    x = torch.ones(1, 2, 2)
    out = model(x.clone(), return_energy=True, return_trajectory=True)
    assert isinstance(out, ETOutput)
    # After two steps with gradient 2.5 per element, tokens are 1 - 2.5 - 2.5 = -4
    assert torch.allclose(out.tokens, torch.full_like(x, -4.0))
    assert out.final_energy is not None
    assert torch.allclose(out.trajectory, torch.tensor([10.0, -15.0]))
    assert out.final_energy.item() == pytest.approx(-40.0)


def test_forward_does_not_mutate_input_in_train_mode() -> None:
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=1,
        α=1.0,
    )
    x = torch.ones(1, 2, 2)
    x_clone = x.clone()
    model(x)
    # Training mode clones input, so original tensor is unchanged
    assert torch.allclose(x, x_clone)


def test_forward_no_clone_preserves_input_and_grad() -> None:
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=1,
        α=1.0,
    )
    model.eval()  # disable automatic cloning
    x = torch.ones(1, 2, 2)
    x_clone = x.clone()
    out = model(x, force_clone=False)
    # Even without cloning, original tensor is unchanged and output keeps grads
    assert torch.allclose(x, x_clone)
    assert isinstance(out, torch.Tensor)
    assert out.requires_grad


def test_forward_force_clone_preserves_input() -> None:
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=1,
        α=1.0,
    )
    model.eval()
    x = torch.ones(1, 2, 2)
    x_clone = x.clone()
    model(x, force_clone=True)
    # Force cloning prevents mutation even in eval mode
    assert torch.allclose(x, x_clone)
