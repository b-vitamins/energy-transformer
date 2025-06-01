import pytest
pytestmark = pytest.mark.unit
import torch

from energy_transformer.layers.base import (
    BaseEnergyAttention,
    BaseHopfieldNetwork,
    BaseLayerNorm,
)
from energy_transformer.models.base import EnergyTransformer, ETOutput


class DummyLayerNorm(BaseLayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure gradient tracking is preserved
        return x * 2.0


class DummyEnergyAttention(BaseEnergyAttention):
    def forward(self, g: torch.Tensor) -> torch.Tensor:
        # Use operations that preserve gradient tracking
        return torch.sum(g)


class DummyHopfieldNetwork(BaseHopfieldNetwork):
    def forward(self, g: torch.Tensor) -> torch.Tensor:
        # Use operations that preserve gradient tracking
        return torch.mean(g)


def test_energy_combines_components() -> None:
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=1,
        alpha=1.0,
    )
    x = torch.ones(2, 2, requires_grad=True)
    energy = model.energy(x)
    # g = 2 * x -> all twos; attention energy = 2*4=8, hopfield energy = 2
    assert energy.item() == pytest.approx(10.0)
    # Ensure energy has gradient capability
    assert energy.requires_grad


def test_forward_returns_optimized_tokens_and_energy() -> None:
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=1,
        alpha=1.0,
    )
    x = torch.ones(1, 2, 2, requires_grad=True)
    # Use SGD mode for predictable gradient descent
    out = model(x.clone(), track="energy", mode="sgd")
    assert isinstance(out, ETOutput)
    # Gradient of energy w.r.t x is 2.5 for each element with alpha=1.0
    expected_tokens = x - 2.5
    assert torch.allclose(out.tokens, expected_tokens)
    assert out.final_energy is not None


def test_forward_detach_disables_grad() -> None:
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=1,
        alpha=1.0,
    )
    x = torch.ones(1, 2, 2)
    out = model(x, detach=True, mode="sgd")
    assert isinstance(out, torch.Tensor)
    assert not out.requires_grad


def test_forward_returns_energy_and_trajectory() -> None:
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=2,
        alpha=1.0,
    )
    x = torch.ones(1, 2, 2)
    # Use SGD mode for predictable step sizes
    out = model(x.clone(), track="both", mode="sgd")
    assert isinstance(out, ETOutput)
    # After 2 steps with gradient 2.5 / element
    # alpha=1.0: 1 - 2.5 - 2.5 = -4
    assert torch.allclose(out.tokens, torch.full_like(x, -4.0))
    assert out.final_energy is not None
    assert out.trajectory is not None
    assert out.trajectory.shape[0] == 2  # 2 steps
    # Initial energy is 10.0, then after each step it changes
    assert out.trajectory[0].item() == pytest.approx(10.0)


def test_forward_returns_trajectory_only() -> None:
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=3,
        alpha=0.5,
    )
    x = torch.ones(1, 2, 2)
    out = model(x.clone(), track="trajectory", mode="sgd")
    assert isinstance(out, ETOutput)
    assert out.trajectory is not None
    assert out.trajectory.shape[0] == 3  # 3 steps
    assert out.final_energy is None  # Not requested


def test_forward_does_not_mutate_input_in_train_mode() -> None:
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=1,
        alpha=1.0,
    )
    x = torch.ones(1, 2, 2)
    x_original = x.clone()
    model(x, mode="sgd")
    # Training mode clones input, so original tensor is unchanged
    assert torch.allclose(x, x_original)


def test_forward_no_clone_preserves_input_and_grad() -> None:
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=1,
        alpha=1.0,
    )
    model.eval()  # disable automatic cloning
    x = torch.ones(1, 2, 2)
    x_original = x.clone()
    out = model(x, force_clone=False, mode="sgd")
    # The function creates internal clones, so original tensor is unchanged
    assert torch.allclose(x, x_original)
    assert isinstance(out, torch.Tensor)
    assert out.requires_grad


def test_forward_force_clone_preserves_input() -> None:
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=1,
        alpha=1.0,
    )
    model.eval()
    x = torch.ones(1, 2, 2)
    x_original = x.clone()
    model(x, force_clone=True, mode="sgd")
    # Force cloning prevents mutation even in eval mode
    assert torch.allclose(x, x_original)


def test_bb_descent_mode() -> None:
    """Test Barzilai-Borwein descent mode."""
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=5,
        alpha=1.0,
    )
    x = torch.randn(1, 3, 3)
    initial_energy = model.energy(x.clone())

    out_bb = model(x.clone(), mode="bb", track="energy")
    out_sgd = model(x.clone(), mode="sgd", track="energy")

    assert isinstance(out_bb, ETOutput)
    assert isinstance(out_sgd, ETOutput)
    # BB and SGD should produce different results
    assert not torch.allclose(out_bb.tokens, out_sgd.tokens)
    # Both should reduce energy
    assert out_bb.final_energy < initial_energy
    assert out_sgd.final_energy < initial_energy


def test_works_in_no_grad_context() -> None:
    """Test that the model works within torch.no_grad() context."""
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=2,
        alpha=1.0,
    )
    model.eval()  # Put in eval mode to avoid automatic cloning
    x = torch.ones(1, 2, 2)

    with torch.no_grad():
        # The model should work in no_grad context due to force_enable_grad
        out = model(x.clone(), track="energy")

    assert isinstance(out, ETOutput)
    assert out.tokens is not None
    assert out.final_energy is not None


def test_inference_mode_requires_detach() -> None:
    """Test that inference_mode raises error without detach."""
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=1,
        alpha=1.0,
    )
    model.eval()  # Put in eval mode
    x = torch.ones(1, 2, 2)

    with torch.inference_mode():
        # Should raise error without detach
        with pytest.raises(
            RuntimeError,
            match="EnergyTransformer requires gradient computation",
        ):
            model(x.clone())

        # Should work with detach
        out = model(x.clone(), detach=True)
        assert isinstance(out, torch.Tensor)
        assert not out.requires_grad


def test_armijo_parameters() -> None:
    """Test custom Armijo backtracking parameters."""
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=3,
        alpha=2.0,
    )
    x = torch.randn(1, 2, 2)
    initial_energy = model.energy(x.clone())

    out1 = model(
        x.clone(),
        mode="bb",
        track="energy",
        armijo_gamma=0.3,
        armijo_max_iter=2,
    )
    out2 = model(
        x.clone(),
        mode="bb",
        track="energy",
        armijo_gamma=0.7,
        armijo_max_iter=6,
    )

    # Different parameters should generally produce different results
    assert isinstance(out1, ETOutput)
    assert isinstance(out2, ETOutput)
    # Both should reduce energy from initial
    assert out1.final_energy < initial_energy
    assert out2.final_energy < initial_energy
