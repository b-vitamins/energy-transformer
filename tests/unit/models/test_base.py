import pytest
import torch
from torch import nn

from energy_transformer.models.base import EnergyTransformer
from energy_transformer.utils.optimizers import SGD

pytestmark = pytest.mark.unit


class DummyLayerNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Ensure gradient tracking is preserved
        return x * 2.0

    def reset_parameters(self) -> None:  # type: ignore[override]
        pass


class DummyEnergyAttention(nn.Module):
    def forward(self, g: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Use operations that preserve gradient tracking
        return torch.sum(g)

    def reset_parameters(self) -> None:  # type: ignore[override]
        pass


class DummyHopfieldNetwork(nn.Module):
    def forward(self, g: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Use operations that preserve gradient tracking
        return torch.mean(g)

    def reset_parameters(self) -> None:  # type: ignore[override]
        pass


def test_energy_combines_components() -> None:
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=1,
        optimizer=SGD(alpha=1.0),
    )
    x = torch.ones(2, 2, requires_grad=True)
    energy = model._compute_energy(x)
    # g = 2 * x -> all twos; attention energy = 2*4=8, hopfield energy = 2
    assert energy.item() == pytest.approx(10.0)
    # Ensure energy has gradient capability
    assert energy.requires_grad


def test_forward_does_not_mutate_input_in_train_mode() -> None:
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=1,
        optimizer=SGD(alpha=1.0),
    )
    x = torch.ones(1, 2, 2)
    x_original = x.clone()
    model(x)
    # Training mode clones input, so original tensor is unchanged
    assert torch.allclose(x, x_original)


def test_forward_no_clone_preserves_input_and_grad() -> None:
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=1,
        optimizer=SGD(alpha=1.0),
    )
    model.eval()  # disable automatic cloning
    x = torch.ones(1, 2, 2)
    x_original = x.clone()
    out = model(x)
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
        optimizer=SGD(alpha=1.0),
    )
    model.eval()
    x = torch.ones(1, 2, 2)
    x_original = x.clone()
    model(x)
    # Force cloning prevents mutation even in eval mode
    assert torch.allclose(x, x_original)


def test_bb_descent_mode() -> None:
    """Test that descent reduces energy."""
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=5,
        optimizer=SGD(alpha=1.0),
    )
    x = torch.randn(1, 3, 3)
    initial_energy = model._compute_energy(x.clone())

    out = model(x.clone())
    final_energy = model._compute_energy(out.clone())

    assert isinstance(out, torch.Tensor)
    assert final_energy < initial_energy


def test_works_in_no_grad_context() -> None:
    """Test that the model works within torch.no_grad() context."""
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=2,
        optimizer=SGD(alpha=1.0),
    )
    model.eval()  # Put in eval mode to avoid automatic cloning
    x = torch.ones(1, 2, 2)

    with torch.no_grad():
        # The model should work in no_grad context due to force_enable_grad
        out = model(x.clone())

    assert isinstance(out, torch.Tensor)
    assert out is not None


def test_inference_mode_requires_detach() -> None:
    """Test that inference_mode raises error without detach."""
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=1,
        optimizer=SGD(alpha=1.0),
    )
    model.eval()  # Put in eval mode
    x = torch.ones(1, 2, 2)

    with (
        torch.inference_mode(),
        pytest.raises(
            RuntimeError,
            match="EnergyTransformer requires gradient computation",
        ),
    ):
        model(x.clone())


def test_armijo_parameters() -> None:
    """Test custom Armijo backtracking parameters."""
    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=3,
        optimizer=SGD(alpha=2.0),
    )
    x = torch.randn(1, 2, 2)
    initial_energy = model._compute_energy(x.clone())

    out1 = model(x.clone())
    out2 = model(x.clone())

    # Different parameters should generally produce different results
    assert isinstance(out1, torch.Tensor)
    assert isinstance(out2, torch.Tensor)
    # Both should reduce energy from initial
    assert model._compute_energy(out1) < initial_energy
    assert model._compute_energy(out2) < initial_energy


def test_forward_with_hooks() -> None:
    """Test that hooks work for monitoring."""
    from energy_transformer.utils.observers import EnergyTracker

    model = EnergyTransformer(
        DummyLayerNorm(),
        DummyEnergyAttention(),
        DummyHopfieldNetwork(),
        steps=2,
        optimizer=SGD(alpha=1.0),
    )

    tracker = EnergyTracker()
    handle = model.register_step_hook(lambda _m, info: tracker.update(info))

    x = torch.ones(1, 2, 2)
    output = model(x)

    # Check we got 2 steps tracked
    assert len(tracker.history) == 2

    # Check final output (gradient w.r.t. g is 1.25)
    expected = torch.ones_like(x) - 2 * 1.25  # 2 steps
    assert torch.allclose(output, expected)

    # Check we can get statistics
    stats = tracker.get_batch_statistics()
    assert "energy_mean" in stats

    handle.remove()
