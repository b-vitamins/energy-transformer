"""Utilities for testing Energy Transformer models."""

from torch import Tensor

from energy_transformer.models import EnergyTransformer


def assert_energy_decreases(
    model: EnergyTransformer,
    x: Tensor,
    tolerance: float = 1e-6,
) -> None:
    """Assert that energy decreases during forward pass.

    Useful for unit tests to verify correct implementation.
    """
    energies: list[float] = []
    model.register_step_hook(
        lambda _m, info: energies.append(info.total_energy.item())
    )
    _ = model(x)

    for i in range(1, len(energies)):
        assert energies[i] <= energies[i - 1] + tolerance, (
            f"Energy increased at step {i}: {energies[i - 1]} -> {energies[i]}"
        )
