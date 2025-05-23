"""Unit tests for the energy-based Hopfield Network implementation."""

from typing import Any

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from energy_transformer.layers import HopfieldNetwork
from energy_transformer.layers.base import BaseHopfieldNetwork
from energy_transformer.layers.hopfield import ActivationFunction


class TestHopfieldNetwork:
    """Test suite for HopfieldNetwork implementation."""

    def test_initialization(self, device: torch.device) -> None:
        """Test that HopfieldNetwork initializes correctly."""
        in_dim = 768
        hidden_dim = 2048

        hopfield = HopfieldNetwork(in_dim=in_dim, hidden_dim=hidden_dim).to(
            device
        )

        # Check that it's an instance of BaseHopfieldNetwork
        assert isinstance(hopfield, BaseHopfieldNetwork)

        # Check parameter shapes
        assert hopfield.ξ.shape == (hidden_dim, in_dim)  # Memory patterns

        # Check that default energy function exists
        assert callable(hopfield.energy_fn)

    def test_energy_output_scalar(self, device: torch.device) -> None:
        """Test that energy output is a scalar."""
        in_dim = 512
        hidden_dim = 1024

        hopfield = HopfieldNetwork(in_dim=in_dim, hidden_dim=hidden_dim).to(
            device
        )

        # Test with various batch sizes and sequence lengths
        test_configs = [
            (1, 10),  # (batch_size, seq_len)
            (2, 8),
            (3, 16),
        ]

        for batch_size, seq_len in test_configs:
            x = torch.randn(batch_size, seq_len, in_dim, device=device)
            energy = hopfield(x)

            # Check that output is a scalar
            assert energy.ndim == 0, (
                f"Energy should be a scalar, got shape {energy.shape}"
            )
            assert energy.numel() == 1, (
                f"Energy should contain a single value, got {energy.numel()}"
            )

    def test_reset_parameters(self, device: torch.device) -> None:
        """Test that reset_parameters properly initializes parameters."""
        in_dim = 512
        hidden_dim = 1024

        hopfield = HopfieldNetwork(in_dim=in_dim, hidden_dim=hidden_dim).to(
            device
        )

        # Store original parameters
        with torch.no_grad():
            original_ξ = hopfield.ξ.clone()

            # Modify parameters
            hopfield.ξ.fill_(10.0)

        # Reset parameters
        hopfield.reset_parameters()
        a, b = hopfield.ξ, torch.full_like(hopfield.ξ, 10.0)
        # Check that parameters are reset and different from original
        assert not torch.allclose(a, b)

        # Parameters should be different from original due to random init
        assert not torch.allclose(hopfield.ξ, original_ξ)

    def test_default_energy_function(self, device: torch.device) -> None:
        """Test the default energy function (ReLU squared)."""
        # Create a test tensor
        h = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)

        # The default energy function is -0.5 * (F.relu(h) ** 2).sum()
        # ReLU(h) = [0, 0, 0, 1, 2]
        # ReLU(h)^2 = [0, 0, 0, 1, 4]
        # -0.5 * sum = -0.5 * 5 = -2.5

        # Create a hopfield network
        hopfield = HopfieldNetwork(in_dim=5, hidden_dim=1).to(device)
        energy = hopfield.energy_fn(h)

        assert energy.item() == pytest.approx(-2.5)

    def test_custom_energy_function(self, device: torch.device) -> None:
        """Test using a custom energy function."""
        in_dim = 64
        hidden_dim = 128

        # Define a simple custom energy function for testing
        def custom_energy_fn(h: torch.Tensor) -> torch.Tensor:
            # Simple quadratic energy: -0.5 * sum(h^2)
            return -0.5 * (h**2).sum()

        # Create two networks: one with default, one with custom
        hopfield_default = HopfieldNetwork(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            activation=ActivationFunction.RELU,  # Default
            bias=False,
        ).to(device)

        hopfield_custom = HopfieldNetwork(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            activation=ActivationFunction.CUSTOM,
            energy_fn=custom_energy_fn,
            bias=False,
        ).to(device)

        # Use the same weights for both networks
        with torch.no_grad():
            hopfield_custom.ξ.copy_(hopfield_default.ξ)

        # Test with the same input
        x = torch.randn(2, 4, in_dim, device=device)

        energy_default = hopfield_default(x)
        energy_custom = hopfield_custom(x)

        # They should be different because they use different energy functions
        assert not torch.allclose(energy_default, energy_custom, atol=1e-5), (
            "Custom and default energy functions should produce different "
            "results"
        )

        # Test that the custom energy function is actually being used
        # by calling it directly with the same computation
        with torch.no_grad():
            h = F.linear(x, hopfield_custom.ξ)
            direct_energy = custom_energy_fn(h)

        assert torch.allclose(energy_custom, direct_energy, atol=1e-5), (
            "Custom energy function should match direct computation"
        )

    def test_gradient_flow(self, device: torch.device) -> None:
        """Test that gradients flow properly through the Hopfield network."""
        in_dim = 32
        hidden_dim = 64
        batch_size = 2
        seq_len = 3

        hopfield = HopfieldNetwork(in_dim=in_dim, hidden_dim=hidden_dim).to(
            device
        )
        x = torch.randn(
            batch_size, seq_len, in_dim, device=device, requires_grad=True
        )

        # Forward pass
        energy = hopfield(x)

        # Backward pass
        energy.backward()

        # Check that gradients are not None
        assert x.grad is not None, "Input gradients should not be None"
        assert hopfield.ξ.grad is not None, (
            "Memory pattern gradients should not be None"
        )

        # Check gradient shapes
        assert x.grad.shape == x.shape
        assert hopfield.ξ.grad.shape == hopfield.ξ.shape

    def test_energy_minimization(
        self, device: torch.device, assert_decreasing: Any
    ) -> None:
        """Test that energy decreases as input moves toward memory patterns."""
        in_dim = 16
        hidden_dim = 8

        # Create a Hopfield network
        hopfield = HopfieldNetwork(in_dim=in_dim, hidden_dim=hidden_dim).to(
            device
        )

        # Fix memory patterns to known values
        with torch.no_grad():
            # Create orthogonal memory patterns
            hopfield.ξ.zero_()
            # Pattern 1: [1, 0, 0, ...]
            hopfield.ξ[0, 0] = 1.0
            # Pattern 2: [0, 1, 0, ...]
            hopfield.ξ[1, 1] = 1.0
            # And so on...
            for i in range(2, min(hidden_dim, in_dim)):
                hopfield.ξ[i, i] = 1.0

        # Create random input
        x = torch.randn(1, 1, in_dim, device=device, requires_grad=True)

        # Gradient descent to minimize energy
        optimizer = torch.optim.SGD([x], lr=0.1)

        # Track energies during optimization
        energies = []

        # Run a few steps of gradient descent
        for _ in range(10):
            optimizer.zero_grad()
            energy = hopfield(x)
            energies.append(energy.item())
            energy.backward()
            optimizer.step()

        # Energies should be decreasing (non-increasing)
        assert_decreasing(
            energies, msg="Energy should decrease during gradient descent"
        )

    def test_batch_independence(self, device: torch.device) -> None:
        """Test energy calculations are independent across batch dimension."""
        in_dim = 32
        hidden_dim = 64
        seq_len = 4

        hopfield = HopfieldNetwork(in_dim=in_dim, hidden_dim=hidden_dim).to(
            device
        )

        # Create a batch where the first and third examples are identical
        # but the second is different
        x1 = torch.randn(1, seq_len, in_dim, device=device)
        x2 = torch.randn(1, seq_len, in_dim, device=device)

        x_batch = torch.cat([x1, x2, x1], dim=0)  # Shape [3, seq_len, in_dim]

        # Calculate energy for batch
        energy_batch = hopfield(x_batch)

        # Calculate energy for individual examples
        energy_1 = hopfield(x1)
        energy_2 = hopfield(x2)
        energy_3 = hopfield(x1)  # Same as energy_1

        # Batch energy should be the sum of individual energies
        expected_energy = energy_1 + energy_2 + energy_3
        assert energy_batch.item() == pytest.approx(expected_energy.item())

    def test_different_input_shapes(self, device: torch.device) -> None:
        """Test that the network handles different input shapes correctly."""
        in_dim = 32
        hidden_dim = 64

        hopfield = HopfieldNetwork(in_dim=in_dim, hidden_dim=hidden_dim).to(
            device
        )

        # Test with various shapes
        test_shapes = [
            (1, in_dim),  # 1D case (no sequence)
            (5, in_dim),  # Batch of vectors
            (1, 3, in_dim),  # Single sequence
            (2, 3, in_dim),  # Batch of sequences
            (2, 3, 4, in_dim),  # Extra dimensions
        ]

        for shape in test_shapes:
            x = torch.randn(*shape, device=device)
            energy = hopfield(x)

            # Energy should be a scalar
            assert energy.ndim == 0, (
                f"Energy should be a scalar for shape {shape}"
            )

    def test_numerical_stability(self, device: torch.device) -> None:
        """Test numerical stability with extreme inputs."""
        in_dim = 16
        hidden_dim = 32

        hopfield = HopfieldNetwork(in_dim=in_dim, hidden_dim=hidden_dim).to(
            device
        )

        # Test with extremely large values
        x_large = torch.full((2, 3, in_dim), 1e10, device=device)
        energy_large = hopfield(x_large)
        assert not torch.isnan(energy_large).any(), (
            "Energy contains NaN with large inputs"
        )
        assert not torch.isinf(energy_large).any(), (
            "Energy contains Inf with large inputs"
        )

        # Test with extremely small values
        x_small = torch.full((2, 3, in_dim), 1e-10, device=device)
        energy_small = hopfield(x_small)
        assert not torch.isnan(energy_small).any(), (
            "Energy contains NaN with small inputs"
        )
        assert not torch.isinf(energy_small).any(), (
            "Energy contains Inf with small inputs"
        )

        # Test with mixed large/small values
        x_mixed = torch.zeros(2, 3, in_dim, device=device)
        x_mixed[:, :, 0] = 1e10  # First dimension large
        x_mixed[:, :, 1] = 1e-10  # Second dimension small
        energy_mixed = hopfield(x_mixed)
        assert not torch.isnan(energy_mixed).any(), (
            "Energy contains NaN with mixed inputs"
        )
        assert not torch.isinf(energy_mixed).any(), (
            "Energy contains Inf with mixed inputs"
        )

    def test_memory_capacity(
        self, device: torch.device, temporary_seed: Any
    ) -> None:
        """Test memory capacity with a simple recall task."""
        with temporary_seed(42):
            # Define dimensions
            in_dim = 32
            # Use more memories than input dimension to test overparameterized
            hidden_dim = 64

            # Create an array of memory patterns
            memories = torch.randn(10, in_dim, device=device)
            memories = F.normalize(memories, dim=1)  # Normalize patterns

            # Create a Hopfield network with fixed patterns
            hopfield = HopfieldNetwork(in_dim, hidden_dim).to(device)

            # Set first 10 memories to our patterns
            with torch.no_grad():
                hopfield.ξ[:10] = memories

            # Test recall: create noisy versions of memories and see if they
            # lead to low energy
            for i in range(10):
                # Create noisy version (30% noise)
                noise = torch.randn_like(memories[i]) * 0.3
                noisy_pattern = (
                    (memories[i] + noise).unsqueeze(0).unsqueeze(0)
                )  # Add batch dims

                # Create random pattern for comparison
                random_pattern = torch.randn(1, 1, in_dim, device=device)
                random_pattern = F.normalize(random_pattern, dim=2)

                # Calculate energies
                noisy_energy = hopfield(noisy_pattern)
                random_energy = hopfield(random_pattern)

                # Noisy version of real memory should have lower energy than
                # random pattern
                assert noisy_energy < random_energy, f"{i} recall failed"
