"""Unit tests for the energy-based LayerNorm implementation."""

import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from energy_transformer.layers import LayerNorm
from energy_transformer.layers.base import BaseLayerNorm


class TestLayerNorm:
    """Test suite for LayerNorm implementation."""

    def test_initialization(self, device: torch.device) -> None:
        """Test that LayerNorm initializes correctly."""
        # Test with default parameters
        in_dim = 768
        layer_norm = LayerNorm(in_dim).to(device)

        # Check that it's an instance of BaseLayerNorm
        assert isinstance(layer_norm, BaseLayerNorm)

        # Check parameter shapes
        assert layer_norm.logγ.shape == ()  # Scalar
        assert layer_norm.δ.shape == (in_dim,)  # Vector of size in_dim

        # Check default initialization values
        assert F.softplus(layer_norm.logγ).item() == pytest.approx(
            1.0, abs=1e-5
        )
        assert torch.allclose(layer_norm.δ, torch.zeros_like(layer_norm.δ))

    def test_custom_epsilon(self, device: torch.device) -> None:
        """Test that custom epsilon value is respected."""
        in_dim = 512
        custom_eps = 1e-7
        layer_norm = LayerNorm(in_dim, eps=custom_eps).to(device)

        # Check that epsilon is stored properly
        assert layer_norm.eps == custom_eps

    def test_forward_shape(self, device: torch.device) -> None:
        """Test that forward pass preserves shape."""
        batch_size = 2
        seq_len = 10
        in_dim = 256

        # Test with various input shapes
        test_shapes = [
            (in_dim,),  # 1D
            (seq_len, in_dim),  # 2D
            (batch_size, seq_len, in_dim),  # 3D
            (batch_size, 3, seq_len, in_dim),  # 4D
        ]

        layer_norm = LayerNorm(in_dim).to(device)

        for shape in test_shapes:
            x = torch.randn(*shape, device=device)
            output = layer_norm(x)
            assert output.shape == x.shape, (
                f"Shape mismatch for input shape {shape}"
            )

    def test_normalization_behavior(self, device: torch.device) -> None:
        """Test that LayerNorm correctly normalizes the input."""
        batch_size = 2
        seq_len = 5
        in_dim = 256

        # Create input with known mean and variance
        x = torch.randn(batch_size, seq_len, in_dim, device=device)

        # Set logγ manually to 0 and δ to zero for easier testing
        layer_norm = LayerNorm(in_dim).to(device)
        with torch.no_grad():
            # Use log(e^1 - 1) to make softplus(logγ) = 1.0
            layer_norm.logγ.fill_(math.log(math.exp(1.0) - 1.0))
            layer_norm.δ.fill_(0.0)

        # Forward pass
        normalized = layer_norm(x)

        # Verify normalization
        for b in range(batch_size):
            for s in range(seq_len):
                token = x[b, s]
                norm_token = normalized[b, s]

                # Mean should be close to 0 after centering
                mean = token.mean()

                # Calculate variance
                centered = token - mean
                variance = (centered**2).mean()
                std = torch.sqrt(variance + layer_norm.eps)

                # Expected output: γ * (x - mean) / sqrt(var + eps) + δ
                # With γ = 1.0 and δ = 0.0
                expected = centered / std

                assert torch.allclose(
                    norm_token, expected, rtol=1e-4, atol=1e-5
                ), f"Normalization incorrect at batch {b}, seq {s}"

                # Test statistical properties
                assert norm_token.mean().abs() < 1e-5, (
                    "Mean should be very close to 0"
                )
                assert norm_token.std().item() == pytest.approx(
                    1.0, abs=1e-2
                ), "Standard deviation should be very close to 1"

    def test_gamma_positivity(self, device: torch.device) -> None:
        """Test that gamma is always positive through softplus."""
        in_dim = 768
        layer_norm = LayerNorm(in_dim).to(device)

        # Test with various values of logγ
        test_values = [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0]

        for val in test_values:
            with torch.no_grad():
                layer_norm.logγ.fill_(val)
            gamma = F.softplus(layer_norm.logγ)
            assert gamma > 0, (
                f"γ should be positive, got {gamma} for logγ = {val}"
            )

    def test_gradient_flow(self, device: torch.device) -> None:
        """Test that gradients flow properly through the layer norm."""
        batch_size = 2
        seq_len = 3
        in_dim = 64

        layer_norm = LayerNorm(in_dim).to(device)
        x = torch.randn(
            batch_size, seq_len, in_dim, device=device, requires_grad=True
        )

        # Forward pass
        y = layer_norm(x)

        # Create a dummy loss that depends on all values
        loss = y.sum()

        # Backward pass
        loss.backward()

        # Check that gradients are not None
        assert x.grad is not None, "Input gradients should not be None"
        assert layer_norm.logγ.grad is not None, (
            "logγ gradients should not be None"
        )
        assert layer_norm.δ.grad is not None, "δ gradients should not be None"

        # Check gradient shapes
        assert x.grad.shape == x.shape
        assert layer_norm.logγ.grad.shape == layer_norm.logγ.shape
        assert layer_norm.δ.grad.shape == layer_norm.δ.shape

    def test_reset_parameters(self, device: torch.device) -> None:
        """Test that reset_parameters properly initializes parameters."""
        in_dim = 768
        layer_norm = LayerNorm(in_dim).to(device)

        # Modify parameters
        with torch.no_grad():
            layer_norm.logγ.fill_(10.0)
            layer_norm.δ.fill_(5.0)

        # Reset parameters
        layer_norm.reset_parameters()

        # Check that parameters are reset correctly
        assert F.softplus(layer_norm.logγ).item() == pytest.approx(
            1.0, abs=1e-5
        )
        assert torch.allclose(layer_norm.δ, torch.zeros_like(layer_norm.δ))

    def test_mixed_precision_compatibility(self, device: torch.device) -> None:
        """Test compatibility with mixed precision training."""
        # Skip if not on CUDA
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available, skipping mixed precision test")

        in_dim = 768
        layer_norm = LayerNorm(in_dim).to(device)

        # Fix random seed for reproducible inputs
        torch.manual_seed(42)

        # Test with float32
        x_f32 = torch.randn(2, 10, in_dim, device=device, dtype=torch.float32)
        out_f32 = layer_norm(x_f32)
        assert out_f32.dtype == torch.float32

        # Reset seed to get the same values
        torch.manual_seed(42)

        # Test with float16
        x_f16 = torch.randn(2, 10, in_dim, device=device, dtype=torch.float16)
        out_f16 = layer_norm(x_f16)
        assert out_f16.dtype == torch.float16

        # Check that results are similar after conversion
        # Use larger tolerances for float16 comparison
        assert torch.allclose(
            out_f32.to(torch.float16), out_f16, rtol=1e-1, atol=1e-1
        ), "Mixed precision output should be close to full precision"

    def test_numerical_stability(self, device: torch.device) -> None:
        """Test numerical stability with extreme inputs."""
        in_dim = 64
        layer_norm = LayerNorm(in_dim).to(device)

        # Test with extremely large values
        x_large = torch.full((2, 3, in_dim), 1e10, device=device)
        y_large = layer_norm(x_large)
        assert not torch.isnan(y_large).any(), (
            "Output contains NaN with large inputs"
        )
        assert not torch.isinf(y_large).any(), (
            "Output contains Inf with large inputs"
        )

        # Test with extremely small values
        x_small = torch.full((2, 3, in_dim), 1e-10, device=device)
        y_small = layer_norm(x_small)
        assert not torch.isnan(y_small).any(), (
            "Output contains NaN with small inputs"
        )
        assert not torch.isinf(y_small).any(), (
            "Output contains Inf with small inputs"
        )

        # Test with constant input (zero variance)
        x_const = torch.ones(2, 3, in_dim, device=device)
        y_const = layer_norm(x_const)
        assert not torch.isnan(y_const).any(), (
            "Output contains NaN with constant inputs"
        )
        assert not torch.isinf(y_const).any(), (
            "Output contains Inf with constant inputs"
        )
