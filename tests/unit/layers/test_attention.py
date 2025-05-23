"""Unit tests for the energy-based attention implementation."""

from typing import Any

import pytest
import torch

from energy_transformer.layers import MultiHeadEnergyAttention
from energy_transformer.layers.base import BaseEnergyAttention


class TestMultiHeadEnergyAttention:
    """Test suite for MultiHeadEnergyAttention implementation."""

    def test_initialization(self, device: torch.device) -> None:
        """Test that MultiHeadEnergyAttention initializes correctly."""
        in_dim = 768
        num_heads = 12
        head_dim = 64
        attention = MultiHeadEnergyAttention(
            in_dim=in_dim, num_heads=num_heads, head_dim=head_dim
        ).to(device)
        # Check that it's an instance of BaseEnergyAttention
        assert isinstance(attention, BaseEnergyAttention)
        # Check parameter shapes
        assert attention.w_k.shape == (num_heads, head_dim, in_dim)
        assert attention.w_q.shape == (num_heads, head_dim, in_dim)
        # Check that β is the expected constant value
        assert attention.β == 0.125, "β should be 0.125"

    def test_scalar_energy_output(self, device: torch.device) -> None:
        """Test that energy output is a scalar."""
        in_dim = 768
        num_heads = 12
        head_dim = 64
        attention = MultiHeadEnergyAttention(
            in_dim=in_dim, num_heads=num_heads, head_dim=head_dim
        ).to(device)
        # Test with various batch sizes and sequence lengths
        test_configs = [
            (1, 10),  # (batch_size, seq_len)
            (2, 8),
            (3, 16),
        ]
        for batch_size, seq_len in test_configs:
            x = torch.randn(batch_size, seq_len, in_dim, device=device)
            energy = attention(x)
            # Check that output is a scalar
            assert energy.ndim == 0, (
                f"Energy should be a scalar, got shape {energy.shape}"
            )
            assert energy.numel() == 1, (
                f"Energy should contain a single value, got {energy.numel()}"
            )

    def test_single_token_edge_case(self, device: torch.device) -> None:
        """Test behavior with single token (seq_len=1)."""
        in_dim = 768
        num_heads = 12
        head_dim = 64
        batch_size = 2
        attention = MultiHeadEnergyAttention(
            in_dim=in_dim, num_heads=num_heads, head_dim=head_dim
        ).to(device)
        # Create input with single token
        x = torch.randn(batch_size, 1, in_dim, device=device)
        energy = attention(x)
        # For single token, energy should be zero (self-attention is undefined)
        assert energy.item() == 0.0, "Energy should be zero for single token"

    def test_zero_energy_gradient(
        self, device: torch.device, grad_check: Any
    ) -> None:
        """Test that energy gradients are correct."""
        # Use smaller dimensions for faster gradient checking
        in_dim = 32
        num_heads = 2
        head_dim = 16
        batch_size = 2
        seq_len = 4
        # Use double precision for more accurate gradient checking
        attention = (
            MultiHeadEnergyAttention(
                in_dim=in_dim, num_heads=num_heads, head_dim=head_dim
            )
            .to(device)
            .double()
        )
        # Create random input with double precision
        x = torch.randn(
            batch_size, seq_len, in_dim, device=device, dtype=torch.float64
        )

        # Define energy function
        def energy_func(inputs: torch.Tensor) -> torch.Tensor:
            return attention(inputs)

        # Check gradients with relaxed tolerances for GPU computation
        is_close = grad_check(
            energy_func,
            x,
            eps=1e-5,  # Larger epsilon
            atol=1e-2,  # Relaxed absolute tolerance
            rtol=1e-1,  # Relaxed relative tolerance
        )
        assert is_close, (
            "Gradients do not match finite difference approximation"
        )

    def test_energy_with_uniform_tokens(self, device: torch.device) -> None:
        """Test energy computation with uniform tokens."""
        in_dim = 768
        num_heads = 12
        head_dim = 64
        batch_size = 2
        seq_len = 10
        attention = MultiHeadEnergyAttention(
            in_dim=in_dim, num_heads=num_heads, head_dim=head_dim
        ).to(device)
        # Create uniform tokens (all entries are the same)
        x_uniform = torch.ones(batch_size, seq_len, in_dim, device=device)
        energy_uniform = attention(x_uniform)
        # Energy should be finite
        assert not torch.isnan(energy_uniform).any(), (
            "Energy contains NaN with uniform tokens"
        )
        assert not torch.isinf(energy_uniform).any(), (
            "Energy contains Inf with uniform tokens"
        )
        # With identical tokens, attention energy should be predictable - but
        # non-zero due to the self-attention masking
        assert energy_uniform.item() != 0.0, (
            "Energy should not be zero for uniform tokens"
        )

    def test_identical_input_identical_energy(
        self, device: torch.device, temporary_seed: Any
    ) -> None:
        """Test that identical inputs produce identical energy values."""
        in_dim = 128
        num_heads = 4
        head_dim = 32
        batch_size = 2
        seq_len = 8
        with temporary_seed(42):
            # Create two identical models
            attention1 = MultiHeadEnergyAttention(
                in_dim=in_dim, num_heads=num_heads, head_dim=head_dim
            ).to(device)
            attention2 = MultiHeadEnergyAttention(
                in_dim=in_dim, num_heads=num_heads, head_dim=head_dim
            ).to(device)
            # Copy parameters from model 1 to model 2
            attention2.load_state_dict(attention1.state_dict())
            # Create random input
            x = torch.randn(batch_size, seq_len, in_dim, device=device)
            # Compute energy with both models
            energy1 = attention1(x)
            energy2 = attention2(x)
            # Energies should be identical
            assert energy1.item() == pytest.approx(energy2.item())

    def test_reset_parameters(self, device: torch.device) -> None:
        """Test that reset_parameters properly initializes parameters."""
        in_dim = 768
        num_heads = 12
        head_dim = 64
        attention = MultiHeadEnergyAttention(
            in_dim=in_dim, num_heads=num_heads, head_dim=head_dim
        ).to(device)
        # Store original parameters
        with torch.no_grad():
            original_w_k = attention.w_k.clone()
            original_w_q = attention.w_q.clone()
            # Modify parameters
            attention.w_k.fill_(10.0)
            attention.w_q.fill_(10.0)
        # Reset parameters
        attention.reset_parameters()
        # Check that parameters are reset and different from original
        assert not torch.allclose(
            attention.w_k, torch.full_like(attention.w_k, 10.0)
        )
        assert not torch.allclose(
            attention.w_q, torch.full_like(attention.w_q, 10.0)
        )
        # Parameters should be different from original due to random init
        assert not torch.allclose(attention.w_k, original_w_k)
        assert not torch.allclose(attention.w_q, original_w_q)

    def test_beta_value(self, device: torch.device) -> None:
        """Test that beta has the expected constant value."""
        in_dim = 768
        num_heads = 12
        head_dim = 64
        attention = MultiHeadEnergyAttention(
            in_dim=in_dim, num_heads=num_heads, head_dim=head_dim
        ).to(device)
        # Beta should be the constant 0.125
        assert attention.β == 0.125, "β should be 0.125"

    def test_batch_independence(self, device: torch.device) -> None:
        """Test energy calculations are independent across batch dimension."""
        in_dim = 64
        num_heads = 2
        head_dim = 32
        seq_len = 5
        attention = MultiHeadEnergyAttention(
            in_dim=in_dim, num_heads=num_heads, head_dim=head_dim
        ).to(device)
        # Create a batch where the first and third examples are identical
        # but the second is different
        x1 = torch.randn(1, seq_len, in_dim, device=device)
        x2 = torch.randn(1, seq_len, in_dim, device=device)
        x_batch = torch.cat([x1, x2, x1], dim=0)  # Shape [3, seq_len, in_dim]
        # Calculate energy for batch
        energy_batch = attention(x_batch)
        # Calculate energy for individual examples
        energy_1 = attention(x1)
        energy_2 = attention(x2)
        energy_3 = attention(x1)  # Same as energy_1
        # Batch energy should be the sum of individual energies
        expected_energy = energy_1 + energy_2 + energy_3
        assert energy_batch.item() == pytest.approx(expected_energy.item())

    def test_different_configurations(self, device: torch.device) -> None:
        """Test with different attention configurations."""
        in_dim = 256
        batch_size = 2
        seq_len = 6
        # Test various head configurations
        configs = [
            (1, 32),  # single head
            (4, 32),  # few heads
            (8, 16),  # many heads, smaller dim
            (16, 8),  # more heads, even smaller dim
        ]
        for num_heads, head_dim in configs:
            attention = MultiHeadEnergyAttention(
                in_dim=in_dim, num_heads=num_heads, head_dim=head_dim
            ).to(device)
            x = torch.randn(batch_size, seq_len, in_dim, device=device)
            energy = attention(x)
            # Check output is a scalar
            assert energy.ndim == 0, (
                f"Energy should be a scalar for config "
                f"({num_heads}, {head_dim})"
            )
            # Check for NaN or Inf
            assert not torch.isnan(energy).any(), (
                f"Energy contains NaN for config ({num_heads}, {head_dim})"
            )
            assert not torch.isinf(energy).any(), (
                f"Energy contains Inf for config ({num_heads}, {head_dim})"
            )
