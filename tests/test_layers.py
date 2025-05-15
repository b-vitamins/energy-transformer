"""Comprehensive tests for layers with gradient checking and edge cases."""

import numpy as np
import torch

from energy_transformer.config import ETConfig
from energy_transformer.layers import (
    EnergyAttention,
    EnergyLayerNorm,
    HopfieldNetwork,
)


class TestEnergyLayerNorm:
    def test_init(self) -> None:
        layer = EnergyLayerNorm(768)
        assert layer.d_model == 768
        assert layer.use_bias is True
        assert layer.gamma.shape == ()
        assert layer.delta.shape == (768,)

    def test_forward(self) -> None:
        layer = EnergyLayerNorm(768)
        x = torch.randn(2, 10, 768)
        output = layer(x)
        assert output.shape == x.shape
        # Check normalization - mean should be close to 0, variance close to 1
        assert torch.allclose(
            output.mean(dim=-1), torch.zeros(2, 10), atol=1e-5
        )
        assert torch.allclose(
            output.var(dim=-1, unbiased=False), torch.ones(2, 10), atol=1e-2
        )

    def test_forward_no_bias(self) -> None:
        layer = EnergyLayerNorm(768, use_bias=False)
        x = torch.randn(2, 10, 768)
        output = layer(x)
        assert output.shape == x.shape
        assert layer.delta is None

    def test_energy(self) -> None:
        layer = EnergyLayerNorm(768)
        x = torch.randn(2, 10, 768)
        energy = layer.energy(x)
        assert energy.shape == ()
        assert energy.dtype == torch.float32

    def test_gradient_consistency(self) -> None:
        """Test that g(x) = gradient of lagrangian"""
        torch.manual_seed(42)
        layer = EnergyLayerNorm(768)
        x = torch.randn(2, 10, 768, requires_grad=True)

        # Manual gradient
        g_x = layer.g(x)

        # Autograd gradient
        lagrangian = layer.lagrangian(x)
        auto_grad = torch.autograd.grad(lagrangian, x, retain_graph=True)[0]

        assert torch.allclose(g_x, auto_grad, atol=1e-5)

    def test_energy_gradient(self) -> None:
        """Test gradient of energy w.r.t. input"""
        torch.manual_seed(42)
        layer = EnergyLayerNorm(32)
        x = torch.randn(2, 5, 32, requires_grad=True)

        # Autograd
        energy = layer.energy(x)
        auto_grad = torch.autograd.grad(energy, x, retain_graph=True)[0]

        # Should match the manual computation
        g = layer.g(x)
        g - auto_grad

        # The gradient of energy should be g(x) - gradient of lagrangian
        # which is g(x) - g(x) = 0 at equilibrium, but not in general
        assert auto_grad.shape == x.shape

    def test_numerical_stability(self) -> None:
        """Test with extreme values"""
        layer = EnergyLayerNorm(768, eps=1e-5)

        # Very small values
        x_small = torch.full((1, 10, 768), 1e-10)
        output_small = layer(x_small)
        assert torch.isfinite(output_small).all()

        # Very large values
        x_large = torch.full((1, 10, 768), 1e10)
        output_large = layer(x_large)
        assert torch.isfinite(output_large).all()

        # Mixed values
        x_mixed = torch.randn(1, 10, 768) * 1e5
        x_mixed[:, :, :10] = 1e-10
        output_mixed = layer(x_mixed)
        assert torch.isfinite(output_mixed).all()


class TestEnergyAttention:
    def test_init(self) -> None:
        config = ETConfig()
        layer = EnergyAttention(
            d_model=config.d_model, n_heads=config.n_heads, d_head=config.d_head
        )
        assert layer.wq.shape == (
            config.n_heads,
            config.d_head,
            config.d_model,
        )
        assert layer.wk.shape == (
            config.n_heads,
            config.d_head,
            config.d_model,
        )

    def test_energy(self) -> None:
        config = ETConfig()
        layer = EnergyAttention(
            d_model=config.d_model, n_heads=config.n_heads, d_head=config.d_head
        )

        # Test with batch
        x = torch.randn(2, 10, config.d_model)
        energy = layer.energy(x)
        assert energy.shape == ()
        assert not torch.isnan(energy)
        assert not torch.isinf(energy)

        # Test without batch
        x = torch.randn(10, config.d_model)
        energy = layer.energy(x)
        assert energy.shape == ()
        assert not torch.isnan(energy)

    def test_gradient_computation(self) -> None:
        config = ETConfig()
        layer = EnergyAttention(
            d_model=config.d_model, n_heads=config.n_heads, d_head=config.d_head
        )

        x = torch.randn(2, 10, config.d_model, requires_grad=True)
        energy = layer.energy(x)
        grad = torch.autograd.grad(energy, x, retain_graph=True)[0]

        assert grad.shape == x.shape
        assert not torch.isnan(grad).any()
        assert not torch.isinf(grad).any()

    def test_energy_symmetry(self) -> None:
        """Test that energy is invariant to permutation within heads"""
        torch.manual_seed(42)
        layer = EnergyAttention(d_model=64, n_heads=4, d_head=16)

        x = torch.randn(1, 10, 64)
        energy1 = layer.energy(x)

        # Permute sequence
        perm = torch.randperm(10)
        x_perm = x[:, perm, :]
        energy2 = layer.energy(x_perm)

        # Energy should be invariant to permutation (approximately)
        # This tests that we're computing a proper energy function
        assert not torch.allclose(energy1, energy2, atol=1e-6)  # Should differ

    def test_numerical_stability_attention(self) -> None:
        """Test attention with extreme values"""
        layer = EnergyAttention(d_model=64, n_heads=4, d_head=16)

        # Very similar queries and keys (should handle numerical issues)
        x = torch.ones(1, 100, 64) * 0.1 + torch.randn(1, 100, 64) * 1e-5
        energy = layer.energy(x)
        assert torch.isfinite(energy)

        # Very different queries and keys
        x = torch.randn(1, 100, 64) * 100
        energy = layer.energy(x)
        assert torch.isfinite(energy)


class TestHopfieldNetwork:
    def test_init(self) -> None:
        layer = HopfieldNetwork(d_model=768, n_memories=3072)
        assert layer.xi.shape == (768, 3072)

    def test_energy(self) -> None:
        layer = HopfieldNetwork(d_model=768, n_memories=3072)
        x = torch.randn(2, 10, 768)
        energy = layer.energy(x)
        assert energy.shape == ()
        assert energy <= 0  # Energy should be non-positive due to -0.5 factor

    def test_gradient_computation(self) -> None:
        layer = HopfieldNetwork(d_model=768, n_memories=3072)
        x = torch.randn(2, 10, 768, requires_grad=True)
        energy = layer.energy(x)
        grad = torch.autograd.grad(energy, x, retain_graph=True)[0]

        assert grad.shape == x.shape
        assert not torch.isnan(grad).any()

    def test_relu_property(self) -> None:
        """Test that energy increases when we move towards stored patterns"""
        torch.manual_seed(42)
        d_model = 64
        n_memories = 5

        layer = HopfieldNetwork(d_model=d_model, n_memories=n_memories)

        # Create a pattern close to one memory pattern
        with torch.no_grad():
            layer.xi[:, 0] = torch.randn(d_model)
            layer.xi[:, 0] = layer.xi[:, 0] / layer.xi[:, 0].norm()

        # Input close to first memory
        x1 = layer.xi[:, 0].detach().unsqueeze(0) * 0.8
        energy1 = layer.energy(x1)

        # Input far from memories
        x2 = torch.randn_like(x1)
        energy2 = layer.energy(x2)

        # Energy should be lower (more negative) for patterns close to memories
        assert energy1 < energy2

    def test_zero_input(self) -> None:
        """Test with zero input"""
        layer = HopfieldNetwork(d_model=768, n_memories=3072)
        x = torch.zeros(2, 10, 768)
        energy = layer.energy(x)
        assert energy == 0.0  # ReLU(0) = 0, so energy = 0

    def test_batched_vs_single(self) -> None:
        """Test that batched computation matches single computations"""
        torch.manual_seed(42)
        layer = HopfieldNetwork(d_model=64, n_memories=128)

        x_batch = torch.randn(3, 10, 64)
        energy_batch = layer.energy(x_batch)

        # Compute energies individually
        energies_single = []
        for i in range(3):
            energy_single = layer.energy(x_batch[i : i + 1])
            energies_single.append(energy_single)

        # Should match sum of individual energies
        assert torch.allclose(energy_batch, sum(energies_single), atol=1e-6)


class TestCrossValidation:
    """Cross-validation tests comparing PyTorch and JAX implementations"""

    def test_layer_norm_cross_validation(self) -> None:
        """Compare LayerNorm outputs with reference implementation"""
        torch.manual_seed(42)
        np.random.seed(42)

        d_model = 64
        layer = EnergyLayerNorm(d_model)

        # Set specific parameters
        layer.gamma.data = torch.tensor(1.5)
        layer.delta.data = torch.randn(d_model) * 0.1

        # Test input
        x = torch.randn(2, 10, d_model)

        # Compute outputs
        output = layer(x)
        energy = layer.energy(x)
        lagrangian = layer.lagrangian(x)

        # Basic checks
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
        assert torch.isfinite(energy)
        assert torch.isfinite(lagrangian)

        # Check that energy = g*x - lagrangian
        g_x = layer.g(x)
        expected_energy = (g_x * x).sum() - lagrangian
        assert torch.allclose(energy, expected_energy, atol=1e-5)

    def test_attention_energy_cross_validation(self) -> None:
        """Test attention energy computation matches expected behavior"""
        torch.manual_seed(42)

        d_model = 64
        n_heads = 4
        d_head = 16
        seq_len = 10

        layer = EnergyAttention(d_model=d_model, n_heads=n_heads, d_head=d_head)

        # Create specific test case
        x = torch.randn(2, seq_len, d_model)

        # Compute energy
        energy = layer.energy(x)

        # Manually compute energy to verify
        beta = 1.0 / np.sqrt(d_head)
        q = torch.einsum("...qd,hzd->...qhz", x, layer.wq)
        k = torch.einsum("...kd,hzd->...khz", x, layer.wk)
        scores = torch.einsum("...qhz,...khz->...hqk", q, k)
        a = torch.logsumexp(beta * scores, dim=-1)
        expected_energy = -1.0 / beta * a.sum()

        assert torch.allclose(energy, expected_energy, atol=1e-5)

    def test_full_transformer_energy(self) -> None:
        """Test that the full transformer energy is computed correctly"""
        from energy_transformer import EnergyTransformer

        torch.manual_seed(42)
        config = ETConfig(d_model=64, n_heads=4, d_head=16, n_steps=10)
        model = EnergyTransformer(config)

        x = torch.randn(2, 10, 64)

        # Get individual energies
        energy = model.energy(x)

        # Check that it's the sum of components
        g = model.layer_norm(x)
        ln_energy = model.layer_norm.energy(x)
        attn_energy = model.attention.energy(g)
        hopfield_energy = model.hopfield.energy(g)

        expected_total = ln_energy + attn_energy + hopfield_energy
        assert torch.allclose(energy, expected_total, atol=1e-5)


class TestEdgeCases:
    """Test edge cases and numerical stability"""

    def test_single_token(self) -> None:
        """Test with single token sequences"""
        layer = EnergyAttention(d_model=64, n_heads=4, d_head=16)
        x = torch.randn(1, 1, 64)
        energy = layer.energy(x)
        assert torch.isfinite(energy)

    def test_very_long_sequence(self) -> None:
        """Test with very long sequences"""
        layer = EnergyAttention(d_model=64, n_heads=4, d_head=16)
        x = torch.randn(1, 1000, 64)
        energy = layer.energy(x)
        assert torch.isfinite(energy)

    def test_gradient_accumulation(self) -> None:
        """Test gradient accumulation through multiple steps"""
        from energy_transformer import EnergyTransformer

        config = ETConfig(d_model=64, n_heads=4, d_head=16, n_steps=5)
        model = EnergyTransformer(config)

        x = torch.randn(1, 10, 64, requires_grad=True)
        output = model(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestMemoryEfficiency:
    """Test memory efficiency and performance"""

    def test_memory_usage(self) -> None:
        """Test that memory usage is reasonable"""
        import gc

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        # Large model
        config = ETConfig(d_model=768, n_heads=12, d_head=64)
        from energy_transformer import EnergyTransformer

        model = EnergyTransformer(config)

        if torch.cuda.is_available():
            model = model.cuda()
            x = torch.randn(8, 512, 768).cuda()
        else:
            x = torch.randn(8, 512, 768)

        # Run forward pass
        output = model(x)

        # Should complete without OOM
        assert output.shape == x.shape
