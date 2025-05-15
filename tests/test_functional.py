"""Comprehensive tests for functional energy interface."""

import pytest
import torch

from energy_transformer.functional import (
    attention_energy,
    energy_gradient,
    hopfield_energy,
    layer_norm_energy,
    minimize_energy,
    total_energy,
)


class TestFunctionalEnergy:
    def test_attention_energy_basic(self) -> None:
        batch_size, seq_len, d_model = 2, 10, 768
        n_heads, d_head = 12, 64

        g = torch.randn(batch_size, seq_len, d_model)
        wq = torch.randn(n_heads, d_head, d_model)
        wk = torch.randn(n_heads, d_head, d_model)

        energy = attention_energy(g, wq, wk)

        assert energy.shape == ()
        assert energy.dtype == torch.float32
        assert not torch.isnan(energy)
        assert not torch.isinf(energy)

    def test_attention_energy_no_batch(self) -> None:
        """Test without batch dimension"""
        seq_len, d_model = 10, 768
        n_heads, d_head = 12, 64

        g = torch.randn(seq_len, d_model)
        wq = torch.randn(n_heads, d_head, d_model)
        wk = torch.randn(n_heads, d_head, d_model)

        energy = attention_energy(g, wq, wk)

        assert energy.shape == ()
        assert not torch.isnan(energy)

    def test_attention_energy_custom_beta(self) -> None:
        """Test with custom beta parameter"""
        g = torch.randn(2, 10, 64)
        wq = torch.randn(4, 16, 64)
        wk = torch.randn(4, 16, 64)

        # Default beta
        energy1 = attention_energy(g, wq, wk)

        # Custom beta
        energy2 = attention_energy(g, wq, wk, beta=0.5)

        assert energy1 != energy2

    def test_attention_energy_per_head_beta(self) -> None:
        """Test attention energy with per-head beta values."""
        g = torch.randn(2, 10, 64)
        wq = torch.randn(4, 16, 64)
        wk = torch.randn(4, 16, 64)

        # Test with per-head beta
        beta_per_head = torch.tensor([0.1, 0.2, 0.3, 0.4])
        energy = attention_energy(g, wq, wk, beta=beta_per_head)

        assert energy.shape == ()
        assert torch.isfinite(energy)

    def test_hopfield_energy_basic(self) -> None:
        batch_size, d_model = 2, 768
        n_memories = 3072

        g = torch.randn(batch_size, d_model)
        xi = torch.randn(d_model, n_memories)

        energy = hopfield_energy(g, xi)

        assert energy.shape == ()
        assert energy <= 0  # Should be non-positive

    def test_hopfield_energy_multi_dim(self) -> None:
        """Test with multiple dimensions"""
        g = torch.randn(2, 10, 768)
        xi = torch.randn(768, 3072)

        energy = hopfield_energy(g, xi)

        assert energy.shape == ()
        assert energy <= 0

    def test_layer_norm_energy_basic(self) -> None:
        batch_size, seq_len, d_model = 2, 10, 768

        x = torch.randn(batch_size, seq_len, d_model)
        gamma = torch.tensor(1.0)
        delta = torch.randn(d_model)

        energy = layer_norm_energy(x, gamma, delta)

        assert energy.shape == ()
        assert energy.dtype == torch.float32

    def test_layer_norm_energy_no_bias(self) -> None:
        x = torch.randn(2, 10, 768)
        gamma = torch.tensor(1.0)

        energy = layer_norm_energy(x, gamma, delta=None)

        assert energy.shape == ()
        assert not torch.isnan(energy)

    def test_layer_norm_energy_scalar_gamma(self) -> None:
        """Test with scalar gamma as in the architecture"""
        x = torch.randn(2, 10, 768)
        gamma = torch.tensor(1.5)  # scalar
        delta = torch.randn(768)

        energy = layer_norm_energy(x, gamma, delta)

        assert energy.shape == ()
        assert not torch.isnan(energy)

    def test_total_energy_comprehensive(self) -> None:
        batch_size, seq_len, d_model = 2, 10, 768
        n_heads, d_head = 12, 64
        n_memories = 3072

        x = torch.randn(batch_size, seq_len, d_model)

        layer_norm_params = {
            "gamma": torch.tensor(1.0),
            "delta": torch.randn(d_model),
        }

        attention_params = {
            "wq": torch.randn(n_heads, d_head, d_model),
            "wk": torch.randn(n_heads, d_head, d_model),
        }

        hopfield_params = {"xi": torch.randn(d_model, n_memories)}

        energy = total_energy(
            x, layer_norm_params, attention_params, hopfield_params
        )

        assert energy.shape == ()
        assert not torch.isnan(energy)

    def test_energy_gradient_simple(self) -> None:
        d_model = 768
        x = torch.randn(10, d_model)

        def simple_energy(x):
            return (x**2).sum()

        grad = energy_gradient(x, simple_energy)

        assert grad.shape == x.shape
        # For quadratic energy, gradient should be 2*x
        assert torch.allclose(grad, 2 * x, atol=1e-5)

    def test_energy_gradient_complex(self) -> None:
        """Test gradient computation with complex energy function"""
        x = torch.randn(2, 10, 64)
        gamma = torch.tensor(1.0)
        delta = torch.randn(64)

        def complex_energy(x):
            return layer_norm_energy(x, gamma, delta)

        grad = energy_gradient(x, complex_energy)

        assert grad.shape == x.shape
        assert not torch.isnan(grad).any()
        assert not torch.isinf(grad).any()

    def test_minimize_energy_convergence(self) -> None:
        d_model = 768
        x_init = torch.randn(10, d_model)

        def quadratic_energy(x):
            # Energy with minimum at origin
            return (x**2).sum()

        # Basic minimization
        final_state, energies = minimize_energy(
            x_init, quadratic_energy, n_steps=100, alpha=0.01
        )

        assert energies.shape == (100,)
        # Energy should decrease
        assert energies[-1] < energies[0]
        # Final state should be closer to origin
        assert final_state.norm() < x_init.norm()
        assert final_state.shape == x_init.shape

    def test_minimize_energy_with_trajectory(self) -> None:
        d_model = 64
        x_init = torch.randn(5, d_model)

        def quadratic_energy(x):
            return (x**2).sum()

        # With trajectory
        final_state, energies, trajectory = minimize_energy(
            x_init,
            quadratic_energy,
            n_steps=50,
            alpha=0.01,
            return_trajectory=True,
        )

        assert trajectory.shape == (51, 5, 64)  # n_steps + 1
        assert torch.allclose(trajectory[0], x_init)
        assert torch.allclose(trajectory[-1], final_state)
        # Check that trajectory moves towards minimum
        for i in range(len(trajectory) - 1):
            assert (
                trajectory[i + 1].norm() <= trajectory[i].norm() + 1e-3
            )  # Allow small noise

    def test_minimize_energy_fixed_point(self) -> None:
        """Test that minimization finds fixed points"""
        x_init = torch.zeros(10, 64)  # Start at minimum

        def quadratic_energy(x):
            return (x**2).sum()

        final_state, energies = minimize_energy(
            x_init, quadratic_energy, n_steps=10, alpha=0.1
        )

        # Should stay at minimum
        assert torch.allclose(final_state, x_init, atol=1e-6)
        assert all(e < 1e-10 for e in energies)

    def test_numerical_stability_functional(self) -> None:
        """Test functional interface with extreme values"""
        # Very small values
        x_small = torch.full((1, 10, 64), 1e-10)
        gamma = torch.tensor(1.0)
        delta = torch.randn(64)

        energy = layer_norm_energy(x_small, gamma, delta)
        assert torch.isfinite(energy)

        # Very large values
        x_large = torch.full((1, 10, 64), 1e5)
        energy = layer_norm_energy(x_large, gamma, delta)
        assert torch.isfinite(energy)

    def test_gradient_consistency_functional(self) -> None:
        """Test that gradients match autograd"""
        x = torch.randn(2, 10, 64, requires_grad=True)
        gamma = torch.tensor(1.0)
        delta = torch.randn(64)

        # Using functional interface
        energy = layer_norm_energy(x, gamma, delta)
        grad_auto = torch.autograd.grad(energy, x, retain_graph=True)[0]

        # Using energy_gradient
        def energy_fn(x):
            return layer_norm_energy(x, gamma, delta)

        grad_functional = energy_gradient(x.detach(), energy_fn)

        assert torch.allclose(grad_auto, grad_functional, atol=1e-5)


class TestEnergyConservation:
    """Test energy conservation properties"""

    def test_energy_descent(self) -> None:
        """Test that gradient descent reduces energy"""
        torch.manual_seed(42)

        x = torch.randn(2, 10, 64)
        layer_norm_params = {
            "gamma": torch.tensor(1.0),
            "delta": torch.randn(64),
        }

        def energy_fn(x):
            return layer_norm_energy(
                x, layer_norm_params["gamma"], layer_norm_params["delta"]
            )

        # Take one gradient step
        grad = energy_gradient(x, energy_fn)
        x_new = x - 0.1 * grad

        energy_before = energy_fn(x)
        energy_after = energy_fn(x_new)

        # Energy should decrease
        assert energy_after < energy_before

    def test_stationary_points(self) -> None:
        """Test that gradients vanish at stationary points"""

        # Create a simple quadratic energy landscape
        def quadratic_energy(x):
            # Minimum at x = [1, 1, ..., 1]
            target = torch.ones_like(x)
            return ((x - target) ** 2).sum()

        # Start at minimum
        x_min = torch.ones(10, 64)
        grad_at_min = energy_gradient(x_min, quadratic_energy)

        # Gradient should be near zero at minimum
        assert torch.allclose(
            grad_at_min, torch.zeros_like(grad_at_min), atol=1e-5
        )


class TestBatchConsistency:
    """Test that batched operations are consistent"""

    def test_batch_vs_individual(self) -> None:
        """Test that batched energy matches sum of individual energies"""
        torch.manual_seed(42)

        # Create batch
        x_batch = torch.randn(3, 10, 64)
        gamma = torch.tensor(1.0)
        delta = torch.randn(64)

        # Batched energy
        energy_batch = layer_norm_energy(x_batch, gamma, delta)

        # Individual energies
        energies_individual = []
        for i in range(3):
            energy_i = layer_norm_energy(x_batch[i : i + 1], gamma, delta)
            energies_individual.append(energy_i)

        # Should match
        assert torch.allclose(energy_batch, sum(energies_individual), atol=1e-5)

    def test_attention_batch_consistency(self) -> None:
        """Test attention energy batch consistency"""
        torch.manual_seed(42)

        g_batch = torch.randn(3, 10, 64)
        wq = torch.randn(4, 16, 64)
        wk = torch.randn(4, 16, 64)

        # Batched
        energy_batch = attention_energy(g_batch, wq, wk)

        # Individual
        energies_individual = []
        for i in range(3):
            energy_i = attention_energy(g_batch[i], wq, wk)
            energies_individual.append(energy_i)

        # Should match
        assert torch.allclose(energy_batch, sum(energies_individual), atol=1e-5)


class TestMemoryAndPerformance:
    """Test memory efficiency and performance characteristics"""

    def test_gradient_checkpointing(self) -> None:
        """Test that gradients can be computed with checkpointing"""
        from torch.utils.checkpoint import checkpoint

        x = torch.randn(4, 256, 768, requires_grad=True)
        layer_norm_params = {
            "gamma": torch.tensor(1.0),
            "delta": torch.randn(768),
        }

        def checkpointed_energy(x):
            return checkpoint(
                layer_norm_energy,
                x,
                layer_norm_params["gamma"],
                layer_norm_params["delta"],
                use_reentrant=False,  # Fix the warning
            )

        # Should work with checkpointing
        energy = checkpointed_energy(x)
        energy.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_gpu_memory(self) -> None:
        """Test memory usage on GPU"""
        import gc

        torch.cuda.empty_cache()
        gc.collect()

        # Large tensors
        x = torch.randn(16, 512, 768).cuda()
        gamma = torch.tensor(1.0).cuda()
        delta = torch.randn(768).cuda()

        # Compute energy
        energy = layer_norm_energy(x, gamma, delta)

        # Should not OOM
        assert torch.isfinite(energy)

        # Clean up
        del x, gamma, delta, energy
        torch.cuda.empty_cache()
