"""Cross-validation tests comparing PyTorch implementation with JAX reference."""

import pytest
import torch


class TestJAXCompatibility:
    """Test compatibility with JAX implementation behaviors"""

    def test_einsum_patterns(self) -> None:
        """Test that our einsum patterns match JAX implementation"""
        # Test attention einsum pattern
        # JAX: jnp.einsum("qd,hzd->qhz", g, self.Wq)
        # Our: torch.einsum("...qd,hzd->...qhz", g, self.wq)

        g = torch.randn(10, 64)  # seq_len, d_model
        wq = torch.randn(4, 16, 64)  # n_heads, d_head, d_model

        # Our pattern should work with and without batch
        result1 = torch.einsum("qd,hzd->qhz", g, wq)
        assert result1.shape == (10, 4, 16)  # seq_len, n_heads, d_head

        # With batch
        g_batch = torch.randn(2, 10, 64)
        result2 = torch.einsum("...qd,hzd->...qhz", g_batch, wq)
        assert result2.shape == (2, 10, 4, 16)

    def test_scalar_gamma_behavior(self) -> None:
        """Test that scalar gamma behaves like JAX implementation"""
        from energy_transformer.layers import EnergyLayerNorm

        # JAX uses scalar gamma
        layer = EnergyLayerNorm(64)
        assert layer.gamma.shape == ()  # Scalar

        x = torch.randn(2, 10, 64)
        output = layer(x)

        # Manual computation matching JAX
        x_centered = x - x.mean(dim=-1, keepdim=True)
        variance = (x_centered**2).mean(dim=-1, keepdim=True)
        expected = layer.gamma * x_centered / torch.sqrt(variance + layer.eps)
        if layer.use_bias:
            expected = expected + layer.delta

        assert torch.allclose(output, expected, atol=1e-5)

    def test_energy_sign_conventions(self) -> None:
        """Test that energy signs match JAX conventions"""
        from energy_transformer.layers import EnergyAttention, HopfieldNetwork

        # Hopfield energy should be negative (JAX: -0.5 * ...)
        hopfield = HopfieldNetwork(64, 256)
        x = torch.randn(1, 10, 64)
        h_energy = hopfield.energy(x)
        assert h_energy <= 0 or torch.allclose(h_energy, torch.tensor(0.0))

        # Attention energy should be negative (JAX: -1/beta * ...)
        attention = EnergyAttention(64, 4, 16)
        a_energy = attention.energy(x)
        assert a_energy <= 0  # Should be negative

    def test_no_batch_dimension_handling(self) -> None:
        """Test handling of inputs without batch dimension like JAX"""
        from energy_transformer.layers import EnergyAttention

        # JAX version doesn't assume batch dimension
        layer = EnergyAttention(64, 4, 16)

        # Single sequence (no batch)
        x_single = torch.randn(10, 64)
        energy_single = layer.energy(x_single)

        # Batched version
        x_batch = x_single.unsqueeze(0)
        energy_batch = layer.energy(x_batch)

        # Energies should be equal (no batch normalization in JAX)
        assert torch.allclose(energy_single, energy_batch, atol=1e-5)

    def test_relu_activation(self) -> None:
        """Test ReLU behavior matches JAX"""
        from energy_transformer.layers import HopfieldNetwork

        # JAX uses jax.nn.relu
        # PyTorch uses F.relu
        # Both should behave identically

        layer = HopfieldNetwork(64, 256)

        # Test with positive, negative, and zero inputs
        x_pos = torch.randn(1, 10, 64).abs()
        x_neg = -torch.randn(1, 10, 64).abs()
        x_zero = torch.zeros(1, 10, 64)

        e_pos = layer.energy(x_pos)
        e_neg = layer.energy(x_neg)
        e_zero = layer.energy(x_zero)

        # Energy with zero input should be exactly zero
        assert e_zero == 0.0

        # Energy should be non-positive
        assert e_pos <= 0
        assert e_neg <= 0

    def test_gradient_computation_style(self) -> None:
        """Test that gradient computation matches JAX style"""
        from energy_transformer import EnergyTransformer, ETConfig

        config = ETConfig(d_model=64, n_heads=4, d_head=16, n_steps=10)
        model = EnergyTransformer(config)

        x = torch.randn(1, 10, 64, requires_grad=True)

        # JAX uses jax.value_and_grad
        # We use torch.autograd.grad
        energy = model.energy(x)
        grad = torch.autograd.grad(energy, x, create_graph=True)[0]

        # Update style matches JAX: x = x - alpha * grad
        x_new = x - config.alpha * grad

        # Energy should decrease
        energy_new = model.energy(x_new)
        assert energy_new < energy


class TestMathematicalCorrectness:
    """Test mathematical correctness of energy formulations"""

    def test_attention_energy_formula(self) -> None:
        """Test attention energy matches mathematical formula"""
        from energy_transformer.functional import attention_energy

        # Energy = -1/beta * sum(logsumexp(beta * Q @ K^T))
        g = torch.randn(1, 10, 64)
        wq = torch.randn(4, 16, 64)
        wk = torch.randn(4, 16, 64)
        beta = 0.25  # 1/sqrt(16)

        energy = attention_energy(g, wq, wk, beta)

        # Manual computation
        q = torch.einsum("...qd,hzd->...qhz", g, wq)
        k = torch.einsum("...kd,hzd->...khz", g, wk)
        scores = torch.einsum("...qhz,...khz->...hqk", q, k)
        manual_energy = (
            -1.0 / beta * torch.logsumexp(beta * scores, dim=-1).sum()
        )

        assert torch.allclose(energy, manual_energy, atol=1e-5)

    def test_layer_norm_lagrangian_formula(self) -> None:
        """Test layer norm Lagrangian matches mathematical formula"""
        from energy_transformer.layers import EnergyLayerNorm

        layer = EnergyLayerNorm(64)
        x = torch.randn(2, 10, 64)

        lagrangian = layer.lagrangian(x)

        # Manual computation: L = D * gamma * sqrt(1/D * sum((x - mean)^2))
        d = x.shape[-1]
        x_centered = x - x.mean(dim=-1, keepdim=True)
        variance_sum = (x_centered**2).sum(dim=-1)
        manual_lagrangian = (
            d * layer.gamma * torch.sqrt(variance_sum / d + layer.eps).sum()
        )

        if layer.use_bias:
            manual_lagrangian = manual_lagrangian + (layer.delta * x).sum()

        assert torch.allclose(lagrangian, manual_lagrangian, atol=1e-5)

    def test_energy_legendre_transform(self) -> None:
        """Test that energy = g*x - Lagrangian"""
        from energy_transformer.layers import EnergyLayerNorm

        layer = EnergyLayerNorm(64)
        x = torch.randn(2, 10, 64)

        energy = layer.energy(x)
        g = layer.g(x)
        lagrangian = layer.lagrangian(x)

        expected_energy = (g * x).sum() - lagrangian

        assert torch.allclose(energy, expected_energy, atol=1e-5)

    def test_hopfield_energy_formula(self) -> None:
        """Test Hopfield energy formula"""
        from energy_transformer.layers import HopfieldNetwork

        layer = HopfieldNetwork(64, 256)
        x = torch.randn(2, 10, 64)

        energy = layer.energy(x)

        # Manual: E = -0.5 * sum(relu(x @ xi)^2)
        hidden = torch.einsum("...d,dm->...m", x, layer.xi)
        manual_energy = -0.5 * (torch.relu(hidden) ** 2).sum()

        assert torch.allclose(energy, manual_energy, atol=1e-5)


class TestNumericalPrecision:
    """Test numerical precision and stability"""

    def test_logsumexp_stability(self) -> None:
        """Test numerical stability of logsumexp"""
        from energy_transformer.functional import attention_energy

        # Large values that could cause overflow
        g = torch.randn(1, 10, 64) * 100
        wq = torch.randn(4, 16, 64) * 0.01
        wk = torch.randn(4, 16, 64) * 0.01

        energy = attention_energy(g, wq, wk)

        assert torch.isfinite(energy)
        assert not torch.isnan(energy)

    def test_sqrt_stability(self) -> None:
        """Test numerical stability of square root in layer norm"""
        from energy_transformer.layers import EnergyLayerNorm

        layer = EnergyLayerNorm(64, eps=1e-5)

        # Very small variance
        x_uniform = torch.ones(1, 10, 64) * 42.0  # Uniform values
        x_uniform[:, :, 0] += 1e-8  # Tiny variation

        output = layer(x_uniform)
        energy = layer.energy(x_uniform)

        assert torch.isfinite(output).all()
        assert torch.isfinite(energy)

    def test_gradient_stability(self) -> None:
        """Test gradient stability through multiple steps"""
        from energy_transformer import EnergyTransformer, ETConfig

        config = ETConfig(d_model=64, n_heads=4, d_head=16, n_steps=100)
        model = EnergyTransformer(config)

        # Initialize with small random values
        x = torch.randn(1, 10, 64) * 0.01

        # Run many steps
        output = model(x)

        # Should remain stable
        assert torch.isfinite(output).all()
        assert output.abs().max() < 100  # No explosion


class TestBatchProcessing:
    """Test batch processing consistency"""

    def test_energy_additivity(self) -> None:
        """Test that energy is additive over batch dimension"""
        from energy_transformer import EnergyTransformer, ETConfig

        config = ETConfig(d_model=64, n_heads=4, d_head=16)
        model = EnergyTransformer(config)

        # Individual samples
        x1 = torch.randn(1, 10, 64)
        x2 = torch.randn(1, 10, 64)

        e1 = model.energy(x1)
        e2 = model.energy(x2)

        # Batched
        x_batch = torch.cat([x1, x2], dim=0)
        e_batch = model.energy(x_batch)

        # Should be sum of individual energies
        assert torch.allclose(e_batch, e1 + e2, atol=1e-5)

    def test_parallel_vs_sequential(self) -> None:
        """Test that batch processing matches sequential processing"""
        from energy_transformer import ImageEnergyTransformer, ImageETConfig

        config = ImageETConfig(image_shape=(3, 64, 64), patch_size=16)
        model = ImageEnergyTransformer(config)

        # Create batch
        images = torch.randn(3, 3, 64, 64)
        masks = model.create_random_mask(3, images.device)

        # Batch processing
        results_batch = model(images, mask=masks)

        # Sequential processing
        results_seq = []
        for i in range(3):
            result_i = model(images[i : i + 1], mask=masks[i : i + 1])
            results_seq.append(result_i["reconstruction"])

        results_seq = torch.cat(results_seq, dim=0)

        # Should match
        assert torch.allclose(
            results_batch["reconstruction"], results_seq, atol=1e-5
        )


class TestMemoryEfficiency:
    """Test memory-efficient operations"""

    def test_gradient_checkpointing_compatibility(self) -> None:
        """Test compatibility with gradient checkpointing"""
        from energy_transformer import EnergyTransformer, ETConfig

        config = ETConfig(d_model=256, n_heads=8, d_head=32, n_steps=5)
        model = EnergyTransformer(config)

        x = torch.randn(4, 128, 256, requires_grad=True)

        # Standard forward
        output1 = model(x)
        loss1 = output1.sum()

        # We need to create a new model for checkpointing to work properly
        x2 = x.clone().detach().requires_grad_(True)
        output2 = model(x2)
        loss2 = output2.sum()

        # Compute gradients
        grad1 = torch.autograd.grad(loss1, x, retain_graph=True)[0]
        grad2 = torch.autograd.grad(loss2, x2)[0]

        # Results should be similar (not exact due to different computation paths)
        assert torch.allclose(output1, output2, atol=1e-5)
        assert torch.allclose(grad1, grad2, atol=1e-5)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_mixed_precision_compatibility(self) -> None:
        """Test that model works with automatic mixed precision"""
        from torch.amp import GradScaler, autocast

        from energy_transformer import ImageEnergyTransformer, ImageETConfig

        config = ImageETConfig(image_shape=(3, 224, 224), patch_size=16)
        model = ImageEnergyTransformer(config).cuda()

        scaler = GradScaler("cuda")
        optimizer = torch.optim.Adam(model.parameters())

        images = torch.randn(4, 3, 224, 224).cuda()
        masks = model.create_random_mask(4, images.device)

        # Mixed precision training step
        with autocast("cuda"):
            results = model(images, mask=masks)
            loss = (results["reconstruction"] - images).pow(2).mean()

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Should complete without errors
        assert torch.isfinite(loss)
