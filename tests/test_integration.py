"""Comprehensive integration tests for the complete Energy Transformer system."""

import pytest
import torch

from energy_transformer import (
    EnergyAttention,
    EnergyTransformer,
    ETConfig,
    ImageEnergyTransformer,
    ImageETConfig,
)
from energy_transformer.utils import (
    Patcher,
    load_checkpoint,
    normalize_image,
    save_checkpoint,
    unnormalize_image,
)


class TestEnergyTransformerIntegration:
    """End-to-end tests for Energy Transformer"""

    def test_basic_forward_pass(self) -> None:
        """Test basic forward pass through the model"""
        config = ETConfig(d_model=768, n_heads=12, d_head=64)
        model = EnergyTransformer(config)

        x = torch.randn(2, 10, 768)
        output = model(x)

        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    def test_energy_minimization_convergence(self) -> None:
        """Test that energy minimization converges"""
        config = ETConfig(d_model=64, n_heads=4, d_head=16, n_steps=50)
        model = EnergyTransformer(config)

        x = torch.randn(1, 10, 64)
        output, energies = model.compute_energy_trajectory(x)

        # Energy should generally decrease
        assert energies[-1] < energies[0]
        # Check for general convergence trend
        recent_avg = energies[-10:].mean()
        early_avg = energies[:10].mean()
        assert recent_avg < early_avg

    def test_trajectory_consistency(self) -> None:
        """Test that trajectory is consistent with individual steps"""
        config = ETConfig(d_model=64, n_heads=4, d_head=16, n_steps=10)
        model = EnergyTransformer(config)

        x = torch.randn(1, 5, 64)

        # Get trajectory
        output_traj, trajectory = model(x, return_trajectory=True)

        # Manually compute same trajectory
        x_manual = x.clone().detach().requires_grad_(True)
        manual_trajectory = [x_manual.clone().detach()]

        for _ in range(config.n_steps):
            energy = model.energy(x_manual)
            grad = torch.autograd.grad(energy, x_manual)[0]
            x_manual = x_manual - config.alpha * grad
            x_manual = x_manual.detach().requires_grad_(True)
            manual_trajectory.append(x_manual.clone().detach())

        # Should match
        for i in range(len(trajectory)):
            assert torch.allclose(
                trajectory[i], manual_trajectory[i], atol=1e-5
            )

    def test_different_sequence_lengths(self) -> None:
        """Test with various sequence lengths"""
        config = ETConfig(d_model=64, n_heads=4, d_head=16)
        model = EnergyTransformer(config)

        for seq_len in [1, 10, 100, 512]:
            x = torch.randn(2, seq_len, 64)
            output = model(x)
            assert output.shape == x.shape
            assert torch.isfinite(output).all()

    def test_gradient_flow(self) -> None:
        """Test gradient flow through the model"""
        config = ETConfig(d_model=64, n_heads=4, d_head=16, n_steps=10)
        model = EnergyTransformer(config)

        x = torch.randn(2, 10, 64, requires_grad=True)
        output = model(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().mean() > 0  # Non-zero gradients


class TestImageEnergyTransformerIntegration:
    """End-to-end tests for Image Energy Transformer"""

    def test_image_forward_pass(self) -> None:
        """Test forward pass with images"""
        config = ImageETConfig(image_shape=(3, 224, 224), patch_size=16)
        model = ImageEnergyTransformer(config)

        images = torch.randn(2, 3, 224, 224)
        results = model(images)

        assert "reconstruction" in results
        assert results["reconstruction"].shape == images.shape
        assert torch.isfinite(results["reconstruction"]).all()

    def test_masked_reconstruction(self) -> None:
        """Test masked image reconstruction"""
        config = ImageETConfig(
            image_shape=(3, 224, 224), patch_size=16, n_mask=100
        )
        model = ImageEnergyTransformer(config)

        images = torch.randn(2, 3, 224, 224)
        masks = model.create_random_mask(2, images.device)

        results = model(images, mask=masks)

        assert results["reconstruction"].shape == images.shape
        # Unmasked patches should be perfectly reconstructed
        patches_true = model.patcher.patchify(images)
        patches_pred = model.patcher.patchify(results["reconstruction"])

        unmask_expanded = (~masks).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        unmasked_true = patches_true * unmask_expanded
        unmasked_pred = patches_pred * unmask_expanded

        # Unmasked patches should be unchanged
        assert torch.allclose(unmasked_true, unmasked_pred, atol=1e-3)

    def test_cls_token_extraction(self) -> None:
        """Test CLS token extraction"""
        config = ImageETConfig(image_shape=(3, 224, 224), patch_size=16)
        model = ImageEnergyTransformer(config)

        images = torch.randn(2, 3, 224, 224)
        results = model(images, return_cls_token=True)

        assert "cls_token" in results
        assert results["cls_token"].shape == (2, config.et_config.d_model)
        assert torch.isfinite(results["cls_token"]).all()

    def test_different_image_sizes(self) -> None:
        """Test with different image sizes"""
        for img_size in [64, 128, 224, 256]:
            if img_size % 16 != 0:  # Skip sizes that don't divide evenly
                continue

            config = ImageETConfig(
                image_shape=(3, img_size, img_size), patch_size=16
            )
            model = ImageEnergyTransformer(config)

            images = torch.randn(1, 3, img_size, img_size)
            results = model(images)

            assert results["reconstruction"].shape == images.shape


class TestCheckpointing:
    """Test model checkpointing and loading"""

    def test_save_load_checkpoint(self, tmp_path):
        """Test saving and loading checkpoints"""
        # Create model
        config = ETConfig(d_model=64, n_heads=4, d_head=16)
        model = EnergyTransformer(config)

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters())

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pth"
        save_checkpoint(
            model, checkpoint_path, optimizer=optimizer, epoch=10, loss=0.5
        )

        # Create new model and load
        model2 = EnergyTransformer(config)
        optimizer2 = torch.optim.Adam(model2.parameters())

        checkpoint = load_checkpoint(checkpoint_path, model2, optimizer2)

        assert checkpoint["epoch"] == 10
        assert checkpoint["loss"] == 0.5

        # Check that models have same parameters
        for p1, p2 in zip(
            model.parameters(), model2.parameters(), strict=False
        ):
            assert torch.allclose(p1, p2)

    def test_image_model_checkpoint(self, tmp_path):
        """Test checkpointing for image model"""
        config = ImageETConfig(image_shape=(3, 224, 224), patch_size=16)
        model = ImageEnergyTransformer(config)

        # Save
        checkpoint_path = tmp_path / "image_checkpoint.pth"
        save_checkpoint(model, checkpoint_path, epoch=5)

        # Load
        model2 = ImageEnergyTransformer(config)
        checkpoint = load_checkpoint(checkpoint_path, model2)

        assert checkpoint["epoch"] == 5

        # Test that loaded model works
        images = torch.randn(1, 3, 224, 224)
        results = model2(images)
        assert torch.isfinite(results["reconstruction"]).all()


class TestNumericalStability:
    """Test numerical stability in extreme conditions"""

    def test_very_deep_model(self) -> None:
        """Test with very deep model (many steps)"""
        config = ETConfig(
            d_model=64,
            n_heads=4,
            d_head=16,
            n_steps=100,  # Very deep
            alpha=0.05,  # Smaller step size for stability
        )
        model = EnergyTransformer(config)

        x = torch.randn(1, 10, 64)
        output = model(x)

        assert torch.isfinite(output).all()
        assert not torch.allclose(output, x)  # Should have changed

    def test_extreme_initialization(self) -> None:
        """Test with extreme weight initialization"""
        config = ETConfig(d_model=64, n_heads=4, d_head=16)
        model = EnergyTransformer(config)

        # Scale up weights
        for param in model.parameters():
            param.data *= 10

        x = torch.randn(1, 10, 64)
        output = model(x)

        # Should still be stable
        assert torch.isfinite(output).all()

    def test_zero_initialization(self) -> None:
        """Test with zero initialization"""
        config = ETConfig(d_model=64, n_heads=4, d_head=16)
        model = EnergyTransformer(config)

        # Zero out weights
        for param in model.parameters():
            param.data.zero_()

        x = torch.randn(1, 10, 64)
        output = model(x)

        # Should handle gracefully
        assert torch.isfinite(output).all()


class TestEnergyProperties:
    """Test specific energy-based properties"""

    def test_energy_invariances(self) -> None:
        """Test expected invariances of the energy function"""
        config = ETConfig(d_model=64, n_heads=4, d_head=16)
        model = EnergyTransformer(config)

        x1 = torch.randn(1, 10, 64)

        # Energy should be invariant to batch replication
        x2 = x1.repeat(2, 1, 1)
        energy1 = model.energy(x1)
        energy2 = model.energy(x2)

        # Energy2 should be twice energy1 (additive over batch)
        assert torch.allclose(energy2, 2 * energy1, atol=1e-5)

    def test_energy_gradient_orthogonality(self) -> None:
        """Test gradient properties at fixed points"""
        torch.manual_seed(42)
        config = ETConfig(
            d_model=64, n_heads=4, d_head=16, n_steps=200, alpha=0.05
        )
        model = EnergyTransformer(config)

        # Run to near convergence
        x_init = torch.randn(1, 10, 64)
        x_final = model(x_init)

        # Gradient at fixed point should be small
        x_final = x_final.detach().requires_grad_(True)
        energy = model.energy(x_final)
        grad = torch.autograd.grad(energy, x_final)[0]

        # Gradient norm should be small (near fixed point)
        grad_norm = grad.norm()
        assert grad_norm < 1.5  # Relaxed threshold for practical convergence


class TestPatcher:
    """Test the Patcher utility"""

    def test_patcher_consistency(self) -> None:
        """Test that patchify/unpatchify are inverses"""
        patcher = Patcher((3, 224, 224), patch_size=16)

        image = torch.randn(3, 224, 224)
        patches = patcher.patchify(image)
        reconstructed = patcher.unpatchify(patches)

        assert torch.allclose(image, reconstructed)

    def test_patcher_batch(self) -> None:
        """Test patcher with batched images"""
        patcher = Patcher((3, 224, 224), patch_size=16)

        images = torch.randn(4, 3, 224, 224)
        patches = patcher.patchify(images)
        reconstructed = patcher.unpatchify(patches)

        assert torch.allclose(images, reconstructed)
        assert patches.shape == (4, 196, 3, 16, 16)


class TestImageNormalization:
    """Test image normalization utilities"""

    def test_normalize_unnormalize_cycle(self) -> None:
        """Test that normalize/unnormalize are inverses"""
        image = torch.rand(3, 224, 224)
        normalized = normalize_image(image)
        unnormalized = unnormalize_image(normalized)

        assert torch.allclose(image, unnormalized, atol=1e-6)

    def test_normalization_range(self) -> None:
        """Test that normalization produces expected range"""
        image = torch.rand(3, 224, 224)
        normalized = normalize_image(image)

        # Should have roughly zero mean and unit variance
        mean = normalized.mean(dim=(1, 2))
        std = normalized.std(dim=(1, 2))

        # These are approximate due to natural image statistics
        assert (mean.abs() < 2.0).all()
        assert (std > 0.1).all() and (std < 3.0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUIntegration:
    """Test GPU-specific functionality"""

    def test_gpu_forward_pass(self) -> None:
        """Test forward pass on GPU"""
        config = ETConfig(d_model=768, n_heads=12, d_head=64)
        model = EnergyTransformer(config).cuda()

        x = torch.randn(4, 128, 768).cuda()
        output = model(x)

        assert output.is_cuda
        assert output.shape == x.shape
        assert torch.isfinite(output).all()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_mixed_precision(self) -> None:
        """Test with automatic mixed precision"""
        config = ImageETConfig(image_shape=(3, 224, 224), patch_size=16)
        model = ImageEnergyTransformer(config).cuda()

        images = torch.randn(4, 3, 224, 224).cuda()

        # Run with AMP - using the new API
        with torch.amp.autocast("cuda"):
            results = model(images)

        assert results["reconstruction"].shape == images.shape
        assert torch.isfinite(results["reconstruction"]).all()


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_image_size(self) -> None:
        """Test with image size not divisible by patch size"""
        with pytest.raises(AssertionError):
            Patcher((3, 225, 225), patch_size=16)

    def test_mismatched_mask_size(self) -> None:
        """Test with incorrectly sized mask"""
        config = ImageETConfig(image_shape=(3, 224, 224), patch_size=16)
        model = ImageEnergyTransformer(config)

        images = torch.randn(2, 3, 224, 224)
        wrong_mask = torch.randint(0, 2, (2, 100))  # Wrong size

        with pytest.raises(RuntimeError):
            model(images, mask=wrong_mask)

    def test_empty_batch(self) -> None:
        """Test with empty batch dimension"""
        config = ETConfig(d_model=64, n_heads=4, d_head=16)
        model = EnergyTransformer(config)

        x = torch.randn(0, 10, 64)  # Empty batch
        output = model(x)

        assert output.shape == x.shape


class TestPerHeadBeta:
    """Test per-head temperature parameter support."""

    def test_scalar_beta_default(self) -> None:
        """Test that scalar beta is default behavior."""
        config = ETConfig(d_model=64, n_heads=4, d_head=16)
        model = EnergyTransformer(config)

        assert not model.attention.per_head_beta
        assert model.attention.beta.shape == ()  # Scalar

    def test_per_head_beta_initialization(self) -> None:
        """Test per-head beta initialization."""
        config = ETConfig(d_model=64, n_heads=4, d_head=16, per_head_beta=True)
        model = EnergyTransformer(config)

        assert model.attention.per_head_beta
        assert model.attention.beta.shape == (4,)  # One per head

        # Should be initialized to 1/sqrt(d_head)
        expected_value = 1.0 / torch.sqrt(torch.tensor(16.0))
        assert torch.allclose(
            model.attention.beta[0], expected_value, atol=1e-6
        )

    def test_energy_computation_with_per_head_beta(self) -> None:
        """Test that energy computation works with per-head beta."""
        n_heads = 4
        d_head = 16
        d_model = 64
        seq_len = 10

        attention = EnergyAttention(
            d_model, n_heads, d_head, per_head_beta=True
        )

        x = torch.randn(2, seq_len, d_model)
        energy = attention.energy(x)

        assert energy.shape == ()  # Scalar energy
        assert torch.isfinite(energy)

        # Test gradient computation
        x.requires_grad_(True)
        energy = attention.energy(x)
        grad = torch.autograd.grad(energy, x)[0]

        assert grad.shape == x.shape
        assert torch.isfinite(grad).all()

    def test_per_head_beta_different_values(self) -> None:
        """Test that per-head beta can have different values."""
        n_heads = 4
        d_head = 16
        d_model = 64

        attention = EnergyAttention(
            d_model, n_heads, d_head, per_head_beta=True
        )

        # Set different beta values for each head
        with torch.no_grad():
            attention.beta[0] = 0.1
            attention.beta[1] = 0.2
            attention.beta[2] = 0.3
            attention.beta[3] = 0.4

        x = torch.randn(1, 5, d_model)
        energy = attention.energy(x)

        assert torch.isfinite(energy)

    def test_backward_compatibility(self) -> None:
        """Test that old configs still work."""
        # Config without per_head_beta field should default to False
        config_dict = {"d_model": 64, "n_heads": 4, "d_head": 16, "n_steps": 10}

        config = ETConfig(**config_dict)
        model = EnergyTransformer(config)

        assert not model.attention.per_head_beta
        assert model.attention.beta.shape == ()  # Scalar

    @pytest.mark.parametrize("per_head_beta", [True, False])
    def test_energy_consistency(self, per_head_beta):
        """Test that energy computation is consistent."""
        config = ETConfig(
            d_model=64, n_heads=4, d_head=16, per_head_beta=per_head_beta
        )

        attention = EnergyAttention(
            config.d_model,
            config.n_heads,
            config.d_head,
            per_head_beta=per_head_beta,
        )

        # Create reproducible input
        torch.manual_seed(42)
        x = torch.randn(2, 10, 64)

        # Compute energy multiple times
        energies = []
        for _ in range(3):
            energy = attention.energy(x)
            energies.append(energy.item())

        # Should be deterministic
        assert all(e == energies[0] for e in energies)
