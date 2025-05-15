"""Tests for Energy Transformer models."""

import torch

from energy_transformer import (
    EnergyTransformer,
    ETConfig,
    ImageEnergyTransformer,
    ImageETConfig,
)


class TestEnergyTransformer:
    def test_init(self) -> None:
        config = ETConfig()
        model = EnergyTransformer(config)
        assert model.config == config
        assert hasattr(model, "attention")
        assert hasattr(model, "hopfield")
        assert hasattr(model, "layer_norm")

    def test_energy(self) -> None:
        config = ETConfig()
        model = EnergyTransformer(config)
        x = torch.randn(2, 10, config.d_model)
        energy = model.energy(x)
        assert energy.shape == ()
        assert energy.dtype == torch.float32

    def test_forward(self) -> None:
        config = ETConfig(n_steps=10)
        model = EnergyTransformer(config)
        x = torch.randn(2, 10, config.d_model)

        # Basic forward
        output = model(x)
        assert output.shape == x.shape

        # Forward with trajectory
        output, trajectory = model(x, return_trajectory=True)
        assert output.shape == x.shape
        assert trajectory.shape == (config.n_steps + 1, 2, 10, config.d_model)

    def test_energy_minimization(self) -> None:
        config = ETConfig(n_steps=50)
        model = EnergyTransformer(config)
        x = torch.randn(2, 10, config.d_model)

        # Get energy trajectory
        final_state, energies = model.compute_energy_trajectory(x)

        # Energy should decrease
        assert energies[-1] < energies[0]
        assert len(energies) == config.n_steps


class TestImageEnergyTransformer:
    def test_init(self) -> None:
        config = ImageETConfig()
        model = ImageEnergyTransformer(config)
        assert model.config == config
        assert hasattr(model, "transformer")
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")
        assert hasattr(model, "patcher")

    def test_patching(self) -> None:
        config = ImageETConfig()
        model = ImageEnergyTransformer(config)

        # Test encode/decode patches - use correct patch dimensions
        # For a 224x224 image with patch_size=16, each patch should be 16x16
        patch_shape = model.patcher.patch_shape  # (3, 16, 16)
        patches = torch.randn(2, config.n_patches, *patch_shape)

        # Test encoding
        encoded = model.encode_patches(patches)
        assert encoded.shape == (2, config.n_patches, config.et_config.d_model)

        # Test decoding
        decoded = model.decode_tokens(encoded)
        assert decoded.shape == patches.shape

    def test_forward(self) -> None:
        config = ImageETConfig()
        model = ImageEnergyTransformer(config)

        # Test without mask
        images = torch.randn(2, *config.image_shape)
        results = model(images)
        assert "reconstruction" in results
        assert results["reconstruction"].shape == images.shape

        # Test with mask
        mask = model.create_random_mask(2, images.device)
        results = model(images, mask=mask)
        assert results["reconstruction"].shape == images.shape

        # Test with CLS token
        results = model(images, return_cls_token=True)
        assert "cls_token" in results
        assert results["cls_token"].shape == (2, config.et_config.d_model)

    def test_mask_creation(self) -> None:
        config = ImageETConfig()
        model = ImageEnergyTransformer(config)

        device = torch.device("cpu")
        mask = model.create_random_mask(4, device)

        assert mask.shape == (4, config.n_patches)
        assert mask.dtype == torch.bool
        # Check number of masked patches
        assert (mask.sum(dim=1) == config.n_mask).all()
