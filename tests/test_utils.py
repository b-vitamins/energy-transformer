"""Tests for utility functions."""

import pytest
import torch

from energy_transformer.utils import Patcher, normalize_image, unnormalize_image


class TestPatcher:
    def test_init(self) -> None:
        patcher = Patcher((3, 224, 224), patch_size=16)
        assert patcher.n_patches == 196  # (224/16) * (224/16)
        assert patcher.patch_shape == (3, 16, 16)
        assert patcher.patch_dim == 3 * 16 * 16

    def test_patchify_unpatchify(self) -> None:
        patcher = Patcher((3, 224, 224), patch_size=16)

        # Test single image
        image = torch.randn(3, 224, 224)
        patches = patcher.patchify(image)
        assert patches.shape == (196, 3, 16, 16)

        reconstructed = patcher.unpatchify(patches)
        assert reconstructed.shape == image.shape
        assert torch.allclose(reconstructed, image)

        # Test batch of images
        images = torch.randn(4, 3, 224, 224)
        patches = patcher.patchify(images)
        assert patches.shape == (4, 196, 3, 16, 16)

        reconstructed = patcher.unpatchify(patches)
        assert reconstructed.shape == images.shape
        assert torch.allclose(reconstructed, images)

    def test_invalid_dimensions(self) -> None:
        with pytest.raises(AssertionError):
            # Height not divisible by patch_size
            Patcher((3, 225, 224), patch_size=16)

        with pytest.raises(AssertionError):
            # Width not divisible by patch_size
            Patcher((3, 224, 225), patch_size=16)


class TestImageNormalization:
    def test_normalize_unnormalize(self) -> None:
        # Test CHW format
        image_chw = torch.rand(3, 224, 224)
        normalized = normalize_image(image_chw)
        unnormalized = unnormalize_image(normalized)

        # Check shape preservation
        assert normalized.shape == image_chw.shape
        assert unnormalized.shape == image_chw.shape

        # Check range
        assert unnormalized.min() >= 0
        assert unnormalized.max() <= 1

        # Test HWC format
        image_hwc = torch.rand(224, 224, 3)
        normalized = normalize_image(image_hwc)
        unnormalized = unnormalize_image(normalized)

        assert normalized.shape == image_hwc.shape
        assert unnormalized.shape == image_hwc.shape

    def test_batch_normalization(self) -> None:
        # Test batch of images
        images = torch.rand(4, 3, 224, 224)
        normalized = normalize_image(images)
        unnormalized = unnormalize_image(normalized)

        assert normalized.shape == images.shape
        assert unnormalized.shape == images.shape
        assert unnormalized.min() >= 0
        assert unnormalized.max() <= 1
