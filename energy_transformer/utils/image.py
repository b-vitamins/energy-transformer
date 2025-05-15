"""Image processing utilities."""

from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor


class Patcher:
    """Utility class for converting between images and patches.

    Parameters
    ----------
    image_shape : tuple[int, int, int]
        Shape of images (C, H, W).
    patch_size : int
        Size of square patches.
    """

    def __init__(
        self, image_shape: tuple[int, int, int], patch_size: int
    ) -> None:
        """Initialize a Patcher for breaking images into and reconstructing from patches.

        Parameters
        ----------
        image_shape : tuple[int, int, int]
            The shape of each image as (channels, height, width).
        patch_size : int
            The height and width of each square patch.
        """
        self.image_shape: tuple[int, int, int] = image_shape
        self.patch_size: int = patch_size

        c, h, w = image_shape
        assert h % patch_size == 0, (
            f"Height {h} not divisible by patch_size {patch_size}"
        )
        assert w % patch_size == 0, (
            f"Width {w} not divisible by patch_size {patch_size}"
        )

        self.n_patches_h: int = h // patch_size
        self.n_patches_w: int = w // patch_size
        self.n_patches: int = self.n_patches_h * self.n_patches_w
        self.patch_shape: tuple[int, int, int] = (c, patch_size, patch_size)
        self.patch_dim: int = c * patch_size * patch_size

    @classmethod
    def from_image_shape(
        cls, image_shape: tuple[int, int, int], patch_size: int
    ) -> Patcher:
        """Create patcher from image shape.

        Parameters
        ----------
        image_shape : tuple[int, int, int]
            Shape of images (C, H, W).
        patch_size : int
            Size of square patches.

        Returns
        -------
        Patcher
        """
        return cls(image_shape, patch_size)

    def patchify(self, images: Tensor) -> Tensor:
        """Convert images to patches.

        Parameters
        ----------
        images : Tensor
            Shape (..., C, H, W).

        Returns
        -------
        Tensor
            Patches of shape (..., n_patches, C, patch_size, patch_size).
        """
        return rearrange(
            images,
            "... c (nh ph) (nw pw) -> ... (nh nw) c ph pw",
            ph=self.patch_size,
            pw=self.patch_size,
        )

    def unpatchify(self, patches: Tensor) -> Tensor:
        """Convert patches back to images.

        Parameters
        ----------
        patches : Tensor
            Shape (..., n_patches, C, patch_size, patch_size).

        Returns
        -------
        Tensor
            Images of shape (..., C, H, W).
        """
        return rearrange(
            patches,
            "... (nh nw) c ph pw -> ... c (nh ph) (nw pw)",
            nh=self.n_patches_h,
            nw=self.n_patches_w,
        )


# ImageNet normalization constants
IMAGENET_MEAN: Tensor = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD: Tensor = torch.tensor([0.229, 0.224, 0.225])


def normalize_image(
    image: Tensor,
    mean: Tensor = IMAGENET_MEAN,
    std: Tensor = IMAGENET_STD,
) -> Tensor:
    """Normalize image using given mean and std.

    Parameters
    ----------
    image : Tensor
        Image tensor of shape (..., C, H, W) or (..., H, W, C).
    mean : Tensor
        Mean values per channel.
    std : Tensor
        Standard deviation per channel.

    Returns
    -------
    Tensor
        Normalized image.
    """
    mean = mean.to(image.device)
    std = std.to(image.device)

    # HWC format when last dim == 3
    if image.shape[-1] == 3:
        return (image - mean) / std

    # CHW format
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    return (image - mean) / std


def unnormalize_image(
    image: Tensor,
    mean: Tensor = IMAGENET_MEAN,
    std: Tensor = IMAGENET_STD,
) -> Tensor:
    """Unnormalize image using given mean and std.

    Parameters
    ----------
    image : Tensor
        Normalized image tensor.
    mean : Tensor
        Mean values per channel.
    std : Tensor
        Standard deviation per channel.

    Returns
    -------
    Tensor
        Unnormalized image in range [0, 1].
    """
    mean = mean.to(image.device)
    std = std.to(image.device)

    if image.shape[-1] == 3:
        image = image * std + mean
    else:
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
        image = image * std + mean

    return torch.clamp(image, 0, 1)
