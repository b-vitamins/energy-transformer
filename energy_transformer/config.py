"""Configuration classes for Energy Transformer."""

from dataclasses import dataclass, field


@dataclass
class ETConfig:
    """Configuration for Energy Transformer.

    Args:
        d_model: Model dimension (token dimension)
        d_head: Dimension per attention head
        n_heads: Number of attention heads
        scale_memories: Scale factor for number of memories relative to d_model
        eps: Small epsilon for numerical stability
        alpha: Step size for gradient descent
        n_steps: Number of gradient descent steps
        per_head_beta: Whether to use per-head temperature parameters
    """

    d_model: int = 768
    d_head: int = 64
    n_heads: int = 12
    scale_memories: float = 4.0
    eps: float = 1e-5
    alpha: float = 0.1
    n_steps: int = 12
    per_head_beta: bool = False

    @property
    def n_memories(self) -> int:
        """Number of memories in Hopfield network."""
        return int(self.scale_memories * self.d_model)


@dataclass
class ImageETConfig:
    """Configuration for Image Energy Transformer.

    Args:
        image_shape: Shape of input images (C, H, W)
        patch_size: Size of image patches
        n_mask: Number of patches to mask
        et_config: Configuration for base ET model
    """

    image_shape: tuple[int, int, int] = (3, 224, 224)
    patch_size: int = 16
    n_mask: int = 100
    et_config: ETConfig = field(default_factory=ETConfig)

    @property
    def n_patches(self) -> int:
        """Number of patches in the image."""
        _, h, w = self.image_shape
        return (h // self.patch_size) * (w // self.patch_size)

    @property
    def patch_dim(self) -> int:
        """Dimension of each patch."""
        c, _, _ = self.image_shape
        return c * self.patch_size * self.patch_size
