from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:  # pragma: no cover
    from .vision.viet import VisionEnergyTransformer
    from .vision.viset import VisionSimplicialTransformer
    from .vision.vit import VisionTransformer


@dataclass(slots=True)
class ViTConfig:
    """Configuration for :class:`VisionTransformer`."""

    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    num_classes: int = 1000
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0

    def apply_overrides(self, **overrides: Any) -> Self:
        """Apply runtime overrides to this configuration."""
        for key, value in overrides.items():
            if not hasattr(self, key):
                msg = f"{self.__class__.__name__}: unknown field {key!r}"
                raise TypeError(msg)
            setattr(self, key, value)
        return self

    def build(self) -> VisionTransformer:
        """Construct a :class:`VisionTransformer` from this configuration."""
        from .vision.vit import VisionTransformer

        return VisionTransformer(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            num_classes=self.num_classes,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            drop_rate=self.drop_rate,
        )


@dataclass(slots=True)
class ViETConfig:
    """Configuration for :class:`VisionEnergyTransformer`."""

    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    num_classes: int = 1000
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    hopfield_hidden_dim: int = 3072
    et_steps: int = 4
    drop_rate: float = 0.0

    def apply_overrides(self, **overrides: Any) -> Self:
        """Apply runtime overrides to this configuration."""
        for key, value in overrides.items():
            if not hasattr(self, key):
                msg = f"{self.__class__.__name__}: unknown field {key!r}"
                raise TypeError(msg)
            setattr(self, key, value)
        return self

    def build(self) -> VisionEnergyTransformer:
        """Construct a :class:`VisionEnergyTransformer` from this configuration."""
        from .vision.viet import VisionEnergyTransformer

        return VisionEnergyTransformer(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            num_classes=self.num_classes,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            hopfield_hidden_dim=self.hopfield_hidden_dim,
            et_steps=self.et_steps,
            drop_rate=self.drop_rate,
        )


@dataclass(slots=True)
class ViSETConfig:
    """Configuration for :class:`VisionSimplicialTransformer`."""

    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    num_classes: int = 1000
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    hopfield_hidden_dim: int = 3072
    et_steps: int = 4
    drop_rate: float = 0.0
    hopfield_beta: float = 0.1
    triangle_fraction: float = 0.5

    def apply_overrides(self, **overrides: Any) -> Self:
        """Apply runtime overrides to this configuration."""
        for key, value in overrides.items():
            if not hasattr(self, key):
                msg = f"{self.__class__.__name__}: unknown field {key!r}"
                raise TypeError(msg)
            setattr(self, key, value)
        return self

    def build(self) -> VisionSimplicialTransformer:
        """Construct a :class:`VisionSimplicialTransformer` from this configuration."""
        from .vision.viset import VisionSimplicialTransformer

        return VisionSimplicialTransformer(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            num_classes=self.num_classes,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            hopfield_hidden_dim=self.hopfield_hidden_dim,
            et_steps=self.et_steps,
            drop_rate=self.drop_rate,
            hopfield_beta=self.hopfield_beta,
            triangle_fraction=self.triangle_fraction,
        )


__all__ = ["ViETConfig", "ViSETConfig", "ViTConfig"]
