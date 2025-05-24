"""Utilities for vision Energy Transformer models."""

from __future__ import annotations

from typing import Any, cast

import torch.nn as nn

__all__ = [
    "VisionModelMixin",
    "init_vit_weights",
    "create_model_config",
]


class VisionModelMixin:
    """Mixin class providing common functionality for vision models.

    Note: This mixin is designed to be used with classes that inherit from
    nn.Module. It provides vision-specific utilities and initialization.
    """

    def init_vit_weights(self) -> None:
        """Initialize weights following Vision Transformer conventions.

        This applies the same initialization scheme as the original ViT:
        - Truncated normal for linear layers and embeddings
        - Zero initialization for biases
        - Zero initialization for classification heads
        - Ones for layer norm weights, zeros for layer norm biases
        """
        module = cast(nn.Module, self)
        module.apply(self._init_vit_weights)

    def _init_vit_weights(self, module: nn.Module) -> None:
        """Initialize a single module following ViT conventions."""
        if isinstance(module, nn.Linear):
            # Truncated normal initialization for linear layers
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            # Standard layer norm initialization
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv2d):
            # Truncated normal for conv layers (patch embedding)
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_model_info(self) -> dict[str, Any]:
        """Get model configuration information.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing model configuration details.
        """
        module = cast(nn.Module, self)
        info = {
            "model_type": module.__class__.__name__,
            "num_parameters": sum(p.numel() for p in module.parameters()),
            "num_trainable_parameters": sum(
                p.numel() for p in module.parameters() if p.requires_grad
            ),
        }

        # Add patch embedding info if available
        if hasattr(self, "patch_embed"):
            info.update(
                {
                    "img_size": self.patch_embed.img_size,
                    "patch_size": self.patch_embed.patch_size,
                    "num_patches": self.patch_embed.num_patches,
                }
            )

        # Add depth info if available
        if hasattr(self, "et_blocks"):
            info["depth"] = len(self.et_blocks)

        return info


def init_vit_weights(module: nn.Module) -> None:
    """Standalone function to initialize weights following ViT conventions.

    Parameters
    ----------
    module : nn.Module
        Module to initialize.
    """
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    elif isinstance(module, nn.Conv2d):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def create_model_config(
    size: str = "base",
    img_size: int = 224,
    patch_size: int = 16,
    num_classes: int = 1000,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create standardized model configuration.

    Parameters
    ----------
    size : str, default="base"
        Model size variant ("tiny", "small", "base", "large").
    img_size : int, default=224
        Input image size.
    patch_size : int, default=16
        Patch size for patch embedding.
    num_classes : int, default=1000
        Number of output classes.
    **kwargs : Any
        Additional configuration parameters.

    Returns
    -------
    Dict[str, Any]
        Model configuration dictionary.

    Raises
    ------
    ValueError
        If size is not recognized.
    """
    # Standard ViT configurations - explicitly typed to allow mixed value types
    configs: dict[str, dict[str, Any]] = {
        "tiny": {
            "embed_dim": 192,
            "depth": 12,
            "num_heads": 3,
            "hopfield_hidden_dim": 768,  # 4 * embed_dim
        },
        "small": {
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "hopfield_hidden_dim": 1536,  # 4 * embed_dim
        },
        "base": {
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "hopfield_hidden_dim": 3072,  # 4 * embed_dim
        },
        "large": {
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "hopfield_hidden_dim": 4096,  # 4 * embed_dim
        },
    }

    if size not in configs:
        available_sizes = ", ".join(configs.keys())
        raise ValueError(f"Unknown size '{size}'.Available: {available_sizes}")

    # Get base configuration and explicitly type it
    config: dict[str, Any] = configs[size].copy()

    # Add common parameters
    config.update(
        {
            "img_size": img_size,
            "patch_size": patch_size,
            "num_classes": num_classes,
            "in_chans": 3,  # RGB by default
            # Energy Transformer specific defaults
            "head_dim": 64,  # Standard attention head dimension
            "et_steps": 4,  # Conservative number of optimization steps
            "et_alpha": 0.125,  # Standard step size
            # ViT compatibility defaults
            "drop_rate": 0.0,
            "representation_size": None,
        }
    )

    # Override with any user-provided kwargs
    config.update(kwargs)

    return config
