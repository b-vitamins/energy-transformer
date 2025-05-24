"""Vision Energy Transformer (ViET) implementation.

Architectural Mapping:
=====================

| ViT Component          | ViET Component         | Status     |
|------------------------|------------------------|------------|
| Patch Embedding        | PatchEmbedding         | Identical  |
| CLS Token              | CLSToken               | Identical  |
| Positional Embedding   | PositionalEmbedding2D  | Identical  |
| Positional Dropout     | nn.Dropout             | Identical  |
| Transformer Encoder    | EnergyTransformer      | Replaced   |
| ├─ Standard Attention  | ├─ Energy Attention    | Replaced   |
| ├─ Standard LayerNorm  | ├─ Energy LayerNorm    | Replaced   |
| └─ MLP                 | └─ Hopfield Network    | Replaced   |
| Final LayerNorm        | nn.LayerNorm           | Identical  |
| Classification Head    | ClassificationHead     | Identical  |
"""

from __future__ import annotations

from typing import Any, cast

import torch.nn as nn
from torch import Tensor

from ...layers.attention import MultiHeadEnergyAttention
from ...layers.embeddings import PatchEmbedding, PositionalEmbedding2D
from ...layers.heads import ClassificationHead, FeatureHead
from ...layers.hopfield import HopfieldNetwork
from ...layers.layer_norm import LayerNorm
from ...layers.tokens import CLSToken
from ...models.base import EnergyTransformer
from .utils import VisionModelMixin, create_model_config

__all__ = [
    "VisionEnergyTransformer",
    "viet_tiny_patch16_224",
    "viet_small_patch16_224",
    "viet_base_patch16_224",
    "viet_large_patch16_224",
]


class VisionEnergyTransformer(nn.Module, VisionModelMixin):
    """Vision Energy Transformer - ViET.

    A Vision Transformer that uses Energy Transformer blocks instead of
    standard Transformer blocks. Maintains complete architectural fidelity
    to the original ViT except for the core computation mechanism:

    - Standard Attention → Multi-Head Energy Attention
    - Standard LayerNorm → Energy-based LayerNorm
    - MLP layers → Hopfield Networks
    - Feedforward computation → Energy minimization

    Parameters
    ----------
    img_size : int, default=224
        Input image size (assumed square).
    patch_size : int, default=16
        Size of image patches (assumed square).
    in_chans : int, default=3
        Number of input channels.
    embed_dim : int, default=768
        Embedding dimension.
    depth : int, default=12
        Number of Energy Transformer blocks.
    num_classes : int, default=1000
        Number of output classes.
    num_heads : int, default=12
        Number of attention heads.
    head_dim : int, default=64
        Dimension of each attention head.
    hopfield_hidden_dim : int, default=3072
        Hidden dimension for Hopfield networks. Typically 4 * embed_dim.
    et_steps : int, default=4
        Number of energy optimization steps per block.
    et_alpha : float, default=0.125
        Step size for energy optimization.
    drop_rate : float, default=0.0
        Dropout rate for positional embeddings.
    representation_size : int, optional
        Size of representation layer before classification head.
        If None, uses embed_dim directly.

    Examples
    --------
    >>> # Standard ViET-Base
    >>> model = VisionEnergyTransformer()
    >>> logits = model(torch.randn(2, 3, 224, 224))
    >>> print(logits.shape)  # (2, 1000)

    >>> # Feature extraction
    >>> features = model(images, return_features=True)
    >>> print(features.shape)  # (2, 768)

    >>> # With energy analysis
    >>> result = model(images, return_energy_info=True)
    >>> print(result['logits'].shape, result['energy_info'])
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_classes: int = 1000,
        num_heads: int = 12,
        head_dim: int = 64,
        hopfield_hidden_dim: int = 3072,
        et_steps: int = 4,
        et_alpha: float = 0.125,
        drop_rate: float = 0.0,
        representation_size: int | None = None,
    ) -> None:
        """Initialize Vision Energy Transformer."""
        super().__init__()

        # Store configuration
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_classes = num_classes

        # Patch embedding (exactly like ViT)
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # CLS token (exactly like ViT)
        self.cls_token = CLSToken(embed_dim)

        # Positional embeddings (exactly like ViT)
        # Include CLS token in positional embedding
        self.pos_embed = PositionalEmbedding2D(
            num_patches=num_patches,
            embed_dim=embed_dim,
            include_cls=True,
        )

        # Positional dropout (exactly like ViT)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Energy Transformer blocks (replaces Transformer encoders)
        self.et_blocks = nn.ModuleList(
            [
                EnergyTransformer(
                    layer_norm=LayerNorm(embed_dim),
                    attention=MultiHeadEnergyAttention(
                        in_dim=embed_dim,
                        num_heads=num_heads,
                        head_dim=head_dim,
                    ),
                    hopfield=HopfieldNetwork(
                        in_dim=embed_dim,
                        hidden_dim=hopfield_hidden_dim,
                    ),
                    steps=et_steps,
                    α=et_alpha,
                )
                for _ in range(depth)
            ]
        )

        # Final layer normalization (exactly like ViT)
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head (exactly like ViT)
        self.head = ClassificationHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            representation_size=representation_size,
            drop_rate=drop_rate,
            use_cls_token=True,
        )

        # Feature extraction head
        self.feature_head = FeatureHead(use_cls_token=True)

        # Initialize weights following ViT conventions
        self.init_vit_weights()

        # Special initialization for CLS token and positional embeddings
        nn.init.trunc_normal_(self.cls_token.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed.pos_embed, std=0.02)

    def forward(
        self,
        x: Tensor,
        return_features: bool = False,
        return_energy_info: bool = False,
        et_kwargs: dict[str, Any] | None = None,
    ) -> Tensor | dict[str, Any]:
        """Forward pass through Vision Energy Transformer.

        Parameters
        ----------
        x : Tensor
            Input images of shape (B, C, H, W).
        return_features : bool, default=False
            If True, return features instead of logits.
        return_energy_info : bool, default=False
            If True, return energy information from ET blocks.
        et_kwargs : dict, optional
            Additional arguments passed to Energy Transformer blocks.

        Returns
        -------
        Union[Tensor, dict]
            If return_energy_info is False:
                - Tensor of shape (B, num_classes) if return_features is False
                - Tensor of shape (B, embed_dim) if return_features is True
            If return_energy_info is True:
                - Dictionary with 'logits'/'features' and 'energy_info' keys
        """
        x.shape[0]
        et_kwargs = et_kwargs or {}

        # Validate input size
        if x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"Input size {x.shape[-2:]} doesn't match model size "
                f"({self.img_size}, {self.img_size})"
            )

        # 1. Patch embedding (ViT Step 1)
        x = self.patch_embed(x)  # (B, N, D)

        # 2. Prepend CLS token (ViT Step 2)
        x = self.cls_token(x)  # (B, N+1, D)

        # 3. Add positional embeddings (ViT Step 3)
        x = self.pos_embed(x)  # (B, N+1, D)
        x = self.pos_drop(x)

        # 4. Energy Transformer blocks (replaces ViT Transformer encoders)
        energy_info: dict[str, Any] = {}
        if return_energy_info:
            block_energies = []
            block_trajectories = []

        for _i, et_block in enumerate(self.et_blocks):
            if return_energy_info:
                # Get energy information from this block
                result = et_block(
                    x, return_energy=True, return_trajectory=True, **et_kwargs
                )
                if hasattr(result, "tokens"):
                    x = result.tokens
                    if result.final_energy is not None:
                        block_energies.append(result.final_energy.item())
                    if result.trajectory is not None:
                        block_trajectories.append(
                            result.trajectory.cpu().numpy()
                        )
                else:
                    x = result
            else:
                # Standard forward pass
                x = et_block(x, **et_kwargs)  # (B, N+1, D)

        if return_energy_info:
            energy_info = {
                "block_energies": block_energies,
                "block_trajectories": block_trajectories,
                "total_energy": (
                    sum(block_energies) if block_energies else None
                ),
            }

        # 5. Final layer normalization (ViT Step 4)
        x = self.norm(x)  # (B, N+1, D)

        # 6. Extract features or classify (ViT Step 5)
        if return_features:
            features = cast(Tensor, self.feature_head(x))  # (B, D)
            if return_energy_info:
                return {"features": features, "energy_info": energy_info}
            return features
        else:
            logits = cast(Tensor, self.head(x))  # (B, num_classes)
            if return_energy_info:
                return {"logits": logits, "energy_info": energy_info}
            return logits

    def get_attention_maps(
        self, x: Tensor, layer_idx: int | None = None
    ) -> dict[str, Any]:
        """Extract attention maps for visualization.

        Note: This returns energy gradients rather than attention weights,
        as Energy Transformers don't compute explicit attention matrices.

        Parameters
        ----------
        x : Tensor
            Input images of shape (B, C, H, W).
        layer_idx : int, optional
            Specific layer to extract from. If None, returns all layers.

        Returns
        -------
        dict
            Dictionary with attention-like information from energy gradients.
        """
        # This would require hooking into the energy computation
        # For now, return a placeholder
        return {
            "message": "Energy-based attention maps require gradient analysis"
        }

    def freeze_patch_embed(self) -> None:
        """Freeze patch embedding parameters for fine-tuning."""
        for param in self.patch_embed.parameters():
            param.requires_grad = False

    def unfreeze_patch_embed(self) -> None:
        """Unfreeze patch embedding parameters."""
        for param in self.patch_embed.parameters():
            param.requires_grad = True

    def freeze_backbone(self) -> None:
        """Freeze all parameters except classification head."""
        for name, param in self.named_parameters():
            if not name.startswith("head."):
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


# Factory functions for standard configurations


def viet_tiny_patch16_224(**kwargs: Any) -> VisionEnergyTransformer:
    """ViET-Tiny with 16x16 patches on 224x224 images.

    Parameters
    ----------
    **kwargs
        Additional arguments passed to VisionEnergyTransformer.

    Returns
    -------
    VisionEnergyTransformer
        ViET-Tiny model instance.
    """
    config = create_model_config("tiny", **kwargs)
    return VisionEnergyTransformer(**config)


def viet_small_patch16_224(**kwargs: Any) -> VisionEnergyTransformer:
    """ViET-Small with 16x16 patches on 224x224 images.

    Parameters
    ----------
    **kwargs
        Additional arguments passed to VisionEnergyTransformer.

    Returns
    -------
    VisionEnergyTransformer
        ViET-Small model instance.
    """
    config = create_model_config("small", **kwargs)
    return VisionEnergyTransformer(**config)


def viet_base_patch16_224(**kwargs: Any) -> VisionEnergyTransformer:
    """ViET-Base with 16x16 patches on 224x224 images.

    Parameters
    ----------
    **kwargs
        Additional arguments passed to VisionEnergyTransformer.

    Returns
    -------
    VisionEnergyTransformer
        ViET-Base model instance.
    """
    config = create_model_config("base", **kwargs)
    return VisionEnergyTransformer(**config)


def viet_large_patch16_224(**kwargs: Any) -> VisionEnergyTransformer:
    """ViET-Large with 16x16 patches on 224x224 images.

    Parameters
    ----------
    **kwargs
        Additional arguments passed to VisionEnergyTransformer.

    Returns
    -------
    VisionEnergyTransformer
        ViET-Large model instance.
    """
    config = create_model_config("large", **kwargs)
    return VisionEnergyTransformer(**config)
