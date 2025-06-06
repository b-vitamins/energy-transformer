"""Vision Simplicial Energy Transformer (ViSET) implementation."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor, nn

from energy_transformer.layers.attention import MultiheadEnergyAttention
from energy_transformer.layers.embeddings import ConvPatchEmbed, PosEmbed2D
from energy_transformer.layers.heads import ClassifierHead
from energy_transformer.layers.layer_norm import EnergyLayerNorm
from energy_transformer.layers.simplicial import SimplicialHopfieldNetwork
from energy_transformer.layers.types import ActivationType
from energy_transformer.models.base import EnergyTransformer
from energy_transformer.utils.optimizers import SGD


class VisionSimplicialTransformer(nn.Module):
    """Vision Simplicial Energy Transformer (ViSET)."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        _head_dim: int,
        hopfield_hidden_dim: int,
        et_steps: int,
        et_alpha: float,
        order: int = 3,
        drop_rate: float = 0.0,
        _representation_size: int | None = None,
        hopfield_activation: str = "relu",
        hopfield_beta: float = 0.01,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_classes = num_classes
        self.order = order

        self.patch_embed = ConvPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = PosEmbed2D(
            num_patches=num_patches, embed_dim=embed_dim, cls_token=True
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.et_blocks = nn.ModuleList(
            [
                EnergyTransformer(
                    layer_norm=EnergyLayerNorm(embed_dim),
                    attention=MultiheadEnergyAttention(
                        embed_dim=embed_dim, num_heads=num_heads
                    ),
                    hopfield=SimplicialHopfieldNetwork(
                        embed_dim,
                        hidden_dim=hopfield_hidden_dim,
                        order=order,
                        activation=cast(ActivationType, hopfield_activation),
                        beta=hopfield_beta,
                    ),
                    steps=et_steps,
                    optimizer=SGD(alpha=et_alpha),
                )
                for _ in range(depth)
            ]
        )

        self.norm = EnergyLayerNorm(embed_dim)
        self.head = ClassifierHead(
            in_features=embed_dim,
            num_classes=num_classes,
            pool_type="token",
            drop_rate=drop_rate,
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed.pos_embed, std=0.02)

    def _process_et_blocks(
        self,
        x: Tensor,
        return_energy_info: bool,
        _et_kwargs: dict[str, Any],
    ) -> tuple[Tensor, dict[str, Any]]:
        energy_info: dict[str, Any] = {}
        if not return_energy_info:
            for et_block in self.et_blocks:
                x = et_block(x)
            return x, energy_info

        block_energies: list[float] = []
        block_trajectories: list[Any] = []
        for et_block in self.et_blocks:
            if hasattr(et_block, "_compute_energy"):
                x = et_block(x)
                block_energies.append(et_block._compute_energy(x).item())
            else:
                result = et_block(x, track="both")
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
        total_energy = sum(block_energies) if block_energies else None
        energy_info = {
            "block_energies": block_energies,
            "block_trajectories": block_trajectories,
            "total_energy": total_energy,
        }
        return x, energy_info

    def forward(
        self,
        x: Tensor,
        return_features: bool = False,
        return_energy_info: bool = False,
        et_kwargs: dict[str, Any] | None = None,
    ) -> Tensor | dict[str, Any]:
        """Forward pass of the model."""
        et_kwargs = et_kwargs or {}

        if x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"Input size {x.shape[-2:]} doesn't match model size ({self.img_size}, {self.img_size})"
            )

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_embed(x)
        x = self.pos_drop(x)

        x, energy_info = self._process_et_blocks(
            x, return_energy_info, et_kwargs
        )

        x = self.norm(x)

        if return_features:
            assert isinstance(x, Tensor)
            features = x[:, 0]
            if return_energy_info:
                return {"features": features, "energy_info": energy_info}
            return features
        logits: Tensor = self.head(x)
        if return_energy_info:
            return {"logits": logits, "energy_info": energy_info}
        return logits


def viset_tiny(**kwargs: Any) -> VisionSimplicialTransformer:
    """ViSET-Tiny configuration."""
    config: dict[str, Any] = {
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "_head_dim": 64,
        "hopfield_hidden_dim": 768,
        "et_steps": 4,
        "et_alpha": 0.125,
        "order": 3,
        "in_chans": 3,
    }
    config.update(kwargs)
    return VisionSimplicialTransformer(**config)


def viset_small(**kwargs: Any) -> VisionSimplicialTransformer:
    """ViSET-Small configuration."""
    config: dict[str, Any] = {
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "_head_dim": 64,
        "hopfield_hidden_dim": 1536,
        "et_steps": 4,
        "et_alpha": 0.125,
        "order": 3,
        "in_chans": 3,
    }
    config.update(kwargs)
    return VisionSimplicialTransformer(**config)


def viset_base(**kwargs: Any) -> VisionSimplicialTransformer:
    """ViSET-Base configuration."""
    config: dict[str, Any] = {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "_head_dim": 64,
        "hopfield_hidden_dim": 3072,
        "et_steps": 4,
        "et_alpha": 0.125,
        "order": 3,
        "in_chans": 3,
    }
    config.update(kwargs)
    return VisionSimplicialTransformer(**config)


def viset_large(**kwargs: Any) -> VisionSimplicialTransformer:
    """ViSET-Large configuration."""
    config: dict[str, Any] = {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "_head_dim": 64,
        "hopfield_hidden_dim": 4096,
        "et_steps": 4,
        "et_alpha": 0.125,
        "order": 3,
        "in_chans": 3,
    }
    config.update(kwargs)
    return VisionSimplicialTransformer(**config)


def viset_tiny_cifar(
    num_classes: int = 100, **kwargs: Any
) -> VisionSimplicialTransformer:
    """ViSET-Tiny for CIFAR datasets."""
    config: dict[str, Any] = {
        "img_size": 32,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": num_classes,
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "_head_dim": 64,
        "hopfield_hidden_dim": 768,
        "et_steps": 4,
        "et_alpha": 0.125,
        "order": 3,
        "drop_rate": 0.1,
    }
    config.update(kwargs)
    return VisionSimplicialTransformer(**config)


def viset_small_cifar(
    num_classes: int = 100, **kwargs: Any
) -> VisionSimplicialTransformer:
    """ViSET-Small for CIFAR datasets."""
    config: dict[str, Any] = {
        "img_size": 32,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": num_classes,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "_head_dim": 64,
        "hopfield_hidden_dim": 1536,
        "et_steps": 4,
        "et_alpha": 0.125,
        "order": 3,
        "drop_rate": 0.1,
    }
    config.update(kwargs)
    return VisionSimplicialTransformer(**config)


def viset_2l_cifar(
    num_classes: int = 100, **kwargs: Any
) -> VisionSimplicialTransformer:
    """2-layer ViSET for CIFAR datasets."""
    config: dict[str, Any] = {
        "img_size": 32,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": num_classes,
        "embed_dim": 192,
        "depth": 2,
        "num_heads": 8,
        "_head_dim": 64,
        "hopfield_hidden_dim": 576,
        "et_steps": 6,
        "et_alpha": 10.0,
        "order": 3,
        "drop_rate": 0.1,
    }
    config.update(kwargs)
    return VisionSimplicialTransformer(**config)


def viset_4l_cifar(
    num_classes: int = 100, **kwargs: Any
) -> VisionSimplicialTransformer:
    """4-layer ViSET for CIFAR datasets."""
    config: dict[str, Any] = {
        "img_size": 32,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": num_classes,
        "embed_dim": 192,
        "depth": 4,
        "num_heads": 8,
        "_head_dim": 64,
        "hopfield_hidden_dim": 576,
        "et_steps": 5,
        "et_alpha": 5.0,
        "order": 3,
        "drop_rate": 0.1,
    }
    config.update(kwargs)
    return VisionSimplicialTransformer(**config)


def viset_6l_cifar(
    num_classes: int = 100, **kwargs: Any
) -> VisionSimplicialTransformer:
    """6-layer ViSET for CIFAR datasets."""
    config: dict[str, Any] = {
        "img_size": 32,
        "patch_size": 4,
        "in_chans": 3,
        "num_classes": num_classes,
        "embed_dim": 192,
        "depth": 6,
        "num_heads": 8,
        "_head_dim": 64,
        "hopfield_hidden_dim": 576,
        "et_steps": 4,
        "et_alpha": 2.5,
        "order": 3,
        "drop_rate": 0.1,
    }
    config.update(kwargs)
    return VisionSimplicialTransformer(**config)
