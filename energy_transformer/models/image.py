"""Image‐specific Energy Transformer model."""

from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from ..config import ImageETConfig
from ..layers import EnergyLayerNorm
from ..models.core import EnergyTransformer
from ..utils.image import Patcher


class ImageEnergyTransformer(nn.Module):
    """Energy Transformer for image processing tasks.

    This model handles image‐specific operations like patching, masking,
    and reconstruction using the Energy Transformer architecture.

    Parameters
    ----------
    config : ImageETConfig
        Configuration for the image model.
    """

    def __init__(self, config: ImageETConfig) -> None:
        super().__init__()
        self.config: ImageETConfig = config
        self.patcher: Patcher = Patcher.from_image_shape(
            image_shape=config.image_shape,
            patch_size=config.patch_size,
        )
        self.transformer: EnergyTransformer = EnergyTransformer(
            config.et_config
        )
        # Patch encoder & decoder
        self.encoder: nn.Linear = nn.Linear(
            in_features=config.patch_dim,
            out_features=config.et_config.d_model,
        )
        self.decoder: nn.Linear = nn.Linear(
            in_features=config.et_config.d_model,
            out_features=config.patch_dim,
        )
        # Special tokens & positional embedding
        d_model = config.et_config.d_model
        self.cls_token: Tensor = nn.Parameter(torch.randn(d_model) * 0.02)
        self.mask_token: Tensor = nn.Parameter(torch.randn(d_model) * 0.02)
        self.pos_embed: Tensor = nn.Parameter(
            torch.randn(1 + config.n_patches, d_model) * 0.02
        )
        # Final layer norm
        self.output_norm: EnergyLayerNorm = EnergyLayerNorm(
            d_model=d_model, use_bias=True
        )

    def encode_patches(self, patches: Tensor) -> Tensor:
        """Encode image patches into transformer tokens.

        Parameters
        ----------
        patches : Tensor
            Shape (..., n_patches, C, patch_size, patch_size).

        Returns
        -------
        Tensor
            Shape (..., n_patches, d_model).
        """
        flat: Tensor = rearrange(patches, "... c h w -> ... (c h w)")
        return self.encoder(flat)

    def decode_tokens(self, tokens: Tensor) -> Tensor:
        """Decode transformer tokens back into image patches.

        Parameters
        ----------
        tokens : Tensor
            Shape (..., n_patches, d_model).

        Returns
        -------
        Tensor
            Reconstructed patches, shape (..., n_patches, C, patch_size, patch_size).
        """
        normed: Tensor = self.output_norm(tokens)
        flat: Tensor = self.decoder(normed)
        c, h, w = self.patcher.patch_shape
        # rearrange returns Any, so cast to Tensor for MyPy
        patches: Tensor = rearrange(
            flat, "... (c h w) -> ... c h w", c=c, h=h, w=w
        )
        return patches

    def prepare_tokens(
        self,
        tokens: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Add CLS token, apply masking, and add positional embeddings.

        Parameters
        ----------
        tokens : Tensor
            Shape (batch_size, n_patches, d_model).
        mask : Tensor or None
            Binary mask shape (batch_size, n_patches), where 1 denotes masked patches.

        Returns
        -------
        Tensor
            Shape (batch_size, 1 + n_patches, d_model).
        """
        batch_size = tokens.size(0)
        if mask is not None:
            mask_f: Tensor = mask.unsqueeze(-1).float()
            tokens = tokens * (1.0 - mask_f) + self.mask_token * mask_f

        cls = (
            self.cls_token.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
        )
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed
        return tokens

    def forward(
        self,
        images: Tensor,
        mask: Tensor | None = None,
        return_cls_token: bool = False,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Forward pass: patchify → encode → transform → decode → unpatchify.

        Parameters
        ----------
        images : Tensor
            Shape (batch_size, C, H, W).
        mask : Tensor or None
            Shape (batch_size, n_patches).
        return_cls_token : bool
            If True, include the CLS token in output.

        Returns
        -------
        results : dict
            - 'reconstruction': Tensor of reconstructed images (batch_size, C, H, W)
            - 'cls_token': Tensor of shape (batch_size, d_model) if requested
        """
        patches: Tensor = self.patcher.patchify(images)
        original: Tensor | None = patches.clone() if mask is not None else None

        tokens = self.encode_patches(patches)
        tokens = self.prepare_tokens(tokens, mask)
        output_tokens = self.transformer(tokens, **kwargs)

        cls_tok: Tensor = output_tokens[:, 0]
        patch_toks: Tensor = output_tokens[:, 1:]
        recon_patches = self.decode_tokens(patch_toks)

        if mask is not None and original is not None:
            m = mask.view(mask.size(0), -1, 1, 1, 1)
            recon_patches = recon_patches * m + original * (~m)

        reconstruction: Tensor = self.patcher.unpatchify(recon_patches)
        results: dict[str, Tensor] = {"reconstruction": reconstruction}
        if return_cls_token:
            results["cls_token"] = cls_tok
        return results

    def create_random_mask(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Generate a random binary mask over patches.

        Parameters
        ----------
        batch_size : int
        device : torch.device

        Returns
        -------
        Tensor
            Boolean mask of shape (batch_size, n_patches).
        """
        n_patches: int = self.config.n_patches
        n_mask: int = self.config.n_mask

        mask = torch.zeros(
            batch_size, n_patches, device=device, dtype=torch.bool
        )
        for i in range(batch_size):
            idx: Tensor = torch.randperm(n_patches, device=device)[:n_mask]
            mask[i, idx] = True
        return mask
