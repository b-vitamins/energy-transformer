"""Vision models built on Energy-Transformer architecture."""

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ..models.base import REALISER_REGISTRY, EnergyTransformer
from ..utils.vision import (
    CLSToken,
    MaskToken,
    PatchEmbed,
    _to_pair,
)


class ViETEncoder(nn.Module):  # type: ignore
    """Task‑agnostic encoder for images using stacked Energy‑Transformer blocks.

    This class uses a builder pattern where pre-constructed components are
    passed in rather than built internally.
    """

    def __init__(
        self,
        patch_embedder: PatchEmbed,
        transformer_blocks: Sequence[EnergyTransformer],
        norm_layer: nn.Module,
        *,
        pos_encoder: nn.Module | None = None,
        cls_token: CLSToken | None = None,
        mask_token: MaskToken | None = None,
    ):
        """Initialize the encoder with pre-constructed components.

        Args:
            patch_embedder: Pre-constructed PatchEmbed module
            transformer_blocks: Sequence of pre-constructed EnergyTransformer blocks
            norm_layer: Final normalization layer (typically LayerNorm)
            pos_encoder: Optional positional encoding module (default: None)
                         Should be either Learnable2DPosEnc or SinCos2DPosEnc
            cls_token: Optional CLS token module (default: None)
            mask_token: Optional mask token module (default: None)
        """
        super().__init__()

        # Ensure norm_layer is compatible with the expected output shape
        if not isinstance(norm_layer, nn.LayerNorm):
            raise ValueError("norm_layer must be an instance of nn.LayerNorm")

        # Store all pre-constructed components
        self.patch_embed = patch_embedder
        self.pos_embed = pos_encoder
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.blocks = nn.ModuleList(transformer_blocks)
        self.norm = norm_layer

        # Store configuration flags
        self.cls_token_flag = cls_token is not None

    def forward(
        self,
        img: Tensor,
        *,
        patch_mask: Tensor
        | None = None,  # Boolean or float mask: 1/True for masked positions (B,N)
        return_sequence: bool = False,
    ) -> Tensor:
        """Forward pass through the encoder.

        Args:
            img: Input images of shape (B, C, H, W)
            patch_mask: Binary mask indicating which patches to mask (1=masked)
                        Shape should be (B, N) where N is the number of patches
            return_sequence: If True, returns the full sequence, otherwise CLS token

        Returns
        -------
            If return_sequence=False and cls_token is present:
                CLS token projection of shape (B, D)
            If return_sequence=True or cls_token is not present:
                Full sequence of shape (B, N, D) or (B, N+1, D) with CLS
        """
        # 1. Patchify
        x = self.patch_embed(img)  # shape: [B, N, D]

        # 2. Prepend CLS token if available
        if self.cls_token is not None:
            x = self.cls_token(x)  # shape: [B, N+1, D]

        # 3. Add positional encodings if available
        if self.pos_embed is not None:
            x = self.pos_embed(x)  # shape: [B, N+1, D] or [B, N, D]

        # 4. Apply masking if requested
        if self.mask_token is not None and patch_mask is not None:
            # If we have CLS token, don't mask it (offset patch_mask indices)
            if self.cls_token_flag:
                # Create mask with first position (CLS) always 0
                full_mask = torch.zeros(
                    (patch_mask.shape[0], patch_mask.shape[1] + 1),
                    device=patch_mask.device,
                    dtype=patch_mask.dtype,
                )
                full_mask[:, 1:] = patch_mask
                x = self.mask_token(x, full_mask)  # shape: [B, N+1, D]
            else:
                x = self.mask_token(x, patch_mask)  # shape: [B, N, D]

        # 5. Apply Energy Transformer blocks
        for block in self.blocks:
            x = block(x)  # shape: [B, N, D] or [B, N+1, D]

        # 6. Apply final normalization
        x = self.norm(x)  # shape: [B, N, D] or [B, N+1, D]

        # 7. Return appropriate output
        if not return_sequence and self.cls_token_flag:
            return x[:, 0]  # shape: [B, D] - CLS token only

        # Verify shape when returning sequence
        if return_sequence and not self.cls_token_flag:
            batch_size, seq_len, embed_dim = x.shape
            assert seq_len == self.patch_embed.num_patches, (
                f"Expected sequence length {self.patch_embed.num_patches}, got {seq_len}"
            )

        return x  # shape: [B, N, D] or [B, N+1, D]


def assemble_encoder(modules: Sequence[nn.Module]) -> nn.Module:
    """Assemble modules into a ViET encoder.

    This function implements the wiring logic for a ViET encoder based on
    a sequence of modules. It identifies component types and arranges them
    according to the expected architecture.

    Args:
        modules: A sequence of PyTorch modules in the expected order:
                - PatchEmbed
                - [Optional] Positional encoding (Learnable2DPosEnc or SinCos2DPosEnc)
                - [Optional] CLSToken
                - [Optional] MaskToken
                - Multiple EnergyTransformer blocks
                - [Optional] Final normalization layer

    Returns
    -------
        A fully wired ViETEncoder instance

    Raises
    ------
        ValueError: If required components are missing or in incorrect order
    """
    from energy_transformer.utils.vision import (
        CLSToken,
        Learnable2DPosEnc,
        MaskToken,
        PatchEmbed,
        SinCos2DPosEnc,
    )

    # Validate we have at least one module
    if not modules:
        raise ValueError("No modules provided to assemble_encoder")

    # Extract components by type
    patch_embed = None
    pos_encoder = None
    cls_token = None
    mask_token = None
    transformer_blocks = []
    norm_layer = None

    for module in modules:
        if isinstance(module, PatchEmbed):
            if patch_embed is not None:
                raise ValueError("Multiple PatchEmbed modules found")
            patch_embed = module
        elif isinstance(module, Learnable2DPosEnc | SinCos2DPosEnc):
            if pos_encoder is not None:
                raise ValueError("Multiple position encoding modules found")
            pos_encoder = module
        elif isinstance(module, CLSToken):
            if cls_token is not None:
                raise ValueError("Multiple CLSToken modules found")
            cls_token = module
        elif isinstance(module, MaskToken):
            if mask_token is not None:
                raise ValueError("Multiple MaskToken modules found")
            mask_token = module
        elif isinstance(module, EnergyTransformer):
            transformer_blocks.append(module)
        elif isinstance(module, nn.LayerNorm):
            if norm_layer is not None:
                raise ValueError("Multiple normalization layers found")
            norm_layer = module

    # Ensure required components are present
    if patch_embed is None:
        raise ValueError("PatchEmbed module is required for ViET encoder")

    if not transformer_blocks:
        raise ValueError("At least one EnergyTransformer block is required")

    # Create default norm layer if none provided
    if norm_layer is None:
        embed_dim = patch_embed.proj.out_channels
        norm_layer = nn.LayerNorm(embed_dim)

    # Create the encoder
    return ViETEncoder(
        patch_embedder=patch_embed,
        transformer_blocks=transformer_blocks,
        norm_layer=norm_layer,
        pos_encoder=pos_encoder,
        cls_token=cls_token,
        mask_token=mask_token,
    )


class ClassificationHead(nn.Module):  # type: ignore
    """Classification head that processes the CLS token."""

    def __init__(self, embed_dim: int, num_classes: int):
        """Initialize classification head.

        Args:
            embed_dim: Dimension of input embeddings
            num_classes: Number of output classes
        """
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Initialize with truncated normal for consistency with other components
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: Tensor) -> Tensor:  # expects CLS token (B,D)
        """Return logits (B, num_classes)."""
        x = self.norm(x)
        return self.classifier(x)  # shape: [B, num_classes]


class MAEDecoder(nn.Module):  # type: ignore
    """Pixel‑space reconstruction for masked‑patch training."""

    def __init__(
        self,
        embed_dim: int,
        patch_size: int | tuple[int, int],
        in_chans: int = 3,
        depth: int = 1,
        hidden: int = 512,
    ):
        """Initialize inpainting decoder.

        Args:
            embed_dim: Dimension of input embeddings
            patch_size: Size of image patches (must match encoder's patch_size)
            in_chans: Number of input channels (must match encoder's in_chans)
            depth: Number of hidden layers
            hidden: Dimension of hidden layers
        """
        super().__init__()

        # Calculate patch dimension (P*P*C)
        patch_h, patch_w = _to_pair(patch_size)
        self.patch_dim = patch_h * patch_w * in_chans

        # Build MLP layers
        layers = []

        # First layer: embed_dim → hidden
        layers.append(nn.Linear(embed_dim, hidden))
        nn.init.trunc_normal_(layers[-1].weight, std=0.02)
        nn.init.zeros_(layers[-1].bias)

        # Add (depth-1) hidden layers
        for _ in range(depth - 1):
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden, hidden))
            nn.init.trunc_normal_(layers[-1].weight, std=0.02)
            nn.init.zeros_(layers[-1].bias)

        # Final activation and output layer
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden, self.patch_dim))
        nn.init.trunc_normal_(layers[-1].weight, std=0.02)
        nn.init.zeros_(layers[-1].bias)

        self.mlp = nn.Sequential(*layers)

    def forward(self, tokens: Tensor) -> Tensor:  # (B,N,D)
        """Return patch‑pixel predictions (B,N,patch_dim)."""
        return self.mlp(tokens)  # shape: [B, N, patch_dim]


class VocabularyHead(nn.Module):  # type: ignore
    """Lightweight AR head for unconditional or class‑conditional image generation."""

    def __init__(self, embed_dim: int, vocab_size: int):
        """Initialize autoregressive generator.

        Args:
            embed_dim: Dimension of input embeddings
            vocab_size: Size of the codebook/vocabulary
        """
        super().__init__()
        self.proj = nn.Linear(embed_dim, vocab_size)

        # Initialize weights
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens: Tensor) -> Tensor:
        """Return logits over codebook/vocab per token: (B, N, vocab_size)."""
        return self.proj(tokens)  # shape: [B, N, vocab_size]


# Register classes in the registry
REALISER_REGISTRY["ViETEncoder"] = ViETEncoder
REALISER_REGISTRY["ClassificationHead"] = ClassificationHead
REALISER_REGISTRY["InpaintingDecoder"] = MAEDecoder
REALISER_REGISTRY["AutoregressiveGenerator"] = VocabularyHead


__all__ = [
    "ViETEncoder",
    "ClassificationHead",
    "MAEDecoder",
    "VocabularyHead",
    "assemble_encoder",
]
