"""Energy Transformer models.

This package provides implementations of the Energy Transformer architecture,
a gradient-based approach to attention and self-attention.

Basic usage with specification API:
    from energy_transformer import realise, Seq, PatchSpec, ETblockSpec

    # Define model architecture as a spec
    spec = Seq(
        PatchSpec(img_size=224, patch_size=16, in_chans=3, embed_dim=768),
        ETblockSpec(steps=4, alpha=0.125)
    )

    # Create the actual model
    model = realise(spec)

    # Use for inference or training
    output = model(input_tensor)
"""

# Core models
from .models import (
    ClassificationHead,
    EnergyTransformer,
    MAEDecoder,
    ViETEncoder,
    VocabularyHead,
)

# Specification API
# Factory functions
from .spec import (
    CLSTokenSpec,
    ETBlockSpec,
    MaskTokenSpec,
    NormSpec,
    Parallel,
    PatchSpec,
    PosEncSpec,
    Repeat,
    # Also expose capitalized aliases
    Seq,
    mae_base,
    parallel,
    realise,
    repeat,
    seq,
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
)

__all__ = [
    # Models
    "EnergyTransformer",
    "ViETEncoder",
    "MAEDecoder",
    "ClassificationHead",
    "VocabularyHead",
    # Spec API and realisation
    "realise",
    "seq",
    "repeat",
    "parallel",
    # Capitalized aliases
    "Seq",
    "Repeat",
    "Parallel",
    # Primitives
    "PatchSpec",
    "PosEncSpec",
    "CLSTokenSpec",
    "MaskTokenSpec",
    "ETBlockSpec",
    "NormSpec",
    # Factory functions
    "vit_tiny",
    "vit_small",
    "vit_base",
    "vit_large",
    "mae_base",
]
