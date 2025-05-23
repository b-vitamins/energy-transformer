"""Energy Transformer model specifications and realisation.

This module provides a principled approach to defining Energy Transformer
model architectures through immutable, validated specifications.

Basic usage:
    from energy_transformer import realise
    from energy_transformer.spec import seq, PatchSpec, PosEncSpec, CLSTokenSpec, ETBlockSpec

    # Define a model specification
    vit_spec = seq(
        PatchSpec(img_size=224, patch_size=16, in_chans=3, embed_dim=768),
        PosEncSpec(kind="sincos"),
        CLSTokenSpec(),
        *[ETBlockSpec(steps=4, alpha=0.125) for _ in range(12)],
        NormSpec()
    )

    # Create an actual model
    model = realise(vit_spec)
"""

from energy_transformer.spec.combinators import (
    Parallel,
    ParallelSpec,
    Repeat,
    # Also expose capitalized aliases for more intuitive API
    Seq,
    SequentialSpec,
    parallel,
    repeat,
    seq,
)
from energy_transformer.spec.primitives import (
    CLSTokenSpec,
    ETBlockSpec,
    MaskTokenSpec,
    NormSpec,
    PatchSpec,
    PosEncSpec,
    Spec,
)
from energy_transformer.spec.realise import (
    SpecInfo,
    realise,
    register_module,
    register_realiser,
)


# Convenience factory functions for common models
def vit_tiny(img_size=224, patch_size=16):
    """Create a tiny Vision Transformer spec."""
    return seq(
        PatchSpec(
            img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=192
        ),
        PosEncSpec("learned"),
        CLSTokenSpec(),
        repeat(ETBlockSpec(steps=4, alpha=0.125), 12),
        NormSpec(),
    )


def vit_small(img_size=224, patch_size=16):
    """Create a small Vision Transformer spec."""
    return seq(
        PatchSpec(
            img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=384
        ),
        PosEncSpec("learned"),
        CLSTokenSpec(),
        repeat(ETBlockSpec(steps=4, alpha=0.125), 12),
        NormSpec(),
    )


def vit_base(img_size=224, patch_size=16):
    """Create a base Vision Transformer spec."""
    return seq(
        PatchSpec(
            img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=768
        ),
        PosEncSpec("learned"),
        CLSTokenSpec(),
        repeat(ETBlockSpec(steps=4, alpha=0.125), 12),
        NormSpec(),
    )


def vit_large(img_size=224, patch_size=16):
    """Create a large Vision Transformer spec."""
    return seq(
        PatchSpec(
            img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=1024
        ),
        PosEncSpec("sincos"),
        CLSTokenSpec(),
        repeat(ETBlockSpec(steps=4, alpha=0.125), 24),
        NormSpec(),
    )


def mae_base(img_size=224, patch_size=16):
    """Create a base Masked Autoencoder spec."""
    return seq(
        PatchSpec(
            img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=768
        ),
        PosEncSpec("learned"),
        MaskTokenSpec(),
        repeat(ETBlockSpec(steps=4, alpha=0.125), 12),
        NormSpec(),
    )


# Expose all public symbols
__all__ = [
    # Primitive specs
    "Spec",
    "PatchSpec",
    "PosEncSpec",
    "CLSTokenSpec",
    "MaskTokenSpec",
    "ETBlockSpec",
    "NormSpec",
    # Combinators
    "SequentialSpec",
    "seq",
    "repeat",
    "ParallelSpec",
    "parallel",
    # Capitalized aliases
    "Seq",
    "Repeat",
    "Parallel",
    # Realisation
    "SpecInfo",
    "realise",
    "register_realiser",
    "register_module",
    # Factory functions
    "vit_tiny",
    "vit_small",
    "vit_base",
    "vit_large",
    "mae_base",
]
