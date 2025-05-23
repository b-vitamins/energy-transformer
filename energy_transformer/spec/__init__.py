"""Energy Transformer model specifications and realisation.

This module provides utilities to defining Energy Transformer
model architectures through immutable, validated specifications.
"""

from energy_transformer.spec.combinators import (
    Parallel,
    ParallelSpec,
    Repeat,
    Seq,
    SequentialSpec,
    parallel,
    repeat,
    seq,
)
from energy_transformer.spec.primitives import (
    CLSTokenSpec,
    EmbeddingDim,
    ETSpec,
    HNSpec,
    ImageSize,
    LayerNormSpec,
    MHEASpec,
    PatchEmbedSpec,
    PatchSize,
    PosEmbedSpec,
    Spec,
    TokenCount,
    ValidationError,
    to_pair,
    validate_positive,
    validate_probability,
)
from energy_transformer.spec.realise import (
    RealisationError,
    Realise,
    SpecInfo,
    realise,
    register_realiser,
)

__all__ = [
    "EmbeddingDim",
    "TokenCount",
    "ImageSize",
    "PatchSize",
    "Spec",
    "ValidationError",
    "RealisationError",
    "CLSTokenSpec",
    "ETSpec",
    "HNSpec",
    "LayerNormSpec",
    "MHEASpec",
    "PatchEmbedSpec",
    "PosEmbedSpec",
    "SequentialSpec",
    "ParallelSpec",
    "seq",
    "repeat",
    "parallel",
    "Seq",
    "Repeat",
    "Parallel",
    "SpecInfo",
    "realise",
    "register_realiser",
    "Realise",
    "validate_positive",
    "validate_probability",
    "to_pair",
    "vit_tiny",
    "vit_small",
    "vit_base",
    "vit_large",
    "custom_et_model",
]
