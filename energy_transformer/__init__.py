"""Energy Transformer: Energy-based transformers with associative memory.

This package implements the Energy Transformer architecture, which replaces
standard transformer components with energy-based alternatives. The key insight
is viewing attention and feed-forward networks as energy functions that can be
optimized through gradient descent.

Quick Start
-----------
>>> # Using the specification system (recommended)
>>> from energy_transformer import seq, realise
>>> from energy_transformer.spec.library import *
>>>
>>> model_spec = seq(
...     PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768),
...     CLSTokenSpec(),
...     loop(ETBlockSpec(), times=12),
...     ClassificationHeadSpec(num_classes=1000)
... )
>>> model = realise(model_spec)

>>> # Or use pre-built vision models
>>> from energy_transformer.models.vision import viet_base
>>> model = viet_base(num_classes=1000)

Key Features
------------
- **Energy-based attention**: Multi-head attention as energy minimization
- **Hopfield networks**: Associative memory replacing feed-forward networks
- **Simplicial complexes**: Topology-aware memory with higher-order interactions
- **Declarative specification**: Build models using composable specifications

Architecture Components
-----------------------
1. **Layer Normalization**: Energy-based normalization with learnable temperature
2. **Energy Attention**: Attention weights derived from energy landscape
3. **Hopfield Networks**: Modern continuous Hopfield networks for memory
4. **Iterative Refinement**: Token optimization through gradient descent

References
----------
.. [1] Hoover et al. "Energy Transformer" arXiv:2302.07253 (2023)
.. [2] Ramsauer et al. "Hopfield Networks is All You Need" ICLR (2021)
.. [3] Burns & Fukai "Simplicial Hopfield Networks" arXiv:2305.05179 (2023)
"""

# Core models
from .models import EnergyTransformer

# Specification API - Core functionality
from .spec import (
    Context,
    RealisationError,
    # Base types for type hints
    Spec,
    # Common errors
    ValidationError,
    cond,
    configure_realisation,
    loop,
    parallel,
    # Main API functions
    realise,
    register,
    # Composition functions
    seq,
    # Advanced features
    visualize,
)

__all__ = [
    # Core model
    "EnergyTransformer",
    # Spec system - Main API
    "realise",
    "register",
    # Spec system - Composition
    "seq",
    "loop",
    "parallel",
    "cond",
    # Spec system - Base types
    "Spec",
    "Context",
    # Spec system - Errors
    "ValidationError",
    "RealisationError",
    # Spec system - Utilities
    "visualize",
    "configure_realisation",
]

__version__ = "0.1.0"
__author__ = "Ayan Das <bvits@riseup.net>"
__license__ = "Apache-2.0"
