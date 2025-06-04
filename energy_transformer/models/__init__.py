"""Energy Transformer models.

This module provides the core Energy Transformer implementation and model registry.

The Energy Transformer replaces standard transformer components with energy-based
alternatives that optimize token representations through gradient descent on an
energy landscape. This approach provides a unified view of attention and memory
mechanisms as energy minimization.

Core Components
---------------
- **EnergyTransformer**: Base class implementing energy-based token refinement
- **REALISER_REGISTRY**: Registry for specification-to-module conversion

Energy Function
---------------
The total energy combines attention and memory components:

    E_total = E_attention + E_hopfield

where tokens are iteratively refined via gradient descent:

    x_{t+1} = x_t - alpha * ∇E_total(x_t)

Example
-------
>>> import torch
>>> from energy_transformer.models import EnergyTransformer
>>> from energy_transformer.layers import (
...     EnergyLayerNorm, MultiheadEnergyAttention, HopfieldNetwork
... )
>>>
>>> # Create an Energy Transformer block
>>> et_block = EnergyTransformer(
...     layer_norm=EnergyLayerNorm(768),
...     attention=MultiheadEnergyAttention(embed_dim=768, num_heads=12),
...     hopfield=HopfieldNetwork(768, hidden_dim=3072),
...     steps=4,
...     alpha=0.125
... )
>>>
>>> # Process tokens through energy optimization
>>> tokens = torch.randn(4, 100, 768)
>>> refined_tokens = et_block(tokens)

See Also
--------
energy_transformer.models.vision : Vision-specific model implementations
energy_transformer.spec : Declarative specification system for model construction
"""

from .base import REALISER_REGISTRY, EnergyTransformer

__all__ = ["REALISER_REGISTRY", "EnergyTransformer"]
