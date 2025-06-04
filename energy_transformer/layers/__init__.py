"""Energy Transformer layers.

This module provides the core layer implementations for Energy Transformer models,
including energy-based alternatives to standard transformer components.

Layer Categories
----------------
- **Attention**: Energy-based multi-head attention mechanism
- **Normalization**: Energy-based layer normalization with learnable temperature
- **Memory**: Hopfield networks (standard and simplicial) for associative memory
- **Embeddings**: Patch and positional embeddings for vision models
- **Tokens**: Special tokens (CLS) for aggregating information
- **Heads**: Task-specific output heads for classification and feature extraction

Example
-------
>>> # Build a simple Energy Transformer block
>>> import torch
>>> from energy_transformer.layers import (
...     LayerNorm, MultiheadEnergyAttention, HopfieldNetwork
... )
>>>
>>> # Create layers
>>> norm = LayerNorm(in_dim=768)
>>> attn = MultiheadEnergyAttention(embed_dim=768, num_heads=12)
>>> hopfield = HopfieldNetwork(in_dim=768, hidden_dim=3072)
>>>
>>> # Use in forward pass
>>> x = torch.randn(4, 100, 768)  # (batch, seq_len, embed_dim)
>>> x_norm = norm(x)
>>> energy = attn(x_norm) + hopfield(x_norm)
"""

from .attention import MultiheadEnergyAttention
from .embeddings import PatchEmbedding, PositionalEmbedding2D
from .heads import ClassificationHead, FeatureHead
from .hopfield import HopfieldNetwork
from .layer_norm import LayerNorm
from .mlp import MLP
from .simplicial import SimplicialHopfieldNetwork
from .tokens import CLSToken

__all__ = [
    "MLP",
    "CLSToken",
    "ClassificationHead",
    "FeatureHead",
    "HopfieldNetwork",
    "LayerNorm",
    "MultiheadEnergyAttention",
    "PatchEmbedding",
    "PositionalEmbedding2D",
    "SimplicialHopfieldNetwork",
]
