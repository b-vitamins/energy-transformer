# Energy Transformer

![GitHub Workflow Status](https://github.com/b-vitamins/energy-transformer/actions/workflows/python.yml/badge.svg)
[![codecov](https://codecov.io/gh/b-vitamins/energy-transformer/graph/badge.svg?token=6DSB7U0GJF)](https://codecov.io/gh/b-vitamins/energy-transformer)

PyTorch implementation of the **Energy Transformer (ET)**, a neural architecture that replaces traditional transformer components with energy-based alternatives. This implementation includes **Simplicial Energy Transformers (SET)** that extend ET with topology-aware memory through Simplicial Hopfield Networks.

## Overview

The Energy Transformer reformulates attention and feedforward mechanisms as energy optimization problems:
- **Multi-Head Energy Attention**: Attention weights derived from energy function gradients
- **Hopfield Networks**: Associative memory replacing MLPs
- **Energy-based LayerNorm**: Layer normalization with learnable temperature
- **Gradient Descent Optimization**: Token updates via energy landscape traversal

The Simplicial Energy Transformer (SET) extension adds:
- **Simplicial Hopfield Networks**: Topology-aware memory using k-NN graphs and Delaunay triangulation
- **Higher-order interactions**: Edges, triangles, and beyond for spatial structure preservation

## Installation

```bash
# Clone the repository
git clone https://github.com/b-vitamins/energy-transformer.git
cd energy-transformer

# Install with Poetry (recommended)
poetry install

# Or with pip
pip install -e .

# For experiments (includes visualization tools)
poetry install --with examples

# For development
poetry install --with dev
```

## Quick start

```python
# Using pre-built vision models
from energy_transformer.models.vision import viet_base

# Create Vision Energy Transformer
model = viet_base(num_classes=1000)

# Or use the specification system
from energy_transformer import seq, realise
from energy_transformer.spec.library import (
    PatchEmbedSpec, CLSTokenSpec, PosEmbedSpec,
    ETBlockSpec, LayerNormSpec, ClassificationHeadSpec
)

model_spec = seq(
    PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768),
    CLSTokenSpec(),
    PosEmbedSpec(include_cls=True),
    ETBlockSpec(),
    LayerNormSpec(),
    ClassificationHeadSpec(num_classes=1000)
)
model = realise(model_spec)
```

## Usage

### Vision models

```python
import torch
from energy_transformer.models.vision import (
    vit_base,          # Standard Vision Transformer
    viet_base,         # Vision Energy Transformer
    viset_tiny         # Vision Simplicial Energy Transformer
)

# Standard Vision Transformer (baseline)
vit = vit_base(img_size=224, patch_size=16, num_classes=1000)

# Vision Energy Transformer
viet = viet_base(img_size=224, patch_size=16, num_classes=1000)

# Vision Simplicial Energy Transformer
viset = viset_tiny(img_size=224, patch_size=16, num_classes=1000)

# Forward pass
images = torch.randn(4, 3, 224, 224)

# For ViT
logits_vit = vit(images)

# For ViET/ViSET (energy-based models)
logits_viet = viet(images, et_kwargs={"detach": False})  # For training
logits_viset = viset(images, et_kwargs={"detach": True})  # For inference
```

### Building custom models

```python
from energy_transformer.models import EnergyTransformer
from energy_transformer.layers import (
    LayerNorm,
    MultiHeadEnergyAttention,
    HopfieldNetwork
)

# Create Energy Transformer block
et_block = EnergyTransformer(
    layer_norm=LayerNorm(in_dim=768),
    attention=MultiHeadEnergyAttention(
        in_dim=768,
        num_heads=12,
        head_dim=64
    ),
    hopfield=HopfieldNetwork(
        in_dim=768,
        hidden_dim=3072
    ),
    steps=4,
    alpha=0.125
)

# Process tokens
tokens = torch.randn(4, 100, 768)
refined_tokens = et_block(tokens)
```

### Using the specification system

```python
from energy_transformer import seq, loop, realise
from energy_transformer.spec.library import (
    PatchEmbedSpec, CLSTokenSpec, PosEmbedSpec,
    ETBlockSpec, LayerNormSpec, ClassificationHeadSpec,
    MHEASpec, HNSpec
)

# Define architecture declaratively
vit_spec = seq(
    # Patch embedding
    PatchEmbedSpec(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        in_chans=3
    ),
    
    # Add CLS token
    CLSTokenSpec(),
    
    # Add positional embeddings
    PosEmbedSpec(include_cls=True),
    
    # Stack transformer blocks
    loop(
        ETBlockSpec(
            attention=MHEASpec(num_heads=12, head_dim=64),
            hopfield=HNSpec(multiplier=4.0),
            steps=4,
            alpha=0.125
        ),
        times=12
    ),
    
    # Final norm and classification
    LayerNormSpec(),
    ClassificationHeadSpec(num_classes=1000)
)

# Convert to PyTorch module
model = realise(vit_spec)
```

## Examples

See the `examples/` directory for:
- CIFAR-100 experiments comparing topologies
- Ablation studies
- Topology visualization

## Citation

If you use this code, please cite:

```bibtex
@article{hoover2023energy,
  title={Energy Transformer},
  author={Hoover, Benjamin and Liang, Yuchen and Pham, Bao and 
          Panda, Rameswar and Strobelt, Hendrik and Chau, Duen Horng and 
          Zaki, Mohammed J and Krotov, Dmitry},
  journal={arXiv preprint arXiv:2302.07253},
  year={2023}
}

@article{burns2023simplicial,
  title={Simplicial Hopfield networks},
  author={Burns, Thomas and Fukai, Tomoki},
  journal={arXiv preprint arXiv:2305.05179},
  year={2023}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
