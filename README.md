# Energy Transformer

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![AI-Assisted](https://img.shields.io/badge/AI%20Assisted-Claude%20%2B%20GPT-purple.svg)
![Status: Under Development](https://img.shields.io/badge/status-under%20development-orange.svg)
![GitHub Workflow Status](https://github.com/b-vitamins/energy-transformer/actions/workflows/ci.yml/badge.svg)

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

### Dependencies

Core requirements:
- Python ≥ 3.11
- PyTorch ≥ 2.0
- einops ≥ 0.6
- numpy ≥ 2.2.5
- scipy ≥ 1.10

Optional:
- torchvision, matplotlib (for examples)

## Quick Start

### Basic Usage

```python
import torch
from energy_transformer.models.vision import viet_tiny, viset_tiny

# Energy Transformer
viet = viet_tiny(img_size=224, patch_size=16, num_classes=1000)

# Simplicial Energy Transformer
viset = viset_tiny(img_size=224, patch_size=16, num_classes=1000)

# Forward pass
images = torch.randn(2, 3, 224, 224)
logits = viset(images)

# With energy tracking
result = viset(images, return_energy_info=True)
print(f"Final energy: {result['energy_info']['total_energy']:.3f}")
```

### Declarative Model Construction

```python
from energy_transformer import seq, realise
from energy_transformer.spec import (
    PatchEmbedSpec, CLSTokenSpec, ETSpec, LayerNormSpec
)

# Define model specification
model_spec = seq(
    PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768),
    CLSTokenSpec(),
    ETSpec(steps=4, alpha=0.125),
    LayerNormSpec()
)

# Realize into PyTorch module
model = realise(model_spec)
```

### CIFAR-100 Experiments

```bash
cd examples/cifar

# Quick test
python quick.py --model viset --epochs 10 --subset 0.1

# Full ablation study
python ablation.py

# Visualize results
python visualize.py

# Demo topology-aware simplices
python demo.py
```

See [examples/cifar/README.md](examples/cifar/README.md) for detailed experimentation pipeline.

## Model Variants

### Vision Energy Transformer (ViET)
Standard ET with energy-based components:
```python
from energy_transformer.models.vision import viet_small_cifar
model = viet_small_cifar(num_classes=100)
```

### Vision Simplicial Energy Transformer (ViSET)
ET with topology-aware simplicial Hopfield networks:
```python
from energy_transformer.models.vision import viset_2l_e50_t50_cifar

# 50% edges, 50% triangles from spatial topology
model = viset_2l_e50_t50_cifar(num_classes=100)
```

### Vision Transformer (ViT)
Standard transformer baseline:
```python
from energy_transformer.models.vision import vit_small_cifar
model = vit_small_cifar(num_classes=100)
```

## Key Components

### Energy Transformer Block
```python
from energy_transformer.models.base import EnergyTransformer
from energy_transformer.layers import (
    LayerNorm, MultiHeadEnergyAttention, HopfieldNetwork
)

et_block = EnergyTransformer(
    layer_norm=LayerNorm(embed_dim),
    attention=MultiHeadEnergyAttention(
        in_dim=embed_dim,
        num_heads=12,
        head_dim=64
    ),
    hopfield=HopfieldNetwork(
        in_dim=embed_dim,
        hidden_dim=embed_dim * 4
    ),
    steps=12,
    alpha=0.125
)
```

### Simplicial Hopfield Network
```python
from energy_transformer.layers.simplicial import SimplicialHopfieldNetwork

# Topology-aware with spatial coordinates
coords = [(i, j) for i in range(8) for j in range(8)]  # 8x8 grid
shn = SimplicialHopfieldNetwork(
    in_dim=768,
    coordinates=coords,
    max_dim=2,  # Include up to triangles
    budget=0.15,  # 15% of full edge budget
    temperature=0.5
)
```

## Architecture Details

The Energy Transformer operates through iterative energy minimization:

1. **Layer Normalization**: Transform tokens with energy-based normalization
2. **Energy Computation**: Combine attention and Hopfield energies
3. **Gradient Descent**: Update tokens via energy landscape optimization
4. **Convergence**: Reach low-energy configuration after T steps

### Optimization Modes

- **SGD**: Fixed step size (classic ET)
- **Barzilai-Borwein**: Adaptive step size with line search (recommended)

### Simplicial Extensions

The SET variant enhances ET with spatial topology awareness:
- **k-NN graphs**: Connect spatially nearby patches
- **Delaunay triangulation**: Capture regional structure
- **Higher-order simplices**: Model complex spatial relationships

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

## Contributors

- **Idea**: Suresh Meena and [Ayan Das](https://github.com/b-vitamins)
- **AI Contributors**:
  - Claude Opus 4 (Anthropic)
  - Claude Sonnet 4 (Anthropic)
  - Claude Sonnet 3.7 (Anthropic)
  - ChatGPT o3 (OpenAI)
  - ChatGPT Codex (OpenAI)
- **Human in the Loop**: Ayan

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
