# Energy Transformer

![GitHub Workflow Status](https://github.com/b-vitamins/energy-transformer/actions/workflows/python.yml/badge.svg)
[![codecov](https://codecov.io/gh/b-vitamins/energy-transformer/graph/badge.svg?token=6DSB7U0GJF)](https://codecov.io/gh/b-vitamins/energy-transformer)

PyTorch implementation of the **Energy Transformer (ET)** and the
**Simplicial Energy Transformer (SET)**. The project also provides a
specification system for declaratively building models and a plugin-based
realiser that converts specifications into PyTorch modules.

## Key Features

- **Energy-based Components** – attention, layer norm and memory are all
  formulated as energy minimisation problems.
- **Simplicial Hopfield Networks** – topology-aware memory that
  preserves higher-order structure.
- **Specification System** – declaratively define models using composable
  specs and realise them into modules.

## Installation

```bash
# Clone the repository
git clone https://github.com/b-vitamins/energy-transformer.git
cd energy-transformer

# Install with Poetry (recommended)
poetry install

# Or with pip
pip install -e .
```

Optional extras are defined in `pyproject.toml`:

```bash
# Install example dependencies
poetry install --with examples

# Install development tools
poetry install --with dev
```

## Quick Start

The packaged builders create fully configured models. After installation
run the following script:

```python
import torch
from energy_transformer.models.vision import viet_base

model = viet_base(img_size=64, patch_size=16, num_classes=10)
images = torch.randn(2, 3, 64, 64)
logits = model(images, et_kwargs={"detach": False})
print(logits.shape)
```

## Building with Specifications

The spec system allows declarative model construction:

```python
from energy_transformer import seq, realise
from energy_transformer.spec.library import (
    PatchEmbedSpec, CLSTokenSpec, PosEmbedSpec,
    ETBlockSpec, LayerNormSpec, ClassificationHeadSpec,
)

spec = seq(
    PatchEmbedSpec(img_size=32, patch_size=4, embed_dim=64),
    CLSTokenSpec(),
    PosEmbedSpec(include_cls=True),
    ETBlockSpec(),
    LayerNormSpec(),
    ClassificationHeadSpec(num_classes=10),
)
model = realise(spec)
```

The resulting module can be used like any other PyTorch model.

### Metrics and Debugging

```python
from energy_transformer.spec import configure_realisation, get_realisation_metrics
from energy_transformer.spec.debug import DebugTracer

tracer = DebugTracer()
configure_realisation(enable_metrics=True, debug_tracer=tracer)

# Realise a spec as above
model = realise(spec)
print(get_realisation_metrics())
tracer.print_summary()
```

## Examples

See the `examples/` directory for CIFAR‑100 experiments, topology
visualisation and ablation studies.

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

This project is licensed under the Apache License 2.0 – see the
[LICENSE](LICENSE) file for details.
