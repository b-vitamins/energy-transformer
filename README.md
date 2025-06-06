# Energy Transformer

![GitHub Workflow Status](https://github.com/b-vitamins/energy-transformer/actions/workflows/python.yml/badge.svg)
[![codecov](https://codecov.io/gh/b-vitamins/energy-transformer/graph/badge.svg?token=6DSB7U0GJF)](https://codecov.io/gh/b-vitamins/energy-transformer)

PyTorch implementation of the **Energy Transformer (ET)**.

## Key Features

- **Energy-based Components** – attention, layer norm and memory are all
  formulated as energy minimisation problems.

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

```

## License

This project is licensed under the Apache License 2.0 – see the
[LICENSE](LICENSE) file for details.
