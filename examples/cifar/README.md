# CIFAR-100 Experiments

Minimal scripts for quick CIFAR-100 experiments.

## Scripts

- `quick.py` - Quick testing (10% data, few epochs)

## Quick Start

```bash
# Quick test (1 minute)
python quick.py --model viet --epochs 5 --subset 0.05

# Memory profiling
python quick.py --memory --model viet

# Full training example
python quick.py --model viet --epochs 10
```

## Results Location

All results are saved to `~/.local/share/energy-transformer/experiments/`

## Models

- `vit` - Standard Vision Transformer (2 layers)
- `viet` - Vision Energy Transformer (2 layers)
