# CIFAR-100 Experiments

Minimal scripts for testing ViSET on CIFAR-100.

## Scripts

- `ablation.py` - Run full ablation study (6 models, ~10 hours)
- `visualize.py` - Generate plots from results
- `quick.py` - Quick testing (10% data, few epochs)
- `demo.py` - Visualize topology-aware simplices

## Quick Start

```bash
# Quick test (1 minute)
python quick.py --model viset --epochs 5 --subset 0.05

# Memory profiling
python quick.py --memory --model viset

# Visualize topology
python demo.py

# Full ablation (10+ hours)
python ablation.py

# Plot results
python visualize.py
```

## Results Location

All results are saved to `~/.local/share/energy-transformer/experiments/`

## Models

- `vit` - Standard Vision Transformer (2 layers)
- `viet` - Vision Energy Transformer (2 layers)
- `viset` - Vision Simplicial Energy Transformer (topology-aware)
- `viset-random` - ViSET with random simplices (baseline)