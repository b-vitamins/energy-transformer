# CIFAR-100 Energy Transformer Examples

Clean, instructive examples comparing Vision Transformer (ViT), Vision Energy Transformer (ViET), and Vision Simplicial Energy Transformer (ViSET) on CIFAR-100.

## Quick Start

```bash
# Train models (200 epochs each)
python train.py vit    # Baseline Vision Transformer
python train.py viet   # Energy Transformer
python train.py viset  # Simplicial Energy Transformer

# Visualize training results
python visualize.py vit -o vit_training.png
python visualize.py viet -o viet_training.png
python visualize.py viset -o viset_training.png

# Run inference on an image
python infer.py viet path/to/image.jpg --analyze-energy
```

## Scripts

### `train.py`
Trains models with optimized hyperparameters:
- **Fixed settings**: 200 epochs, AdamW optimizer, cosine LR schedule
- **Data augmentation**: CutMix (after warmup), random crops, flips
- **Energy monitoring**: Tracks attention and Hopfield energies (ET models)
- **Automatic saving**: Best model and training history to XDG data directory

### `visualize.py`
Creates publication-quality training plots showing:
- Training/validation loss and accuracy curves
- Energy component evolution (ET models only)
- Learning rate schedule and gradient norms
- Summary statistics table

### `infer.py`
Demonstrates model usage:
- Load trained models from checkpoints
- Make predictions on new images
- Analyze energy dynamics across layers (ET models)

## Model Architectures

All models use small variants for fast experimentation:

| Model | Parameters | Key Features |
|-------|------------|--------------|
| ViT   | ~1.2M      | 6-layer transformer baseline |
| ViET  | ~300K      | Energy-based attention & Hopfield memory, 2 layers |
| ViSET | ~350K      | Higher-order (3-way) interactions, 2 layers |

## Expected Results

After 200 epochs on CIFAR-100:
- **ViT**: ~55-58% validation accuracy
- **ViET**: ~60-63% validation accuracy
- **ViSET**: ~61-64% validation accuracy

Energy models typically show:
- Better gradient flow through layers
- More stable training dynamics
- Improved feature representations

## File Locations

All data is saved to:
```
~/.local/share/energy-transformer/
└── models/
    ├── vit/
    │   ├── best_model.pth
    │   └── history.json
    ├── viet/
    │   └── ...
    └── viset/
        └── ...
```

