#!/usr/bin/env python3
"""Load and use trained Energy Transformer models for inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torchvision import transforms

from energy_transformer.models.vision import (
    viet_2l_cifar,
    viset_2l_cifar,
    vit_small_cifar,
)

CIFAR100_CLASSES = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]

MODELS = {
    "vit": lambda: vit_small_cifar(num_classes=100, depth=6),
    "viet": lambda: viet_2l_cifar(num_classes=100),
    "viset": lambda: viset_2l_cifar(num_classes=100),
}


def get_data_dir() -> Path:
    """Return XDG data directory for energy-transformer."""
    import os

    xdg_data_home = os.environ.get(
        "XDG_DATA_HOME", str(Path("~/.local/share").expanduser())
    )
    return Path(xdg_data_home) / "energy-transformer"


def load_model(model_name: str, device: str = "cpu") -> torch.nn.Module:
    """Load a trained model from checkpoint."""
    model = MODELS[model_name]()
    checkpoint_path = get_data_dir() / "models" / model_name / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found. Train {model_name} first."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(
        f"Loaded {model_name.upper()} from epoch {checkpoint['epoch']} "
        f"with {checkpoint['best_acc']:.2f}% validation accuracy"
    )
    return model


def preprocess_image(image_path: str) -> torch.Tensor:
    """Preprocess an image for CIFAR-100 models."""
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            ),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    _model_name: str,
    top_k: int = 5,
) -> list[dict[str, float | int | str]]:
    """Make predictions with a model."""
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
    top_probs, top_indices = torch.topk(probabilities[0], top_k)
    predictions = []
    for prob, idx in zip(top_probs, top_indices, strict=False):
        predictions.append(
            {
                "class": CIFAR100_CLASSES[idx],
                "index": idx.item(),
                "probability": prob.item(),
            }
        )
    return predictions


def analyze_energy_dynamics(
    model: torch.nn.Module, image_tensor: torch.Tensor
) -> list[dict[str, float]]:
    """Analyze energy for Energy Transformer models."""
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        if hasattr(model, "et_blocks"):
            _, (e_att, e_hop) = model(image_tensor, return_energies=True)
            return [
                {
                    "attention": e_att.item(),
                    "hopfield": e_hop.item(),
                    "total": e_att.item() + e_hop.item(),
                }
            ]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference with trained models"
    )
    parser.add_argument(
        "model",
        choices=["vit", "viet", "viset"],
        help="Model to use for inference",
    )
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top predictions to show"
    )
    parser.add_argument(
        "--analyze-energy",
        action="store_true",
        help="Analyze energy dynamics (ET models only)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference",
    )
    args = parser.parse_args()

    print(f"Loading {args.model.upper()} model...")
    model = load_model(args.model, args.device)

    print(f"\nProcessing image: {args.image}")
    image_tensor = preprocess_image(args.image)

    print(f"\nTop {args.top_k} predictions:")
    predictions = predict(model, image_tensor, args.model, args.top_k)
    for i, pred in enumerate(predictions, 1):
        print(
            f"{i}. {pred['class']:20s} (class {pred['index']:3d}): {pred['probability']:6.2%}"
        )

    if args.analyze_energy and args.model in ["viet", "viset"]:
        print(f"\n{args.model.upper()} Energy Dynamics:")
        energies = analyze_energy_dynamics(model, image_tensor)
        total_reduction = sum(e["reduction"] for e in energies)
        print(
            f"Total energy reduction across {len(energies)} blocks: {total_reduction:.2f}"
        )
        for e in energies:
            print(
                f"  Block {e['block']}: {e['initial_energy']:.2f} → {e['final_energy']:.2f} "
                f"(reduction: {e['reduction']:.2f}, steps: {e['steps']})"
            )
    elif args.analyze_energy:
        print(
            f"\nEnergy analysis not available for {args.model.upper()} (not an ET model)"
        )


if __name__ == "__main__":
    main()
