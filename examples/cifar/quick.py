#!/usr/bin/env python3
"""Quick testing script."""

import argparse
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from energy_transformer.models.vision.viet import viet_2l_cifar
from energy_transformer.models.vision.viset import (
    viset_2l_e50_t50_cifar,
    viset_2l_random_cifar,
)
from energy_transformer.models.vision.vit import vit_tiny_cifar


def quick_test(
    model_name: str = "viset",
    epochs: int = 10,
    subset: float = 0.1,
) -> None:
    """Quick model test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model selection
    models: dict[str, tuple[nn.Module, bool]] = {
        "vit": (vit_tiny_cifar(num_classes=100, depth=2), False),
        "viet": (viet_2l_cifar(num_classes=100), True),
        "viset": (viset_2l_e50_t50_cifar(num_classes=100), True),
        "viset-random": (viset_2l_random_cifar(num_classes=100), True),
    }

    if model_name not in models:
        print(f"Unknown model: {model_name}")
        return

    model, is_et = models[model_name]
    model = model.to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}")
    print(f"Parameters: {params:,}")
    print(f"Device: {device}")
    print("-" * 50)

    # Data
    tx = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761),
            ),
        ],
    )

    train_dataset = datasets.CIFAR100(
        "./data",
        train=True,
        download=True,
        transform=tx,
    )
    test_dataset = datasets.CIFAR100("./data", train=False, transform=tx)

    # Use subset
    if subset < 1.0:
        train_size = int(len(train_dataset) * subset)
        test_size = int(len(test_dataset) * subset)
        train_dataset = Subset(train_dataset, range(train_size))
        test_dataset = Subset(test_dataset, range(test_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
    )
    test_loader = DataLoader(test_dataset, batch_size=256, num_workers=2)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Training
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss, train_correct = 0, 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = (
                model(data, et_kwargs={"detach": False})
                if is_et
                else model(data)
            )
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            train_correct += output.argmax(1).eq(target).sum().item()

        # Test
        model.eval()
        test_correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = (
                    model(data, et_kwargs={"detach": True})
                    if is_et
                    else model(data)
                )
                test_correct += output.argmax(1).eq(target).sum().item()

        train_acc = 100.0 * train_correct / len(train_dataset)
        test_acc = 100.0 * test_correct / len(test_dataset)

        print(
            f"Epoch {epoch:2d}/{epochs} | "
            f"Train: {train_acc:5.1f}% | Test: {test_acc:5.1f}%",
        )

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({elapsed / epochs:.1f}s per epoch)")


def memory_test(model_name: str = "viset") -> None:
    """Test memory usage."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Create model
    models: dict[str, tuple[nn.Module, bool]] = {
        "vit": (vit_tiny_cifar(num_classes=100, depth=2), False),
        "viet": (viet_2l_cifar(num_classes=100), True),
        "viset": (viset_2l_e50_t50_cifar(num_classes=100), True),
    }

    model, is_et = models[model_name]
    model = model.to(device)

    # Test forward
    start_mem = torch.cuda.memory_allocated() / 1024**2

    dummy = torch.randn(64, 3, 32, 32).to(device)
    with torch.no_grad():
        _ = model(dummy, et_kwargs={"detach": True}) if is_et else model(dummy)

    peak_forward = torch.cuda.max_memory_allocated() / 1024**2

    # Test backward
    optim.Adam(model.parameters())
    output = (
        model(dummy, et_kwargs={"detach": False}) if is_et else model(dummy)
    )
    loss = output.mean()
    loss.backward()

    peak_total = torch.cuda.max_memory_allocated() / 1024**2

    print(f"Memory usage for {model_name}:")
    print(f"  Model: {start_mem:.1f} MB")
    print(f"  Forward: {peak_forward:.1f} MB")
    print(f"  Total: {peak_total:.1f} MB")


def main() -> None:
    """Handle command line arguments and run tests."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="viset",
        choices=["vit", "viet", "viset", "viset-random"],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--subset",
        type=float,
        default=0.1,
        help="Fraction of data to use",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Test memory usage",
    )

    args = parser.parse_args()

    if args.memory:
        memory_test(args.model)
    else:
        quick_test(args.model, args.epochs, args.subset)


if __name__ == "__main__":
    main()
