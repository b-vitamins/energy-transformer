"""Ablation study for ViSET on CIFAR-100."""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from energy_transformer.models.vision import (
    viet_2l_cifar,
    viset_2l_e50_t50_cifar,
    viset_2l_e100_cifar,
    viset_2l_random_cifar,
    viset_2l_t100_cifar,
    vit_tiny_cifar,
)

# Use XDG base directory or fallback
DATA_HOME = Path.home() / ".local" / "share" / "energy-transformer"
DATA_HOME.mkdir(parents=True, exist_ok=True)


class TrainingResult(TypedDict):
    """Type definition for training results."""

    name: str
    params: int
    best_val: float
    best_test: float
    time: float


@dataclass
class Config:
    """Training configuration."""

    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    seed: int = 42


def setup_data(
    config: Config,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """Minimal data setup."""
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            ),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            ),
        ]
    )

    train_dataset = datasets.CIFAR100(
        DATA_HOME, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR100(
        DATA_HOME, train=False, transform=transform_test
    )

    # 90/10 train/val split
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )

    val_dataset = datasets.CIFAR100(
        DATA_HOME, train=True, download=False, transform=transform_test
    )

    val_subset_with_test_transform = Subset(val_dataset, val_subset.indices)

    train_loader = DataLoader(
        train_subset,
        config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset_with_test_transform,
        config.batch_size * 2,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, config.batch_size * 2, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def train_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    is_et: bool = False,
) -> tuple[float, float]:
    """Train for one epoch with live batch stats."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        device_type, enabled = device.type, scaler is not None
        with torch.amp.autocast(device_type=device_type, enabled=enabled):
            output = (
                model(data, et_kwargs={"detach": False})
                if is_et
                else model(data)
            )
            loss = criterion(output, target)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update running statistics
        total_loss += loss.item() * data.size(0)
        correct += output.argmax(1).eq(target).sum().item()
        total += target.size(0)

        # Calculate current stats
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time

        # Print live stats every batch
        print("\r\033[K", end="")  # Clear entire line
        print(
            f"Epoch {epoch:3d}/{total_epochs:3d} | "
            f"Batch {batch_idx + 1:3d}/{len(loader):3d} | "
            f"CE Loss: {avg_loss:.4f} | "
            f"Acc: {accuracy:6.2f}% | "
            f"LR: {current_lr:.2e} | "
            f"t: {elapsed:3.0f}s",
            end="",
            flush=True,
        )

    # Print final newline after epoch completes
    print()
    return total_loss / total, 100.0 * correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    is_et: bool = False,
) -> tuple[float, float]:
    """Evaluate model."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = (
                model(data, et_kwargs={"detach": True})
                if is_et
                else model(data)
            )
            total_loss += criterion(output, target).item() * data.size(0)
            correct += output.argmax(1).eq(target).sum().item()
            total += target.size(0)

    return total_loss / total, 100.0 * correct / total


def train_model(
    name: str,
    model_fn: Callable[[], nn.Module],
    config: Config,
    is_et: bool = False,
) -> TrainingResult:
    """Train a single model."""
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup
    train_loader, val_loader, test_loader = setup_data(config)
    model = model_fn().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Training {name}")
    print(f"Parameters: {params:,} ({params / 1e6:.2f}M)")
    print("-" * 90)

    optimizer = optim.AdamW(
        model.parameters(),
        config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    if device.type == "cuda":
        scaler = torch.amp.GradScaler(device.type)
    else:
        scaler = None

    # Cosine schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)

    best_val, best_test = 0.0, 0.0
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            epoch,
            config.epochs,
            is_et,
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, is_et
        )
        _, test_acc = evaluate(model, test_loader, criterion, device, is_et)

        scheduler.step()

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc

    total_time = time.time() - start_time
    print("-" * 90)
    print(
        f"Training completed! Best validation: {best_val:.2f}%, "
        f"Best test: {best_test:.2f}%"
    )

    return TrainingResult(
        name=name,
        params=params,
        best_val=best_val,
        best_test=best_test,
        time=total_time,
    )


def main() -> None:
    """Run ablation experiments."""
    config = Config(epochs=100)

    experiments = [
        (
            "ViSET-E50-T50",
            lambda: viset_2l_e50_t50_cifar(num_classes=100),
            True,
        ),
        ("ViSET-Random", lambda: viset_2l_random_cifar(num_classes=100), True),
        ("ViSET-E100", lambda: viset_2l_e100_cifar(num_classes=100), True),
        ("ViSET-T100", lambda: viset_2l_t100_cifar(num_classes=100), True),
        ("ViET-2L", lambda: viet_2l_cifar(num_classes=100), True),
        ("ViT", lambda: vit_tiny_cifar(num_classes=100, depth=2), False),
    ]

    results = []
    for name, model_fn, is_et in experiments:
        result = train_model(name, model_fn, config, is_et)
        results.append(result)
        torch.cuda.empty_cache()

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = DATA_HOME / "experiments" / f"ablation_{timestamp}"
    results_dir.mkdir(parents=True)

    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 90)
    print(f"{'Model':<20} {'Params':>10} {'Val Acc':>10} {'Test Acc':>10}")
    print("-" * 50)
    for r in sorted(results, key=lambda x: x["best_val"], reverse=True):
        print(
            f"{r['name']:<20} {r['params']:>10,} "
            f"{r['best_val']:>9.2f}% {r['best_test']:>9.2f}%"
        )

    # Key comparisons
    viset = next((r for r in results if r["name"] == "ViSET-E50-T50"), None)
    viset_rand = next((r for r in results if r["name"] == "ViSET-Random"), None)
    vit = next((r for r in results if r["name"] == "ViT"), None)

    if viset and viset_rand:
        print("\nViSET-E50-T50 vs ViSET-Random:")
        print(
            f"  Validation accuracy difference: "
            f"{viset['best_val'] - viset_rand['best_val']:+.2f}%"
        )
        print(
            f"  Test accuracy difference: "
            f"{viset['best_test'] - viset_rand['best_test']:+.2f}%"
        )

    if viset and vit:
        param_diff = viset["params"] - vit["params"]
        print(f"  Parameter difference: {param_diff:+,}")


if __name__ == "__main__":
    main()
