#!/usr/bin/env python3
"""Train Energy Transformer models on CIFAR-100 with fixed, optimal settings."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from energy_transformer.models.vision import (
    viet_2l_cifar,
    viset_2l_cifar,
    vit_small_cifar,
)

# Fixed hyperparameters (optimized for CIFAR-100)
BATCH_SIZE = 128
EPOCHS = 200
LR_MIN = 1e-5
LR_MAX = 1e-3
WARMUP_EPOCHS = 10
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
LABEL_SMOOTHING = 0.1
CUTMIX_PROB = 0.5
CUTMIX_ALPHA = 1.0
LAM_THRESHOLD = 0.5

# Model factories
MODELS: dict[str, Callable[[], nn.Module]] = {
    "vit": lambda: vit_small_cifar(num_classes=100, depth=6),
    "viet": lambda: viet_2l_cifar(num_classes=100),
    "viset": lambda: viset_2l_cifar(num_classes=100),
}


def get_data_dir() -> Path:
    """Return XDG data directory for energy-transformer."""
    xdg_data_home = os.environ.get(
        "XDG_DATA_HOME", str(Path("~/.local/share").expanduser())
    )
    return Path(xdg_data_home) / "energy-transformer"


def get_lr(epoch: int, step: int, steps_per_epoch: int) -> float:
    """Warmup + cosine decay schedule."""
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    current_step = epoch * steps_per_epoch + step
    total_steps = EPOCHS * steps_per_epoch
    if current_step < warmup_steps:
        return LR_MIN + (LR_MAX - LR_MIN) * current_step / warmup_steps
    progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + np.cos(np.pi * progress))


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute L2 norm of gradients."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm**0.5


def cutmix_data(
    x: torch.Tensor, y: torch.Tensor, alpha: float = CUTMIX_ALPHA
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply CutMix augmentation."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Generate box
    w, h = x.size(2), x.size(3)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

    return mixed_x, y, y[index], float(lam)


def train(model_name: str) -> Path:  # noqa: C901, PLR0912, PLR0915
    """Train a model with fixed optimal settings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_name.upper()} on {device}")

    data_dir = get_data_dir()
    save_dir = data_dir / "models" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

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

    cache_dir = Path.home() / ".cache" / "torchvision"
    train_dataset = datasets.CIFAR100(
        cache_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR100(
        cache_dir, train=False, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = MODELS[model_name]().to(device)
    is_energy_model = model_name in {"viet", "viset"}
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    optimizer = optim.AdamW(
        model.parameters(), lr=LR_MIN, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    history: list[dict[str, float]] = []
    steps_per_epoch = len(train_loader)

    if is_energy_model:
        from energy_transformer.utils import EnergyTracker

        tracker = EnergyTracker()
        if hasattr(model, "et_blocks") and len(model.et_blocks) > 0:

            def track_energy(m: nn.Module, info: Any) -> None:  # type: ignore[override]
                if info.iteration == m.steps - 1:
                    tracker.update(info)

            handle = model.et_blocks[0].register_step_hook(track_energy)

    print("-" * 120)
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_e_att = 0.0
        running_e_hop = 0.0
        running_grad_norm = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)  # noqa: PLW2901

            if epoch >= WARMUP_EPOCHS and np.random.rand() < CUTMIX_PROB:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets)  # noqa: PLW2901
                mixed = True
            else:
                targets_a = targets_b = targets
                lam = 1.0
                mixed = False

            lr = get_lr(epoch, batch_idx, steps_per_epoch)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if is_energy_model:
                outputs = model(inputs, et_kwargs={"detach": False})
            else:
                outputs = model(inputs)

            if mixed:
                loss = lam * criterion(outputs, targets_a) + (
                    1 - lam
                ) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = compute_gradient_norm(model)
            running_grad_norm = (
                grad_norm
                if batch_idx == 0
                else 0.9 * running_grad_norm + 0.1 * grad_norm
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            running_loss = (
                loss.item()
                if batch_idx == 0
                else 0.9 * running_loss + 0.1 * loss.item()
            )

            if is_energy_model and batch_idx % 10 == 0:
                stats = tracker.get_batch_statistics()
                if stats:
                    running_e_att = stats.get("attention_mean", 0).item()
                    running_e_hop = stats.get("hopfield_mean", 0).item()
                tracker.history.clear()

            _, predicted = outputs.max(1)
            if lam > LAM_THRESHOLD:
                total += targets_a.size(0)
                correct += predicted.eq(targets_a).sum().item()
            else:
                total += targets_b.size(0)
                correct += predicted.eq(targets_b).sum().item()

            if batch_idx % 10 == 0:
                elapsed = time.time() - epoch_start
                acc = 100.0 * correct / total
                sys.stdout.write(
                    f"\rEpoch {epoch + 1:3d}/{EPOCHS} | Batch {batch_idx + 1:3d}/{steps_per_epoch} | "
                    f"CE: {running_loss:.4f} | Acc: {acc:5.1f}% | E(A): {running_e_att:7.1f} | "
                    f"E(H): {running_e_hop:7.1f} | |∇|: {running_grad_norm:5.1f} | LR: {lr:.0e} | t: {elapsed:3.0f}s"
                )
                sys.stdout.flush()

        train_acc = 100.0 * correct / total

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # noqa: PLW2901
                if is_energy_model:
                    outputs = model(inputs, et_kwargs={"detach": True})
                else:
                    outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        test_acc = 100.0 * correct / total
        test_loss /= len(test_loader)

        print(f" | Val Loss: {test_loss:.4f} | Val Acc: {test_acc:.2f}%")

        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(running_loss),
                "train_acc": float(train_acc),
                "val_loss": float(test_loss),
                "val_acc": float(test_acc),
                "e_att": float(running_e_att),
                "e_hop": float(running_e_hop),
                "grad_norm": float(running_grad_norm),
                "lr": float(lr),
            }
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "model_name": model_name,
                    "total_params": total_params,
                },
                save_dir / "best_model.pth",
            )

    if is_energy_model and "handle" in locals():
        handle.remove()

    with (save_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining completed! Best accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {save_dir}")
    return save_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train models on CIFAR-100")
    parser.add_argument(
        "model", choices=["vit", "viet", "viset"], help="Model to train"
    )
    args = parser.parse_args()
    train(args.model)


if __name__ == "__main__":
    main()
