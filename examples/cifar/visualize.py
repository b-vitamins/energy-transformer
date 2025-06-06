#!/usr/bin/env python3
"""Create publication-quality training plots for CIFAR-100 experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["lines.linewidth"] = 1.5
plt.rcParams["axes.linewidth"] = 0.8


def get_data_dir() -> Path:
    """Return XDG data directory for energy-transformer."""
    import os

    xdg_data_home = os.environ.get(
        "XDG_DATA_HOME", str(Path("~/.local/share").expanduser())
    )
    return Path(xdg_data_home) / "energy-transformer"


def load_history(model_name: str) -> dict[str, np.ndarray]:
    """Load training history for a model."""
    data_dir = get_data_dir() / "models" / model_name
    history_file = data_dir / "history.json"
    if not history_file.exists():
        raise FileNotFoundError(
            f"History not found for {model_name}. Train the model first."
        )
    with history_file.open() as f:
        history = json.load(f)
    data: dict[str, np.ndarray] = {}
    for key in [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "e_att",
        "e_hop",
        "grad_norm",
        "lr",
    ]:
        data[key] = np.array([h[key] for h in history])
    return data


def create_plot(model_name: str, output_path: str | None = None) -> None:  # noqa: PLR0915
    """Create training plot."""
    data = load_history(model_name)
    epochs = data["epoch"]

    color_train = "#1f77b4"
    color_val = "#ff7f0e"
    color_attention = "#2ca02c"
    color_hopfield = "#d62728"
    color_lr = "#9467bd"
    color_grad = "#8c564b"

    fig = plt.figure(figsize=(11, 14))
    gs = gridspec.GridSpec(
        5,
        2,
        height_ratios=[1.2, 1.2, 1, 1, 0.8],
        width_ratios=[1, 1],
        hspace=0.55,
        wspace=0.35,
        left=0.07,
        right=0.97,
        top=0.92,
        bottom=0.04,
    )

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(
        epochs,
        data["train_loss"],
        color=color_train,
        label="Training Loss",
        alpha=0.8,
    )
    ax1.plot(
        epochs,
        data["val_loss"],
        color=color_val,
        label="Validation Loss",
        linewidth=2,
    )
    ax1.set_xlabel("Epoch", labelpad=8)
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training and Validation Loss", fontweight="bold", pad=15)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 200)

    ax1.axvspan(0, 10, alpha=0.1, color="gray", label="Warmup")
    ax1.axvspan(10, 38, alpha=0.1, color="blue", label="Peak LR")
    ax1.axvspan(38, 200, alpha=0.1, color="green", label="Decay")

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(
        epochs,
        data["train_acc"],
        color=color_train,
        label="Training Accuracy",
        alpha=0.8,
    )
    ax2.plot(
        epochs,
        data["val_acc"],
        color=color_val,
        label="Validation Accuracy",
        linewidth=2,
    )
    ax2.set_xlabel("Epoch", labelpad=8)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training and Validation Accuracy", fontweight="bold", pad=15)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 200)
    ax2.set_ylim(0, 65)

    best_val_idx = np.argmax(data["val_acc"])
    ax2.plot(
        epochs[best_val_idx],
        data["val_acc"][best_val_idx],
        "r*",
        markersize=12,
        label=f"Best: {data['val_acc'][best_val_idx]:.2f}%",
    )
    ax2.legend(loc="lower right")

    ax3 = fig.add_subplot(gs[2, 0])
    if model_name in ["viet", "viset"] and np.any(data["e_att"] != 0):
        ax3.plot(
            epochs, data["e_att"] / 1000, color=color_attention, linewidth=1.5
        )
        ax3.set_ylabel("Attention Energy (x10³)")
    else:
        ax3.text(
            0.5,
            0.5,
            "N/A\n(Standard Model)",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=12,
            color="gray",
        )
        ax3.set_ylim(0, 1)
    ax3.set_xlabel("Epoch", labelpad=8)
    ax3.set_title("Attention Energy E(A)", fontweight="bold", pad=15)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 200)

    ax4 = fig.add_subplot(gs[2, 1])
    if model_name in ["viet", "viset"] and np.any(data["e_hop"] != 0):
        ax4.plot(
            epochs, data["e_hop"] / 1000, color=color_hopfield, linewidth=1.5
        )
        ax4.set_ylabel("Hopfield Energy (x10³)")
    else:
        ax4.text(
            0.5,
            0.5,
            "N/A\n(Standard Model)",
            ha="center",
            va="center",
            transform=ax4.transAxes,
            fontsize=12,
            color="gray",
        )
        ax4.set_ylim(0, 1)
    ax4.set_xlabel("Epoch", labelpad=8)
    ax4.set_title("Hopfield Energy E(H)", fontweight="bold", pad=15)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 200)

    ax5 = fig.add_subplot(gs[3, 0])
    ax5.semilogy(epochs, data["lr"], color=color_lr, linewidth=2)
    ax5.set_xlabel("Epoch", labelpad=8)
    ax5.set_ylabel("Learning Rate")
    ax5.set_title("Learning Rate Schedule", fontweight="bold", pad=15)
    ax5.grid(True, alpha=0.3, which="both")
    ax5.set_xlim(0, 200)
    ax5.set_ylim(5e-6, 2e-3)

    ax6 = fig.add_subplot(gs[3, 1])
    ax6.plot(
        epochs, data["grad_norm"], color=color_grad, linewidth=1.5, alpha=0.8
    )
    ax6.set_xlabel("Epoch", labelpad=8)
    ax6.set_ylabel("Gradient Norm")
    ax6.set_title("Gradient Norm |∇|", fontweight="bold", pad=15)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 200)
    ax6.set_ylim(0, 14)

    ax7 = fig.add_subplot(gs[4, :])
    ax7.axis("off")

    summary_data = [
        ["Metric", "Initial", "Best", "Final"],
        [
            "Training Loss",
            f"{data['train_loss'][0]:.3f}",
            f"{np.min(data['train_loss']):.3f}",
            f"{data['train_loss'][-1]:.3f}",
        ],
        [
            "Training Acc (%)",
            f"{data['train_acc'][0]:.1f}",
            f"{np.max(data['train_acc']):.1f}",
            f"{data['train_acc'][-1]:.1f}",
        ],
        [
            "Validation Loss",
            f"{data['val_loss'][0]:.3f}",
            f"{np.min(data['val_loss']):.3f}",
            f"{data['val_loss'][-1]:.3f}",
        ],
        [
            "Validation Acc (%)",
            f"{data['val_acc'][0]:.1f}",
            f"{np.max(data['val_acc']):.1f}",
            f"{data['val_acc'][-1]:.1f}",
        ],
        ["Best Val Epoch", "-", f"{epochs[best_val_idx]:.0f}", "-"],
    ]
    table = ax7.table(
        cellText=summary_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.35, 0.22, 0.22, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.8)
    for i in range(4):
        table[(0, i)].set_facecolor("#E6E6E6")
        table[(0, i)].set_text_props(weight="bold")

    checkpoint_path = get_data_dir() / "models" / model_name / "best_model.pth"
    if checkpoint_path.exists():
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        params = checkpoint["total_params"]
        params_str = f"{params / 1e6:.2f}M"
    else:
        params_str = "Unknown"

    if model_name == "viet":
        title = (
            "Energy Transformer Training on CIFAR-100\n2 Layers, 4 Iterations"
        )
    elif model_name == "viset":
        title = "Simplicial Energy Transformer Training on CIFAR-100\n2 Layers, 4 Iterations, Order-3"
    else:
        title = "Vision Transformer Training on CIFAR-100\n6 Layers (Baseline)"
    fig.suptitle(
        f"{title}, {params_str} Parameters",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    if output_path:
        plt.savefig(
            output_path,
            dpi=500,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize CIFAR-100 training results"
    )
    parser.add_argument(
        "model", choices=["vit", "viet", "viset"], help="Model to visualize"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path for the plot",
    )
    args = parser.parse_args()

    create_plot(args.model, args.output)


if __name__ == "__main__":
    main()
