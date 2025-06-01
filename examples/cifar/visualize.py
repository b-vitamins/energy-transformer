"""Visualize ablation results."""

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

DATA_HOME = (
    Path.home() / ".local" / "share" / "energy-transformer" / "experiments"
)


def load_latest_results() -> tuple[list[dict[str, Any]], Path]:
    """Load most recent results."""
    if not DATA_HOME.exists():
        print("No experiments found. Run ablation.py first.")
        sys.exit(1)

    latest = max(DATA_HOME.glob("ablation_*"), key=lambda p: p.stat().st_mtime)
    with open(latest / "results.json") as f:
        return json.load(f), latest


def plot_results(results: list[dict[str, Any]], save_dir: Path) -> None:
    """Create minimal comparison plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Sort by validation accuracy
    results = sorted(results, key=lambda x: x["best_val"], reverse=True)
    names = [r["name"] for r in results]
    val_accs = [r["best_val"] for r in results]
    test_accs = [r["best_test"] for r in results]

    # Accuracy comparison
    x = np.arange(len(names))
    width = 0.35

    ax1.bar(x - width / 2, val_accs, width, label="Validation", alpha=0.8)
    ax1.bar(x + width / 2, test_accs, width, label="Test", alpha=0.8)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Add values on bars
    for i, (v, t) in enumerate(zip(val_accs, test_accs, strict=False)):
        ax1.text(
            i - width / 2,
            v + 0.5,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax1.text(
            i + width / 2,
            t + 0.5,
            f"{t:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Parameter efficiency
    params = [r["params"] / 1e6 for r in results]
    efficiency = [v / p for v, p in zip(val_accs, params, strict=False)]

    ax2.bar(names, efficiency, alpha=0.8)
    ax2.set_ylabel("Accuracy per Million Parameters")
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Topology analysis if available
    viset_results = {r["name"]: r for r in results if "ViSET" in r["name"]}
    if len(viset_results) >= 3:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Extract different configurations
        configs = ["E100", "E50-T50", "T100", "Random"]
        accs = []
        for cfg in configs:
            for name, res in viset_results.items():
                if cfg in name:
                    accs.append(res["best_val"])
                    break

        if len(accs) == len(configs):
            colors = ["blue", "green", "red", "gray"]
            bars = ax.bar(configs, accs, color=colors, alpha=0.7)
            ax.set_ylabel("Validation Accuracy (%)")
            ax.set_title("Impact of Simplex Configuration")
            ax.grid(axis="y", alpha=0.3)

            for bar, acc in zip(bars, accs, strict=False):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{acc:.1f}%",
                    ha="center",
                    va="bottom",
                )

            plt.tight_layout()
            plt.savefig(
                save_dir / "topology_impact.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()


def main() -> None:
    """Generate visualizations."""
    results, exp_dir = load_latest_results()
    plot_results(results, exp_dir)

    print(f"Visualizations saved to: {exp_dir}")
    print("\nSummary:")
    print("-" * 50)

    # Best model
    best = max(results, key=lambda x: x["best_val"])
    print(f"Best model: {best['name']}")
    print(f"Validation: {best['best_val']:.2f}%")
    print(f"Test: {best['best_test']:.2f}%")

    # Key insights
    viset = next((r for r in results if r["name"] == "ViSET-E50-T50"), None)
    viset_rand = next((r for r in results if r["name"] == "ViSET-Random"), None)

    if viset and viset_rand:
        diff = viset["best_val"] - viset_rand["best_val"]
        print(f"\nTopology impact: {diff:+.2f}% improvement")


if __name__ == "__main__":
    main()
