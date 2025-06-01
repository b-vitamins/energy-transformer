"""Script to compare performance across model variants."""

import argparse
import json
import time
from pathlib import Path

import torch

from energy_transformer.models.vision import (
    viet_base,
    viet_tiny,
    viset_base,
    viset_tiny,
    vit_base,
    vit_tiny,
)


def benchmark_model(
    model_factory,
    model_name: str,
    config: dict,
    device: torch.device,
    num_runs: int = 100,
) -> dict:
    """Benchmark a single model configuration."""
    model = model_factory(**config).to(device).eval()
    param_count = sum(p.numel() for p in model.parameters())

    img_size = config.get("img_size", 224)
    x = torch.randn(1, 3, img_size, img_size, device=device)

    et_kwargs = (
        {"et_kwargs": {"detach": True}}
        if "viet" in model_name or "viset" in model_name
        else {}
    )

    with torch.no_grad():
        for _ in range(20):
            _ = model(x, **et_kwargs)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x, **et_kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    avg_time = elapsed / num_runs

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(x, **et_kwargs)
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_memory = 0.0

    return {
        "model": model_name,
        "params_millions": param_count / 1e6,
        "avg_time_ms": avg_time * 1000,
        "throughput_img_s": 1 / avg_time,
        "peak_memory_mb": peak_memory,
        "device": device.type,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model performance")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--output", default="performance_comparison.json")
    args = parser.parse_args()

    device = torch.device(args.device)

    models = {
        "vit_tiny": (
            vit_tiny,
            {"img_size": 224, "patch_size": 16, "num_classes": 1000},
        ),
        "viet_tiny": (
            viet_tiny,
            {"img_size": 224, "patch_size": 16, "num_classes": 1000},
        ),
        "viset_tiny": (
            viset_tiny,
            {"img_size": 224, "patch_size": 16, "num_classes": 1000},
        ),
        "vit_base": (
            vit_base,
            {"img_size": 224, "patch_size": 16, "num_classes": 1000},
        ),
        "viet_base": (
            viet_base,
            {"img_size": 224, "patch_size": 16, "num_classes": 1000},
        ),
        "viset_base": (
            viset_base,
            {"img_size": 224, "patch_size": 16, "num_classes": 1000},
        ),
    }

    results = []

    print(f"Benchmarking on {device}...")
    print("-" * 80)

    for model_name, (factory, config) in models.items():
        print(f"Benchmarking {model_name}...", end=" ", flush=True)
        result = benchmark_model(
            factory, model_name, config, device, args.num_runs
        )
        results.append(result)
        print(f"Done! ({result['avg_time_ms']:.2f}ms per inference)")

    results.sort(key=lambda x: x["avg_time_ms"])

    print("\n" + "=" * 80)
    print("Performance Summary:")
    print("-" * 80)
    print(
        f"{'Model':<15} {'Params (M)':<12} {'Time (ms)':<12} {'Throughput':<15} {'Memory (MB)':<12}"
    )
    print("-" * 80)

    for r in results:
        print(
            f"{r['model']:<15} {r['params_millions']:<12.1f} {r['avg_time_ms']:<12.2f} "
            f"{r['throughput_img_s']:<15.1f} {r['peak_memory_mb']:<12.1f}"
        )

    print("\n" + "=" * 80)
    print("Energy Transformer Overhead:")
    print("-" * 80)

    for size in ["tiny", "base"]:
        vit_result = next(r for r in results if r["model"] == f"vit_{size}")
        viet_result = next(r for r in results if r["model"] == f"viet_{size}")
        viset_result = next(r for r in results if r["model"] == f"viset_{size}")

        viet_overhead = (
            viet_result["avg_time_ms"] / vit_result["avg_time_ms"] - 1
        ) * 100
        viset_overhead = (
            viset_result["avg_time_ms"] / vit_result["avg_time_ms"] - 1
        ) * 100

        print(f"{size.capitalize()} models:")
        print(f"  ViET overhead:  {viet_overhead:+.1f}%")
        print(f"  ViSET overhead: {viset_overhead:+.1f}%")

    with Path(args.output).open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
