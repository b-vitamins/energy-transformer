#!/usr/bin/env python3
"""Profile memory usage of models in detail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from energy_transformer.models.vision import (
    vit_tiny,
    viet_tiny,
    viset_tiny,
    vit_base,
    viet_base,
    viset_base,
)
from memory_utils import MemoryProfiler, analyze_memory_scaling


def profile_model_memory(
    model_name: str,
    model_factory,
    model_config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Profile detailed memory usage of a model."""
    profiler = MemoryProfiler(device)

    print(f"\nProfiling {model_name}...")

    profiler.clean_memory()
    before_model = profiler.get_memory_snapshot()

    model = model_factory(**model_config).to(device).eval()

    after_model = profiler.get_memory_snapshot()
    model_memory = after_model - before_model

    param_size = (
        sum(p.numel() * p.element_size() for p in model.parameters())
        / 1024
        / 1024
    )
    buffer_size = (
        sum(b.numel() * b.element_size() for b in model.buffers()) / 1024 / 1024
    )

    img_size = model_config.get("img_size", 224)
    x = torch.randn(1, 3, img_size, img_size, device=device)

    profiler.clean_memory()
    before_inference = profiler.get_memory_snapshot()

    with torch.no_grad():
        kwargs = (
            {"et_kwargs": {"detach": True}}
            if "viet" in model_name or "viset" in model_name
            else {}
        )
        _ = model(x, **kwargs)

    after_inference = profiler.get_memory_snapshot()
    inference_memory = after_inference - before_inference

    batch_sizes = [1, 2, 4, 8, 16] if device.type == "cuda" else [1, 2, 4]
    scaling_results = analyze_memory_scaling(
        model_factory,
        model_config,
        batch_sizes,
        (3, img_size, img_size),
        device,
    )

    return {
        "model": model_name,
        "param_size_mb": param_size,
        "buffer_size_mb": buffer_size,
        "model_memory": {
            "gpu_mb": model_memory.gpu_allocated_mb,
            "cpu_mb": model_memory.cpu_rss_mb,
        },
        "inference_memory": {
            "gpu_mb": inference_memory.gpu_allocated_mb,
            "cpu_mb": inference_memory.cpu_rss_mb,
            "peak_gpu_mb": inference_memory.gpu_max_allocated_mb,
        },
        "batch_scaling": scaling_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile model memory usage")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["vit_tiny", "viet_tiny", "viset_tiny"],
    )
    parser.add_argument("--output", default="memory_profile.json")
    args = parser.parse_args()

    device = torch.device(args.device)

    all_models = {
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

    for model_name in args.models:
        if model_name not in all_models:
            print(f"Unknown model: {model_name}")
            continue

        factory, config = all_models[model_name]
        result = profile_model_memory(model_name, factory, config, device)
        results.append(result)

        print(f"\n{model_name} Memory Summary:")
        print(f"  Parameters: {result['param_size_mb']:.1f} MB")
        print(f"  Buffers: {result['buffer_size_mb']:.1f} MB")
        print(
            f"  Model total: {result['model_memory']['gpu_mb' if device.type == 'cuda' else 'cpu_mb']:.1f} MB"
        )
        print(
            f"  Inference: {result['inference_memory']['gpu_mb' if device.type == 'cuda' else 'cpu_mb']:.1f} MB"
        )
        if device.type == "cuda":
            print(
                f"  Peak GPU: {result['inference_memory']['peak_gpu_mb']:.1f} MB"
            )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
