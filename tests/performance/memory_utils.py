"""Memory profiling utilities for performance tests."""

from __future__ import annotations

import gc
import os
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import psutil
import torch


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""

    cpu_rss_mb: float
    cpu_vms_mb: float
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_max_allocated_mb: float

    def __sub__(self, other: MemorySnapshot) -> MemorySnapshot:
        """Calculate memory difference."""
        return MemorySnapshot(
            cpu_rss_mb=self.cpu_rss_mb - other.cpu_rss_mb,
            cpu_vms_mb=self.cpu_vms_mb - other.cpu_vms_mb,
            gpu_allocated_mb=self.gpu_allocated_mb - other.gpu_allocated_mb,
            gpu_reserved_mb=self.gpu_reserved_mb - other.gpu_reserved_mb,
            gpu_max_allocated_mb=self.gpu_max_allocated_mb
            - other.gpu_max_allocated_mb,
        )


class MemoryProfiler:
    """Profile memory usage during model operations."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.process = psutil.Process(os.getpid())

    def get_memory_snapshot(self) -> MemorySnapshot:
        """Get current memory usage."""
        mem_info = self.process.memory_info()
        cpu_rss = mem_info.rss / 1024 / 1024
        cpu_vms = mem_info.vms / 1024 / 1024

        if self.device.type == "cuda":
            gpu_allocated = (
                torch.cuda.memory_allocated(self.device) / 1024 / 1024
            )
            gpu_reserved = torch.cuda.memory_reserved(self.device) / 1024 / 1024
            gpu_max_allocated = (
                torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            )
        else:
            gpu_allocated = gpu_reserved = gpu_max_allocated = 0.0

        return MemorySnapshot(
            cpu_rss_mb=cpu_rss,
            cpu_vms_mb=cpu_vms,
            gpu_allocated_mb=gpu_allocated,
            gpu_reserved_mb=gpu_reserved,
            gpu_max_allocated_mb=gpu_max_allocated,
        )

    @contextmanager
    def measure_memory_usage(self, clean_before: bool = True):
        """Context manager to measure memory usage of a block."""
        if clean_before:
            self.clean_memory()

        before = self.get_memory_snapshot()

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        yield

        after = self.get_memory_snapshot()
        diff = after - before

        return diff, after

    def clean_memory(self) -> None:
        """Clean up memory before measurement."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)

    def profile_function(
        self,
        func: Callable[..., Any],
        *args: Any,
        num_runs: int = 1,
        return_output: bool = False,
        **kwargs: Any,
    ) -> tuple[MemorySnapshot, Any]:
        """Profile memory usage of a function."""
        self.clean_memory()

        output = func(*args, **kwargs)

        self.clean_memory()

        before = self.get_memory_snapshot()

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        for _ in range(num_runs):
            output = func(*args, **kwargs)

        after = self.get_memory_snapshot()
        memory_used = after - before

        if self.device.type == "cuda":
            memory_used.gpu_max_allocated_mb = (
                torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            )

        if return_output:
            return memory_used, output
        return memory_used, None


def analyze_memory_scaling(
    model_factory: Callable[..., Any],
    model_kwargs: dict[str, Any],
    batch_sizes: list[int],
    input_size: tuple[int, ...],
    device: torch.device,
) -> dict[str, list[tuple[int, float]]]:
    """Analyze how memory scales with batch size."""
    profiler = MemoryProfiler(device)
    results = {
        "model_memory": [],
        "activation_memory": [],
        "peak_memory": [],
    }

    profiler.clean_memory()
    model = model_factory(**model_kwargs).to(device).eval()
    model_snapshot = profiler.get_memory_snapshot()
    if device.type == "cuda":
        model_memory = model_snapshot.gpu_allocated_mb
    else:
        model_memory = model_snapshot.cpu_rss_mb

    for batch_size in batch_sizes:
        profiler.clean_memory()
        x = torch.randn(batch_size, *input_size, device=device)
        with torch.no_grad():
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            before = profiler.get_memory_snapshot()
            _ = model(x)
            after = profiler.get_memory_snapshot()
            if device.type == "cuda":
                activation_memory = (
                    after.gpu_allocated_mb - before.gpu_allocated_mb
                )
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            else:
                activation_memory = after.cpu_rss_mb - before.cpu_rss_mb
                peak_memory = after.cpu_rss_mb

        results["model_memory"].append((batch_size, model_memory))
        results["activation_memory"].append((batch_size, activation_memory))
        results["peak_memory"].append((batch_size, peak_memory))

        del _
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results
