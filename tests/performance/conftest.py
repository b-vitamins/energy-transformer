"""Performance testing configuration and fixtures."""

import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison."""

    mean: float
    stddev: float
    min: float
    max: float

    def is_regression(self, new_mean: float, threshold: float = 0.1) -> bool:
        """Check if new measurement is a regression."""
        return new_mean > self.mean * (1 + threshold)


class PerformanceTracker:
    """Track and compare performance across runs."""

    def __init__(
        self, baseline_file: Path = Path("tests/performance/baselines.json")
    ):
        self.baseline_file = baseline_file
        self.baselines: dict[str, PerformanceBaseline] = {}
        self._load_baselines()

    def _load_baselines(self) -> None:
        """Load baseline performance data."""
        if self.baseline_file.exists():
            with self.baseline_file.open() as f:
                data = json.load(f)
                for key, values in data.items():
                    self.baselines[key] = PerformanceBaseline(**values)

    def check_regression(
        self, name: str, stats: dict[str, float], threshold: float = 0.1
    ) -> bool:
        """Check if current performance is a regression."""
        if name not in self.baselines:
            return False
        baseline = self.baselines[name]
        return baseline.is_regression(stats["mean"], threshold)

    def update_baseline(self, name: str, stats: dict[str, float]) -> None:
        """Update baseline with new performance data."""
        self.baselines[name] = PerformanceBaseline(
            mean=stats["mean"],
            stddev=stats["stddev"],
            min=stats["min"],
            max=stats["max"],
        )


@pytest.fixture(scope="session")
def performance_tracker() -> PerformanceTracker:
    """Global performance tracker."""
    return PerformanceTracker()


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    """Check if CUDA is available for benchmarks."""
    return torch.cuda.is_available()


@pytest.fixture
def device(cuda_available: bool) -> torch.device:
    """Get device for benchmarks, prefer CUDA if available."""
    if cuda_available and not os.environ.get("FORCE_CPU_BENCHMARKS"):
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture
def clean_state(device: torch.device) -> None:
    """Clean up memory and state before and after each benchmark."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    yield

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


@pytest.fixture(params=[1, 4, 16])
def batch_size(request: Any) -> int:
    """Common batch sizes for benchmarking."""
    return request.param


@pytest.fixture(params=[32, 224])
def image_size(request: Any) -> int:
    """Common image sizes for vision models (CIFAR/ImageNet)."""
    return request.param


@pytest.fixture
def model_configs() -> dict[str, dict[str, Any]]:
    """Configuration for different model architectures."""
    return {
        "vit": {"et_steps": 0},
        "viet": {"et_steps": 4, "et_alpha": 0.125},
        "viset": {"et_steps": 4, "et_alpha": 0.125, "use_topology": True},
    }


def assert_performance(
    benchmark: Any,
    performance_tracker: PerformanceTracker,
    test_name: str,
    device_type: str,
    thresholds: dict[str, float],
) -> None:
    """Assert performance meets requirements and check for regressions."""
    # Access benchmark stats correctly
    stats_dict = {
        "mean": benchmark.stats.mean,
        "stddev": benchmark.stats.stddev,
        "min": benchmark.stats.min,
        "max": benchmark.stats.max,
    }
    full_name = f"{test_name}_{device_type}"

    if device_type in thresholds:
        assert stats_dict["mean"] < thresholds[device_type], (
            f"Performance threshold exceeded: {stats_dict['mean']:.3f}s > {thresholds[device_type]}s"
        )

    if performance_tracker.check_regression(full_name, stats_dict):
        pytest.fail(
            f"Performance regression detected for {full_name}: "
            f"current={stats_dict['mean']:.3f}s vs baseline={performance_tracker.baselines[full_name].mean:.3f}s"
        )
