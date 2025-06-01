"""Performance metrics for the spec system."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RealisationMetrics:
    """Metrics collected during spec realisation."""

    # Timing metrics
    total_time: float = 0.0
    cache_lookup_time: float = 0.0
    realisation_time: float = 0.0

    # Count metrics
    specs_realised: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    # Complexity metrics
    max_depth: int = 0

    # Per-spec breakdown
    spec_times: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    spec_counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def add_spec_time(self, spec_name: str, duration: float) -> None:
        """Record time taken to realise a specific spec type."""
        self.spec_times[spec_name].append(duration)
        self.spec_counts[spec_name] += 1

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of collected metrics."""
        cache_total = self.cache_hits + self.cache_misses
        summary: dict[str, Any] = {
            "total_time": self.total_time,
            "specs_realised": self.specs_realised,
            "cache_hit_rate": self.cache_hits / cache_total
            if cache_total > 0
            else 0.0,
            "avg_realisation_time": self.realisation_time / self.specs_realised
            if self.specs_realised > 0
            else 0.0,
            "max_depth": self.max_depth,
        }

        spec_stats: dict[str, dict[str, float | int]] = {}
        for spec_name, times in self.spec_times.items():
            if times:
                spec_stats[spec_name] = {
                    "count": self.spec_counts[spec_name],
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                }
        summary["spec_stats"] = spec_stats
        return summary


class MetricsCollector:
    """Thread-safe metrics collector."""

    def __init__(self) -> None:
        self._metrics = RealisationMetrics()
        self._lock = threading.Lock()
        self.enabled = False

    @contextmanager
    def timer(self, metric_name: str) -> Iterator[None]:
        """Context manager for timing operations."""
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            with self._lock:
                current = getattr(self._metrics, metric_name, 0.0)
                setattr(self._metrics, metric_name, current + duration)

    @contextmanager
    def spec_timer(self, spec_name: str) -> Iterator[None]:
        """Context manager for timing spec realisation."""
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            with self._lock:
                self._metrics.add_spec_time(spec_name, duration)
                self._metrics.specs_realised += 1
                self._metrics.realisation_time += duration

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        if not self.enabled:
            return
        with self._lock:
            self._metrics.cache_hits += 1

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        if not self.enabled:
            return
        with self._lock:
            self._metrics.cache_misses += 1

    def update_max_depth(self, depth: int) -> None:
        """Update maximum recursion depth."""
        if not self.enabled:
            return
        with self._lock:
            self._metrics.max_depth = max(self._metrics.max_depth, depth)

    def get_metrics(self) -> RealisationMetrics:
        """Get a copy of current metrics."""
        with self._lock:
            import copy

            return copy.deepcopy(self._metrics)

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics = RealisationMetrics()
