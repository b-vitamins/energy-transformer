"""Performance tests for realisation system."""

import time
from dataclasses import dataclass

import pytest

from energy_transformer.spec import (
    Context,
    Spec,
    configure_realisation,
    param,
    realise,
)
from energy_transformer.spec.realise import ModuleCache, register

pytestmark = pytest.mark.benchmark


@dataclass(frozen=True)
class BenchmarkSpec(Spec):
    """Simple spec for benchmarking."""

    size: int = param(default=100)


@register(BenchmarkSpec)
def realise_benchmark(spec, _context):
    from torch import nn

    # Create a more complex module to make caching worthwhile
    return nn.Sequential(
        nn.Linear(spec.size, spec.size * 2),
        nn.ReLU(),
        nn.Linear(spec.size * 2, spec.size * 4),
        nn.ReLU(),
        nn.Linear(spec.size * 4, spec.size),
    )


class TestCachePerformance:
    """Benchmark cache performance."""

    def test_cache_hit_performance(self):
        configure_realisation(cache=ModuleCache(max_size=1000, enabled=True))

        # Use fewer unique specs but realize them more times
        # This creates a better scenario for demonstrating cache benefits
        unique_sizes = [50, 100, 150, 200, 250]
        specs = [
            BenchmarkSpec(size=size) for size in unique_sizes
        ] * 20  # 100 total

        # Warm-up to stabilize timings
        for _ in range(5):
            for spec in specs[:5]:
                realise(spec)

        # Clear cache for fair comparison
        configure_realisation(cache=ModuleCache(enabled=False))

        # Time without cache
        start = time.perf_counter()
        for spec in specs:
            realise(spec)
        time_no_cache = time.perf_counter() - start

        # Time with cold cache
        configure_realisation(cache=ModuleCache(enabled=True))
        start = time.perf_counter()
        for spec in specs:
            realise(spec)
        time_cache_cold = time.perf_counter() - start

        # Time with hot cache (should be much faster)
        start = time.perf_counter()
        for spec in specs:
            realise(spec)
        time_cache_hot = time.perf_counter() - start

        print("\nCache Performance:")
        print(f"  No cache: {time_no_cache:.3f}s")
        print(f"  Cold cache: {time_cache_cold:.3f}s")
        print(f"  Hot cache: {time_cache_hot:.3f}s")
        print(f"  Speedup: {time_no_cache / time_cache_hot:.1f}x")

        # More realistic expectation: hot cache should be at least 30% faster
        # This accounts for CI environment variability
        assert time_cache_hot < time_no_cache * 0.7, (
            f"Cache speedup insufficient: {time_cache_hot:.3f}s (hot) "
            f"vs {time_no_cache:.3f}s (no cache), "
            f"ratio: {time_cache_hot / time_no_cache:.2f}"
        )

    def test_deep_structure_cache_keys(self):
        cache = ModuleCache()
        deep_dict = {}
        current = deep_dict
        for i in range(50):
            current[f"level_{i}"] = {}
            current = current[f"level_{i}"]
        ctx = Context(metadata={"deep": deep_dict})
        spec = BenchmarkSpec()
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            cache._make_key(spec, ctx)
        elapsed = time.perf_counter() - start
        per_key_ms = (elapsed / iterations) * 1000
        print(f"\nDeep structure key generation: {per_key_ms:.3f}ms per key")
        assert per_key_ms < 10
