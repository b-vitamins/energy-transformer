"""Performance tests for realisation system."""

import time
from dataclasses import dataclass

from energy_transformer.spec import (
    Context,
    Spec,
    configure_realisation,
    param,
    realise,
)
from energy_transformer.spec.realise import ModuleCache, register


@dataclass(frozen=True)
class BenchmarkSpec(Spec):
    """Simple spec for benchmarking."""

    size: int = param(default=100)


@register(BenchmarkSpec)
def realise_benchmark(spec, context):
    import torch.nn as nn

    return nn.Linear(spec.size, spec.size)


class TestCachePerformance:
    """Benchmark cache performance."""

    def test_cache_hit_performance(self):
        configure_realisation(cache=ModuleCache(max_size=1000, enabled=True))
        specs = [BenchmarkSpec(size=i % 10 + 1) for i in range(100)]

        configure_realisation(cache=ModuleCache(enabled=False))
        start = time.perf_counter()
        for spec in specs:
            realise(spec)
        time_no_cache = time.perf_counter() - start

        configure_realisation(cache=ModuleCache(enabled=True))
        start = time.perf_counter()
        for spec in specs:
            realise(spec)
        time_cache_cold = time.perf_counter() - start

        start = time.perf_counter()
        for spec in specs:
            realise(spec)
        time_cache_hot = time.perf_counter() - start

        print("\nCache Performance:")
        print(f"  No cache: {time_no_cache:.3f}s")
        print(f"  Cold cache: {time_cache_cold:.3f}s")
        print(f"  Hot cache: {time_cache_hot:.3f}s")
        print(f"  Speedup: {time_no_cache / time_cache_hot:.1f}x")

        assert time_cache_hot < time_no_cache * 0.4

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
