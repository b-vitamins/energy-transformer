"""Performance tests for the spec system."""

import gc
import time

import pytest
import torch

from energy_transformer.spec import (
    Context,
    Identity,
    configure_realisation,
    get_realisation_metrics,
    graph,
    loop,
    parallel,
    realise,
    reset_metrics,
    seq,
)
from energy_transformer.spec.library import (
    ClassificationHeadSpec,
    CLSTokenSpec,
    ETBlockSpec,
    HNSpec,
    LayerNormSpec,
    MHEASpec,
    PatchEmbedSpec,
    PosEmbedSpec,
)
from energy_transformer.spec.realise import _get_config


class TestSpecPerformance:
    """Benchmark spec system performance."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset state before each test."""
        # Clear cache
        _get_config().cache.clear()
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @pytest.mark.benchmark
    def test_realisation_speed(self, benchmark):
        """Benchmark basic realisation speed."""
        spec = seq(
            PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            loop(ETBlockSpec(), times=12),
            LayerNormSpec(),
            ClassificationHeadSpec(num_classes=1000),
        )

        # Benchmark realisation
        result = benchmark(realise, spec)
        assert isinstance(result, torch.nn.Module)

        # Check benchmark stats
        assert benchmark.stats['mean'] < 1.0  # Should complete in under 1 second

    @pytest.mark.benchmark
    def test_cache_performance(self, benchmark):
        """Benchmark cache hit performance."""
        # Enable cache
        config = _get_config()
        config.cache.enabled = True
        config.cache.clear()

        spec = LayerNormSpec()
        ctx = Context(dimensions={"embed_dim": 768})

        # Warm up cache
        realise(spec, ctx)

        # Benchmark cache hits
        def cache_hit():
            return realise(spec, ctx)

        result = benchmark(cache_hit)
        assert isinstance(result, torch.nn.Module)

        # Cache hits should be very fast
        assert benchmark.stats['mean'] < 0.001  # Under 1ms

        # Verify cache was used
        assert config.cache.hit_rate > 0.9

    @pytest.mark.benchmark
    def test_deep_nesting_performance(self, benchmark):
        """Benchmark deeply nested spec realisation."""
        # Create deeply nested spec
        depth = 50
        spec = LayerNormSpec()
        for _ in range(depth):
            spec = seq(spec, LayerNormSpec())

        ctx = Context(dimensions={"embed_dim": 768})

        # Configure for deep nesting
        configure_realisation(MAX_RECURSION=100)

        # Benchmark
        result = benchmark(realise, spec, ctx)
        assert isinstance(result, torch.nn.Module)

        # Should still be reasonably fast despite depth
        assert benchmark.stats['mean'] < 2.0

    @pytest.mark.benchmark
    def test_parallel_realisation_performance(self, benchmark):
        """Benchmark parallel spec realisation."""
        # Create parallel spec with many branches
        branches = [ETBlockSpec() for _ in range(10)]
        spec = parallel(*branches, merge="add")

        ctx = Context(dimensions={"embed_dim": 768})

        result = benchmark(realise, spec, ctx)
        assert isinstance(result, torch.nn.Module)

        # Parallel should not add too much overhead
        assert benchmark.stats['mean'] < 1.0

    @pytest.mark.benchmark
    def test_graph_realisation_performance(self, benchmark):
        """Benchmark graph-based spec realisation."""
        # Create a moderately complex graph
        g = graph()

        # Add nodes
        for i in range(10):
            g = g.add_node(f"layer_{i}", LayerNormSpec())

        # Connect in sequence
        g = g.add_edge("input", "layer_0")
        for i in range(9):
            g = g.add_edge(f"layer_{i}", f"layer_{i+1}")
        g = g.add_edge("layer_9", "output")

        g.inputs = ["input"]
        g.outputs = ["output"]

        ctx = Context(dimensions={"embed_dim": 768})

        result = benchmark(realise, g, ctx)
        assert isinstance(result, torch.nn.Module)

        # Graph overhead should be reasonable
        assert benchmark.stats['mean'] < 0.5

    @pytest.mark.memory_bench
    def test_memory_efficiency(self):
        """Test memory usage during large model realisation."""
        # Large model spec
        spec = seq(
            PatchEmbedSpec(img_size=384, patch_size=16, embed_dim=1024),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            loop(
                ETBlockSpec(
                    attention=MHEASpec(num_heads=16, head_dim=64),
                    hopfield=HNSpec(multiplier=4.0),
                ),
                times=24,  # Large model
            ),
            LayerNormSpec(),
            ClassificationHeadSpec(num_classes=21843),  # ImageNet-21k
        )

        # Measure memory before
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_memory = torch.cuda.memory_allocated()
        else:
            import psutil
            process = psutil.Process()
            start_memory = process.memory_info().rss

        # Realise model
        model = realise(spec)

        # Measure memory after
        if torch.cuda.is_available():
            end_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
        else:
            end_memory = process.memory_info().rss
            peak_memory = end_memory

        memory_used_mb = (end_memory - start_memory) / 1024 / 1024
        peak_memory_mb = peak_memory / 1024 / 1024

        print(f"\nMemory used: {memory_used_mb:.2f} MB")
        print(f"Peak memory: {peak_memory_mb:.2f} MB")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Verify reasonable memory usage
        param_memory_mb = sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024
        overhead_ratio = memory_used_mb / param_memory_mb if param_memory_mb > 0 else 0
        assert overhead_ratio < 2.0, f"Memory overhead too high: {overhead_ratio:.2f}x"

    @pytest.mark.benchmark
    def test_optimization_impact(self, benchmark):
        """Test impact of spec optimizations."""
        # Create spec with redundant identity nodes
        spec = seq(
            LayerNormSpec(),
            Identity(),  # Should be removed by optimization
            ETBlockSpec(),
            Identity(),  # Should be removed by optimization
            LayerNormSpec(),
        )

        ctx = Context(dimensions={"embed_dim": 768})

        # Test with optimizations enabled (default)
        configure_realisation(optimizations=True)
        benchmark.pedantic(
            realise,
            args=(spec, ctx),
            rounds=10,
            iterations=5
        )
        opt_time = benchmark.stats['mean']

        # Test with optimizations disabled
        configure_realisation(optimizations=False)
        _get_config().cache.clear()

        start = time.perf_counter()
        for _ in range(50):
            realise(spec, ctx)
        no_opt_time = (time.perf_counter() - start) / 50

        # Optimizations should make it faster or at least not slower
        assert opt_time <= no_opt_time * 1.1  # Allow 10% variance

        # Reset to default
        configure_realisation(optimizations=True)

    @pytest.mark.benchmark
    def test_metrics_overhead(self, benchmark):
        """Test overhead of metrics collection."""
        spec = seq(
            PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768),
            loop(ETBlockSpec(), times=6),
            LayerNormSpec(),
        )

        # Benchmark without metrics
        configure_realisation(enable_metrics=False)

        def no_metrics():
            _get_config().cache.clear()
            return realise(spec)

        benchmark.pedantic(no_metrics, rounds=10, iterations=5)
        no_metrics_time = benchmark.stats['mean']

        # Benchmark with metrics
        configure_realisation(enable_metrics=True)
        reset_metrics()

        def with_metrics():
            _get_config().cache.clear()
            return realise(spec)

        # Manually time since we need fresh benchmark
        start = time.perf_counter()
        for _ in range(50):
            with_metrics()
        with_metrics_time = (time.perf_counter() - start) / 50

        # Check metrics were collected
        metrics = get_realisation_metrics()
        assert metrics['specs_realised'] > 0

        # Metrics overhead should be minimal (< 20%)
        overhead = (with_metrics_time - no_metrics_time) / no_metrics_time
        assert overhead < 0.2, f"Metrics overhead too high: {overhead:.1%}"

        # Reset to default
        configure_realisation(enable_metrics=False)


@pytest.mark.benchmark
class TestScalability:
    """Test scalability with increasing complexity."""

    def test_linear_scaling_with_depth(self):
        """Test that realisation time scales linearly with depth."""
        times = []
        depths = [1, 5, 10, 20]

        for depth in depths:
            spec = loop(ETBlockSpec(), times=depth)
            ctx = Context(dimensions={"embed_dim": 768})

            # Clear cache for fair comparison
            _get_config().cache.clear()

            # Time realisation
            start = time.perf_counter()
            realise(spec, ctx)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        # Check approximately linear scaling
        # Time for depth 20 should be < 4.5x time for depth 5 (allowing overhead)
        scaling_factor = times[-1] / times[1]
        expected_factor = depths[-1] / depths[1]
        assert scaling_factor < expected_factor * 1.2, \
            f"Non-linear scaling: {scaling_factor:.2f}x for {expected_factor}x depth"

