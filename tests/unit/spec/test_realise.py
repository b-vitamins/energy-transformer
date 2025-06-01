"""Tests for specification realisation system.

This module tests the realisation functionality including caching,
plugins, error handling, and module generation.
"""

import warnings
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from energy_transformer.spec import library
from energy_transformer.spec.combinators import (
    Graph,
    Identity,
    Lambda,
    Sequential,
    cond,
    graph,
    loop,
    parallel,
    residual,
    seq,
    switch,
)
from energy_transformer.spec.primitives import (
    Context,
    Spec,
    SpecMeta,
    ValidationError,
    param,
    provides,
    requires,
)
from energy_transformer.spec.realise import (
    GraphModule,
    LoopModule,
    ModuleCache,
    ParallelModule,
    RealisationError,
    Realiser,
    ResidualModule,
    configure_realisation,
    from_yaml,
    optimize_spec,
    realise,
    register,
    register_typed,
    to_yaml,
    visualize,
)


# Mock specs for testing
@dataclass(frozen=True)
class SimpleSpec(Spec):
    """Simple spec for testing."""

    size: int = param(default=100)  # noqa: RUF009


@dataclass(frozen=True)
@requires("input_dim")
@provides("output_dim")
class LinearSpec(Spec):
    """Linear layer spec."""

    output_dim: int = param(default=256)  # noqa: RUF009
    bias: bool = param(default=True)  # noqa: RUF009

    def apply_context(self, context: Context) -> Context:
        context = super().apply_context(context)
        context.set_dim("output_dim", self.output_dim)
        return context


@dataclass(frozen=True)
class FailingSpec(Spec):
    """Spec that always fails realisation."""

    message: str = param(default="Intentional failure")  # noqa: RUF009


class TestModuleCache:
    """Test ModuleCache functionality."""

    def test_basic_caching(self):
        """Test basic cache operations."""
        cache = ModuleCache(max_size=10)

        spec = SimpleSpec(size=42)
        ctx = Context(dimensions={"test": 1})
        module = nn.Linear(10, 10)

        # Initial state
        assert cache.get(spec, ctx) is None
        assert cache.hit_rate == 0.0

        # Put and get
        cache.put(spec, ctx, module)
        cached = cache.get(spec, ctx)
        assert cached is module
        assert cache.hit_rate == 0.5  # 1 hit, 1 miss

        # Multiple hits
        cache.get(spec, ctx)
        cache.get(spec, ctx)
        assert cache.hit_rate == 0.75  # 3 hits, 1 miss

    def test_cache_key_generation(self):
        """Test cache keys are unique for different specs/contexts."""
        cache = ModuleCache()

        spec1 = SimpleSpec(size=10)
        spec2 = SimpleSpec(size=20)
        ctx1 = Context(dimensions={"a": 1})
        ctx2 = Context(dimensions={"a": 2})

        module1 = nn.Linear(1, 1)
        module2 = nn.Linear(2, 2)
        module3 = nn.Linear(3, 3)
        module4 = nn.Linear(4, 4)

        # Different combinations
        cache.put(spec1, ctx1, module1)
        cache.put(spec1, ctx2, module2)
        cache.put(spec2, ctx1, module3)
        cache.put(spec2, ctx2, module4)

        # Verify all are cached separately
        assert cache.get(spec1, ctx1) is module1
        assert cache.get(spec1, ctx2) is module2
        assert cache.get(spec2, ctx1) is module3
        assert cache.get(spec2, ctx2) is module4

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = ModuleCache(max_size=3)

        specs = [SimpleSpec(size=i) for i in range(5)]
        ctx = Context()
        modules = [nn.Linear(i + 1, i + 1) for i in range(5)]

        # Fill cache
        for i in range(3):
            cache.put(specs[i], ctx, modules[i])

        # Access first to make it recently used
        cache.get(specs[0], ctx)

        # Add new item - should evict specs[1] (least recently used)
        cache.put(specs[3], ctx, modules[3])

        assert cache.get(specs[0], ctx) is modules[0]  # Still there
        assert cache.get(specs[1], ctx) is None  # Evicted
        assert cache.get(specs[2], ctx) is modules[2]  # Still there
        assert cache.get(specs[3], ctx) is modules[3]  # New one

    def test_disabled_cache(self):
        """Test disabled cache behavior."""
        cache = ModuleCache(enabled=False)

        spec = SimpleSpec()
        ctx = Context()
        module = nn.Identity()

        cache.put(spec, ctx, module)
        assert cache.get(spec, ctx) is None

        # Hit rate should stay 0
        assert cache.hit_rate == 0.0

    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = ModuleCache()

        # Add some items
        for i in range(5):
            cache.put(SimpleSpec(size=i), Context(), nn.Linear(i + 1, i + 1))

        # Verify they're cached
        assert cache.get(SimpleSpec(size=0), Context()) is not None

        # Clear
        cache.clear()

        # Verify cache is empty
        assert cache.get(SimpleSpec(size=0), Context()) is None
        assert cache.hit_rate == 0.0  # Stats also reset

    def test_complex_cache_keys(self):
        """Test caching with complex nested structures."""
        cache = ModuleCache()

        # Nested specs
        spec1 = seq(SimpleSpec(1), SimpleSpec(2))
        spec2 = seq(SimpleSpec(1), SimpleSpec(2))  # Same structure
        spec3 = seq(SimpleSpec(2), SimpleSpec(1))  # Different order

        ctx = Context()
        module1 = nn.Sequential()
        module2 = nn.Sequential()

        cache.put(spec1, ctx, module1)

        # Same structure should hit cache
        assert cache.get(spec2, ctx) is module1

        # Different structure should miss
        cache.put(spec3, ctx, module2)
        assert cache.get(spec3, ctx) is module2
        assert cache.get(spec3, ctx) is not module1

    def test_etblockspec_key_uses_id(self):
        """ETBlockSpec instances use their id in cache keys."""
        cache = ModuleCache()
        spec = library.ETBlockSpec(
            attention=library.MHEASpec(num_heads=1, head_dim=16),
            hopfield=library.HNSpec(),
        )
        key = cache._make_key(spec, Context())
        assert key[0] == "nocache"
        assert key[1] == id(spec)


class TestRealisationError:
    """Test RealisationError functionality."""

    def test_error_construction(self):
        """Test error construction with all fields."""
        spec = SimpleSpec()
        ctx = Context(dimensions={"test": 42})
        cause = ValueError("underlying error")

        err = RealisationError(
            "Main message",
            spec=spec,
            context=ctx,
            cause=cause,
            suggestion="Try this fix",
        )

        assert err.spec is spec
        assert err.context is ctx
        assert err.cause is cause
        assert err.suggestion == "Try this fix"

    def test_error_string_representation(self):
        """Test error message formatting."""
        err = RealisationError(
            "Failed to realise",
            spec=SimpleSpec(size=42),
            context=Context(dimensions={"dim": 100}),
            cause=RuntimeError("boom"),
            suggestion="Check your configuration",
        )

        msg = str(err)
        assert "Failed to realise" in msg
        assert "SimpleSpec" in msg
        assert "RuntimeError: boom" in msg
        assert "Check your configuration" in msg
        assert "dim" in msg  # Context info

    def test_minimal_error(self):
        """Test error with minimal information."""
        err = RealisationError("Just a message")

        msg = str(err)
        assert msg == "Just a message"


class TestConfiguration:
    """Test realisation configuration."""

    def test_configure_basic_options(self):
        """Test configuring basic options."""
        # Set up new configuration
        configure_realisation(
            strict=False,
            warnings=False,
            auto_import=False,
            optimizations=False,
            max_recursion=50,
        )

        # Reset to defaults for other tests
        configure_realisation(
            strict=True,
            warnings=True,
            auto_import=True,
            optimizations=True,
            max_recursion=100,
        )

    def test_configure_cache(self):
        """Test configuring cache."""
        new_cache = ModuleCache(max_size=256, enabled=False)
        configure_realisation(cache=new_cache)

        # Reset
        configure_realisation(cache=ModuleCache())

    def test_configure_plugins(self):
        """Test configuring plugins."""

        class TestPlugin:
            def can_realise(self, spec):
                return isinstance(spec, SimpleSpec)

            def realise(self, spec, _context):
                return nn.Linear(spec.size, spec.size)

        plugin = TestPlugin()
        configure_realisation(plugins=[plugin])

        # Reset
        configure_realisation(plugins=[])

    def test_invalid_configuration(self):
        """Test invalid configuration raises error."""
        with pytest.raises(ValueError, match="Unknown configuration"):
            configure_realisation(invalid_option=True)


class TestRealiserPlugins:
    """Test plugin system."""

    def test_plugin_protocol(self):
        """Test plugin protocol implementation."""

        class WorkingPlugin:
            def can_realise(self, spec):
                return spec.size > 50 if isinstance(spec, SimpleSpec) else False

            def realise(self, spec, _context):
                return nn.Linear(spec.size, spec.size)

        plugin = WorkingPlugin()

        # Test can_realise
        assert plugin.can_realise(SimpleSpec(size=100))
        assert not plugin.can_realise(SimpleSpec(size=10))
        assert not plugin.can_realise(LinearSpec())

        # Test realise
        module = plugin.realise(SimpleSpec(size=100), Context())
        assert isinstance(module, nn.Linear)
        assert module.in_features == 100

    def test_plugin_in_realiser(self):
        """Test plugin integration with realiser."""

        class CustomPlugin:
            def can_realise(self, spec):
                return isinstance(spec, SimpleSpec) and spec.size == 42

            def realise(self, spec, _context):
                # Return a custom module
                return nn.Conv2d(3, spec.size, 3)

        # Configure with plugin
        configure_realisation(plugins=[CustomPlugin()])

        try:
            realiser = Realiser()

            # Should use plugin for size=42
            module = realiser.realise(SimpleSpec(size=42))
            assert isinstance(module, nn.Conv2d)
            assert module.out_channels == 42

            # Should fail for other sizes (no realiser)
            with pytest.raises(RealisationError):
                realiser.realise(SimpleSpec(size=100))
        finally:
            # Clean up
            configure_realisation(plugins=[])

    def test_plugin_exception_handling(self):
        """Test plugin exceptions are handled."""

        class BrokenPlugin:
            def can_realise(self, spec):
                return isinstance(spec, SimpleSpec)

            def realise(self, _spec, _context):
                raise RuntimeError("Plugin is broken")

        configure_realisation(plugins=[BrokenPlugin()], warnings=True)

        try:
            realiser = Realiser()

            # Should catch plugin error and continue
            with warnings.catch_warnings(record=True) as w:
                with pytest.raises(RealisationError, match="No realiser"):
                    realiser.realise(SimpleSpec())

                # Should have warned about plugin failure
                assert len(w) > 0
                assert "Plugin" in str(w[0].message)
                assert "failed" in str(w[0].message)
        finally:
            configure_realisation(plugins=[], warnings=True)

    def test_multiple_plugins(self):
        """Test multiple plugins with priority."""

        class PluginA:
            def can_realise(self, spec):
                return isinstance(spec, SimpleSpec) and spec.size < 50

            def realise(self, spec, _context):
                return nn.Linear(spec.size, 10)

        class PluginB:
            def can_realise(self, spec):
                return isinstance(spec, SimpleSpec) and spec.size >= 50

            def realise(self, spec, _context):
                return nn.Linear(spec.size, 20)

        configure_realisation(plugins=[PluginA(), PluginB()])

        try:
            realiser = Realiser()

            # Small size uses PluginA
            module1 = realiser.realise(SimpleSpec(size=30))
            assert module1.out_features == 10

            # Large size uses PluginB
            module2 = realiser.realise(SimpleSpec(size=70))
            assert module2.out_features == 20
        finally:
            configure_realisation(plugins=[])


class TestRealiser:
    """Test main Realiser class."""

    def test_basic_realisation(self):
        """Test basic spec realisation."""

        @register(SimpleSpec)
        def realise_simple(spec, _context):
            return nn.Linear(spec.size, spec.size)

        try:
            realiser = Realiser()
            module = realiser.realise(SimpleSpec(size=64))

            assert isinstance(module, nn.Linear)
            assert module.in_features == 64
            assert module.out_features == 64
        finally:
            SpecMeta._realisers.pop(SimpleSpec, None)

    def test_context_propagation(self):
        """Test context is passed to realiser."""

        @register(LinearSpec)
        def realise_linear(spec, context):
            input_dim = context.get_dim("input_dim")
            if input_dim is None:
                raise RealisationError("Missing input_dim")
            return nn.Linear(input_dim, spec.output_dim, bias=spec.bias)

        try:
            realiser = Realiser(Context(dimensions={"input_dim": 128}))
            module = realiser.realise(LinearSpec(output_dim=256))

            assert isinstance(module, nn.Linear)
            assert module.in_features == 128
            assert module.out_features == 256
            assert module.bias is not None
        finally:
            SpecMeta._realisers.pop(LinearSpec, None)

    def test_missing_realiser(self):
        """Test error when no realiser found."""
        realiser = Realiser()

        with pytest.raises(RealisationError) as exc_info:
            realiser.realise(SimpleSpec())

        assert "No realiser registered" in str(exc_info.value)
        assert "SimpleSpec" in str(exc_info.value)
        assert "@register" in str(exc_info.value)  # Suggestion

    def test_realiser_exception_handling(self):
        """Test exceptions in realisers are wrapped."""

        @register(FailingSpec)
        def realise_failing(spec, _context):
            raise RuntimeError(spec.message)

        try:
            realiser = Realiser()

            with pytest.raises(RealisationError) as exc_info:
                realiser.realise(FailingSpec(message="Boom!"))

            err = exc_info.value
            assert "Realiser failed" in str(err)
            assert "FailingSpec" in str(err)
            assert isinstance(err.cause, RuntimeError)
            assert str(err.cause) == "Boom!"
        finally:
            SpecMeta._realisers.pop(FailingSpec, None)

    def test_recursive_realisation(self):
        """Test realising nested specs."""

        @register(SimpleSpec)
        def realise_simple(spec, _context):
            return nn.Linear(10, spec.size)

        @register(LinearSpec)
        def realise_linear(spec, context):
            input_dim = context.get_dim("input_dim") or 10
            return nn.Linear(input_dim, spec.output_dim)

        try:
            # Sequential with proper context flow
            seq_spec = seq(SimpleSpec(size=20), LinearSpec(output_dim=30))

            realiser = Realiser(Context(dimensions={"input_dim": 20}))
            module = realiser.realise(seq_spec)

            assert isinstance(module, nn.Sequential)
            assert len(module) == 2
            assert isinstance(module[0], nn.Linear)
            assert isinstance(module[1], nn.Linear)
        finally:
            SpecMeta._realisers.pop(SimpleSpec, None)
            SpecMeta._realisers.pop(LinearSpec, None)

    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        # This is tricky to test without complex setup
        # Would need specs that reference each other

    def test_max_recursion_limit(self):
        """Test maximum recursion limit."""
        configure_realisation(max_recursion=5, optimizations=False)

        @register(SimpleSpec)
        def realise_simple(spec, _context):
            return nn.Linear(spec.size, spec.size)

        try:
            # Create deeply nested structure WITHOUT using seq() which flattens
            spec = SimpleSpec()
            for _ in range(10):
                # Use Sequential constructor directly to avoid flattening
                spec = Sequential(parts=(spec, SimpleSpec()))

            realiser = Realiser()

            with pytest.raises(RealisationError) as exc_info:
                realiser.realise(spec)

            assert "Maximum recursion" in str(exc_info.value)
        finally:
            SpecMeta._realisers.pop(SimpleSpec, None)
            configure_realisation(max_recursion=100, optimizations=True)

    def test_caching_integration(self):
        """Test caching works with realiser."""
        call_count = 0

        @register(SimpleSpec)
        def realise_simple(spec, _context):
            nonlocal call_count
            call_count += 1
            return nn.Linear(spec.size, spec.size)

        try:
            # Enable caching
            cache = ModuleCache(enabled=True)
            configure_realisation(cache=cache)

            realiser = Realiser()
            spec = SimpleSpec(size=50)

            # First call - cache miss
            module1 = realiser.realise(spec)
            assert call_count == 1

            # Second call - cache hit
            module2 = realiser.realise(spec)
            assert call_count == 1  # Not called again
            assert module2 is module1  # Same instance

            # Different spec - cache miss
            realiser.realise(SimpleSpec(size=100))
            assert call_count == 2
        finally:
            SpecMeta._realisers.pop(SimpleSpec, None)
            configure_realisation(cache=ModuleCache())


class TestBuiltinRealisers:
    """Test built-in combinator realisers."""

    def test_identity_realisation(self):
        """Test Identity spec realisation."""
        realiser = Realiser()
        module = realiser.realise(Identity())

        assert isinstance(module, nn.Identity)

    def test_lambda_realisation(self):
        """Test Lambda spec realisation."""

        def my_fn(x, ctx):
            scale = ctx.get_dim("scale", 2.0)
            return x * scale

        realiser = Realiser()
        lambda_spec = Lambda(fn=my_fn, name="scaler")
        module = realiser.realise(lambda_spec)

        assert hasattr(module, "fn")
        assert module._name == "scaler"

        # Test it works
        x = torch.ones(2, 3)
        y = module(x)
        assert torch.allclose(y, x * 2.0)

    def test_sequential_realisation(self):
        """Test Sequential realisation."""

        @register(SimpleSpec)
        def realise_simple(spec, _context):
            _context.set_dim("test", spec.size)
            return nn.Linear(10, spec.size)

        try:
            seq_spec = seq(
                SimpleSpec(size=20),
                SimpleSpec(size=30),
                SimpleSpec(size=40),
            )

            realiser = Realiser()
            module = realiser.realise(seq_spec)

            assert isinstance(module, nn.Sequential)
            assert len(module) == 3
            assert all(isinstance(m, nn.Linear) for m in module)
        finally:
            SpecMeta._realisers.pop(SimpleSpec, None)

    def test_parallel_realisation(self):
        """Test Parallel realisation."""

        @register(SimpleSpec)
        def realise_simple(spec, _context):
            return nn.Linear(10, spec.size)

        try:
            par_spec = parallel(
                SimpleSpec(size=20),
                SimpleSpec(size=30),
                merge="add",
            )

            realiser = Realiser()
            module = realiser.realise(par_spec)

            assert isinstance(module, ParallelModule)
            assert len(module.branches) == 2
            assert module.merge == "add"
        finally:
            SpecMeta._realisers.pop(SimpleSpec, None)

    def test_conditional_realisation(self):
        """Test Conditional realisation."""

        @register(SimpleSpec)
        def realise_simple(spec, _context):
            return nn.Linear(spec.size, spec.size)

        try:
            # Condition true
            cond_spec = cond(
                lambda ctx: ctx.get_dim("branch") == "a",
                SimpleSpec(size=10),
                SimpleSpec(size=20),
            )

            realiser = Realiser(Context(dimensions={"branch": "a"}))
            module = realiser.realise(cond_spec)

            assert isinstance(module, nn.Linear)
            assert module.in_features == 10  # True branch

            # Condition false
            realiser2 = Realiser(Context(dimensions={"branch": "b"}))
            module2 = realiser2.realise(cond_spec)

            assert isinstance(module2, nn.Linear)
            assert module2.in_features == 20  # False branch

            # No else branch
            cond_spec2 = cond(lambda _ctx: False, SimpleSpec(size=30))

            realiser3 = Realiser()
            module3 = realiser3.realise(cond_spec2)

            assert isinstance(module3, nn.Identity)  # Default
        finally:
            SpecMeta._realisers.pop(SimpleSpec, None)

    def test_residual_realisation(self):
        """Test Residual realisation."""

        @register(SimpleSpec)
        def realise_simple(spec, _context):
            return nn.Linear(spec.size, spec.size)

        try:
            res_spec = residual(SimpleSpec(size=64), scale=0.5)

            realiser = Realiser()
            module = realiser.realise(res_spec)

            assert isinstance(module, ResidualModule)
            assert module.merge == "add"
            assert module.scale == 0.5
            assert isinstance(module.inner, nn.Linear)
        finally:
            SpecMeta._realisers.pop(SimpleSpec, None)

    def test_loop_realisation(self):
        """Test Loop realisation."""

        @register(SimpleSpec)
        def realise_simple(spec, _context):
            return nn.Linear(spec.size, spec.size)

        try:
            # Dynamic loop
            loop_spec = loop(SimpleSpec(size=32), times=4)

            realiser = Realiser()
            module = realiser.realise(loop_spec)

            assert isinstance(module, LoopModule)
            assert module.times == 4

            # Unrolled loop with shared weights
            loop_spec2 = loop(SimpleSpec(size=32), times=3, unroll=True)
            module2 = realiser.realise(loop_spec2)

            assert isinstance(module2, nn.Sequential)
            assert len(module2) == 3
            # All should be same instance (shared weights)
            assert module2[0] is module2[1] is module2[2]

            # Unrolled without shared weights
            loop_spec3 = loop(
                SimpleSpec(size=32),
                times=3,
                unroll=True,
                share_weights=False,
            )
            module3 = realiser.realise(loop_spec3)

            assert isinstance(module3, nn.Sequential)
            assert len(module3) == 3
            # All should be different instances
            assert module3[0] is not module3[1]
            assert module3[1] is not module3[2]
        finally:
            SpecMeta._realisers.pop(SimpleSpec, None)

    def test_switch_realisation(self):
        """Test Switch realisation."""

        @register(SimpleSpec)
        def realise_simple(spec, _context):
            return nn.Linear(spec.size, spec.size)

        try:
            switch_spec = switch(
                "mode",
                {"small": SimpleSpec(size=10), "large": SimpleSpec(size=100)},
                default=SimpleSpec(size=50),
            )

            # Test each case
            ctx_small = Context(metadata={"mode": "small"})
            realiser1 = Realiser(ctx_small)
            module1 = realiser1.realise(switch_spec)
            assert module1.in_features == 10

            ctx_large = Context(metadata={"mode": "large"})
            realiser2 = Realiser(ctx_large)
            module2 = realiser2.realise(switch_spec)
            assert module2.in_features == 100

            ctx_other = Context(metadata={"mode": "medium"})
            realiser3 = Realiser(ctx_other)
            module3 = realiser3.realise(switch_spec)
            assert module3.in_features == 50  # Default
        finally:
            SpecMeta._realisers.pop(SimpleSpec, None)

    def test_graph_realisation(self):
        """Test Graph realisation."""

        @register(SimpleSpec)
        def realise_simple(spec, _context):
            return nn.Linear(spec.size, spec.size)

        try:
            g = (
                graph()
                .add_node("a", SimpleSpec(size=10))
                .add_node("b", SimpleSpec(size=20))
                .add_node("c", SimpleSpec(size=30))
                .add_edge("a", "b")
                .add_edge("b", "c")
            )

            graph_spec = Graph(
                nodes=g.nodes,
                edges=g.edges,
                inputs=["a"],
                outputs=["c"],
            )

            realiser = Realiser()
            module = realiser.realise(graph_spec)

            assert isinstance(module, GraphModule)
            assert len(module.nodes) == 3
            assert all(isinstance(m, nn.Linear) for m in module.nodes.values())
        finally:
            SpecMeta._realisers.pop(SimpleSpec, None)


class TestOptimization:
    """Test spec optimization."""

    def test_identity_removal(self):
        """Test identity nodes are removed."""
        spec = seq(
            Identity(),
            SimpleSpec(size=10),
            Identity(),
            SimpleSpec(size=20),
            Identity(),
        )

        realiser = Realiser()
        optimized = realiser._optimize_spec(spec)

        assert isinstance(optimized, Sequential)
        assert len(optimized) == 2
        assert all(isinstance(p, SimpleSpec) for p in optimized.parts)

    def test_sequential_flattening(self):
        """Test nested sequentials are flattened."""
        spec = seq(
            SimpleSpec(1),
            seq(SimpleSpec(2), SimpleSpec(3)),
            seq(seq(SimpleSpec(4), SimpleSpec(5))),
        )

        realiser = Realiser()
        optimized = realiser._optimize_spec(spec)

        assert isinstance(optimized, Sequential)
        assert len(optimized) == 5
        assert [p.size for p in optimized.parts] == [1, 2, 3, 4, 5]

    def test_optimization_preserves_non_identity(self):
        """Test optimization preserves important nodes."""
        spec = seq(SimpleSpec(1), LinearSpec(output_dim=100), SimpleSpec(2))

        realiser = Realiser()
        optimized = realiser._optimize_spec(spec)

        assert len(optimized) == 3
        assert isinstance(optimized.parts[1], LinearSpec)

    def test_optimization_disabled(self):
        """Test optimization can be disabled."""
        configure_realisation(optimizations=False)

        try:
            spec = seq(Identity(), SimpleSpec(), Identity())

            @register(SimpleSpec)
            def realise_simple(_spec, _context):
                return nn.Identity()

            try:
                realiser = Realiser()
                # Can't easily test optimization didn't happen without
                # accessing internals, but at least test it works
                module = realiser.realise(spec)
                assert isinstance(module, nn.Sequential)
            finally:
                SpecMeta._realisers.pop(SimpleSpec, None)
        finally:
            configure_realisation(optimizations=True)


class TestPublicAPI:
    """Test public API functions."""

    def test_realise_function(self):
        """Test main realise function."""

        @register(LinearSpec)
        def realise_linear(spec, context):
            input_dim = context.get_dim("input_dim")
            return nn.Linear(input_dim, spec.output_dim)

        try:
            # With keyword arguments
            module = realise(LinearSpec(output_dim=128), input_dim=64)
            assert isinstance(module, nn.Linear)
            assert module.in_features == 64
            assert module.out_features == 128

            # With explicit context
            ctx = Context(dimensions={"input_dim": 32})
            module2 = realise(LinearSpec(output_dim=256), ctx)
            assert module2.in_features == 32
            assert module2.out_features == 256

            # Context updates
            module3 = realise(
                LinearSpec(output_dim=512),
                ctx,
                input_dim=16,  # Override context
            )
            assert module3.in_features == 16
        finally:
            SpecMeta._realisers.pop(LinearSpec, None)

    def test_realise_validation(self):
        """Test realise validates specs."""
        spec = LinearSpec()  # Requires input_dim

        with pytest.raises(ValidationError) as exc_info:
            realise(spec)  # No input_dim provided

        assert "validation failed" in str(exc_info.value)

    def test_realise_non_strict_mode(self):
        """Test non-strict mode skips validation."""
        configure_realisation(strict=False)

        try:

            @register(LinearSpec)
            def realise_linear(spec, context):
                # Provide default
                input_dim = context.get_dim("input_dim") or 10
                return nn.Linear(input_dim, spec.output_dim)

            try:
                # Would fail validation but should work
                module = realise(LinearSpec())
                assert isinstance(module, nn.Linear)
                assert module.in_features == 10  # Default
            finally:
                SpecMeta._realisers.pop(LinearSpec, None)
        finally:
            configure_realisation(strict=True)

    def test_register_decorator(self):
        """Test register decorator."""

        # Create a unique spec class for this test to avoid interference
        @dataclass(frozen=True)
        class UniqueTestSpec(Spec):
            """Unique spec for testing registration."""

            size: int = param(default=64)  # noqa: RUF009

        # Clear any potential cached modules
        from energy_transformer.spec.realise import _config

        _config.cache.clear()

        @register(UniqueTestSpec)
        def my_realiser(spec, _context):
            return nn.Conv2d(3, spec.size, 3)

        try:
            # Check it's registered
            assert SpecMeta.get_realiser(UniqueTestSpec) is my_realiser

            # Check it works
            module = realise(UniqueTestSpec(size=64))
            assert isinstance(module, nn.Conv2d)
            assert module.out_channels == 64
        finally:
            SpecMeta._realisers.pop(UniqueTestSpec, None)

    def test_register_typed_decorator(self):
        """Test register_typed decorator."""

        @register_typed
        def realise_simple_spec(
            spec: SimpleSpec,
            _context: Context,
        ) -> nn.Module:
            return nn.Linear(spec.size, spec.size)

        try:
            # Check it's registered
            assert SpecMeta.get_realiser(SimpleSpec) is realise_simple_spec

            module = realise(SimpleSpec(size=32))
            assert isinstance(module, nn.Linear)
        finally:
            SpecMeta._realisers.pop(SimpleSpec, None)

    def test_visualize(self):
        """Test spec visualization."""
        spec = seq(
            SimpleSpec(size=10),
            parallel(SimpleSpec(size=20), SimpleSpec(size=30)),
            SimpleSpec(size=40),
        )

        dot = visualize(spec)

        assert "digraph" in dot
        assert "Sequential" in dot
        assert "SimpleSpec" in dot
        assert "Parallel" in dot
        assert "->" in dot  # Has edges

    def test_optimize_spec_function(self):
        """Test public optimization function."""
        spec = seq(Identity(), seq(SimpleSpec(1), SimpleSpec(2)), Identity())

        optimized = optimize_spec(spec)

        assert isinstance(optimized, Sequential)
        assert len(optimized) == 2
        assert all(isinstance(p, SimpleSpec) for p in optimized.parts)

    @pytest.mark.skipif(
        not pytest.importorskip("yaml", reason="PyYAML not installed"),
        reason="PyYAML required",
    )
    def test_yaml_serialization(self):
        """Test YAML serialization."""
        spec = seq(
            SimpleSpec(size=42),
            parallel(SimpleSpec(size=10), SimpleSpec(size=20), merge="add"),
        )

        # To YAML
        yaml_str = to_yaml(spec)
        assert "_type: Sequential" in yaml_str
        assert "size: 42" in yaml_str
        assert "merge: add" in yaml_str

        # From YAML
        spec2 = from_yaml(yaml_str)
        assert isinstance(spec2, Sequential)
        assert len(spec2.parts) == 2
        assert spec2.parts[0].size == 42
        assert spec2.parts[1].merge == "add"


class TestModuleImplementations:
    """Test PyTorch module implementations."""

    def test_parallel_module_merges(self):
        """Test ParallelModule merge strategies."""
        branches = [nn.Identity(), nn.Identity()]
        x = torch.randn(2, 10)

        # Concat
        pm = ParallelModule(branches, merge="concat")
        y = pm(x)
        assert y.shape == (2, 20)

        # Add
        pm = ParallelModule(branches, merge="add")
        y = pm(x)
        assert torch.allclose(y, x * 2)

        # Multiply
        pm = ParallelModule(branches, merge="multiply")
        y = pm(x)
        assert torch.allclose(y, x * x)

        # Mean
        pm = ParallelModule(branches, merge="mean")
        y = pm(x)
        assert torch.allclose(y, x)

        # Max
        pm = ParallelModule(branches, merge="max")
        y = pm(x)
        assert torch.allclose(y, x)

        # Weighted add
        pm = ParallelModule(branches, merge="add", weights=(0.3, 0.7))
        y = pm(torch.ones(2, 5))
        assert torch.allclose(y, torch.ones(2, 5))

        # Unknown merge
        pm = ParallelModule(branches, merge="unknown")
        with pytest.raises(ValueError, match="Unknown merge"):
            pm(x)

    def test_residual_module(self):
        """Test ResidualModule functionality."""
        inner = nn.Linear(10, 10)
        x = torch.randn(2, 10)

        # Add merge
        rm = ResidualModule(inner, merge="add", scale=0.5)
        y = rm(x)
        expected = x + 0.5 * inner(x)
        assert y.shape == expected.shape

        # Concat merge
        rm = ResidualModule(inner, merge="concat")
        y = rm(x)
        assert y.shape == (2, 20)

        # Gate merge (simplified)
        rm = ResidualModule(inner, merge="gate")
        y = rm(x)
        assert y.shape == x.shape

        # Unknown merge
        rm = ResidualModule(inner, merge="unknown")
        with pytest.raises(ValueError, match="Unknown merge"):
            rm(x)

    def test_loop_module(self):
        """Test LoopModule functionality."""
        body = nn.Linear(10, 10)
        lm = LoopModule(body, times=3)

        x = torch.randn(2, 10)
        y = lm(x)

        # Should apply body 3 times
        expected = x
        for _ in range(3):
            expected = body(expected)

        assert y.shape == expected.shape

    def test_graph_module(self):
        """Test GraphModule functionality."""
        # Simple linear graph: a -> b -> c
        nodes = {
            "a": nn.Linear(10, 20),
            "b": nn.Linear(20, 30),
            "c": nn.Linear(30, 40),
        }
        edges = [("input", "a"), ("a", "b"), ("b", "c")]

        gm = GraphModule(nodes, edges, inputs=["input"], outputs=["c"])

        x = torch.randn(2, 10)
        # Note: Current implementation is simplified
        # In real implementation would need proper routing
        y = gm(x)
        assert isinstance(y, torch.Tensor)

    def test_graph_module_errors(self):
        """GraphModule raises helpful errors on invalid graphs."""
        nodes = {"a": nn.Identity(), "b": nn.Identity()}
        gm_cycle = GraphModule(
            nodes,
            [("a", "b"), ("b", "a")],
            inputs=["x"],
            outputs=["a"],
        )
        with pytest.raises(RuntimeError, match="cycles"):
            gm_cycle(torch.randn(1, 1))

        gm_missing = GraphModule(
            nodes, [("a", "b")], inputs=["x"], outputs=["b"]
        )
        with pytest.raises(RuntimeError, match="no inputs"):
            gm_missing(torch.randn(1, 1))

    def test_apply_edge_transform_variants(self):
        """``_apply_edge_transform`` handles known and unknown transforms."""
        gm = GraphModule({}, [], [], [])
        t = torch.ones(1)
        assert torch.allclose(
            gm._apply_edge_transform(t, "relu"), torch.relu(t)
        )
        assert torch.allclose(
            gm._apply_edge_transform(t, "sigmoid"), torch.sigmoid(t)
        )
        with pytest.raises(ValueError, match="unknown"):
            gm._apply_edge_transform(t, "unknown")


class TestAutoImport:
    """Test automatic import functionality."""

    @patch("importlib.import_module")
    def test_auto_import_success(self, mock_import):
        """Test successful auto import."""
        # Create mock module
        mock_module = MagicMock()
        mock_layer_class = MagicMock(return_value=nn.Linear(10, 10))
        mock_module.TestLayer = mock_layer_class
        mock_import.return_value = mock_module

        # Create spec that matches import pattern
        @dataclass(frozen=True)
        class TestLayerSpec(Spec):
            size: int = param(default=10)  # noqa: RUF009

        # Would need to modify realiser to test properly
        # This is more of an integration test

    def test_auto_import_disabled(self):
        """Test auto import can be disabled."""

        # Create a spec that definitely won't have a realiser
        @dataclass(frozen=True)
        class UnregisteredSpec(Spec):
            """Spec with no realiser."""

            value: int = param(default=42)  # noqa: RUF009

        # Ensure it's not registered
        SpecMeta._realisers.pop(UnregisteredSpec, None)

        configure_realisation(auto_import=False)
        try:
            realiser = Realiser()
            with pytest.raises(RealisationError, match="No realiser"):
                realiser.realise(UnregisteredSpec())
        finally:
            configure_realisation(auto_import=True)


def test_validate_spec_tree_verbose_no_issues_again(capsys) -> None:
    """Verbose validation prints success and recurses."""
    import energy_transformer.spec as et_spec

    spec = et_spec.seq(et_spec.IdentitySpec(), et_spec.IdentitySpec())
    issues = et_spec.validate_spec_tree(spec, verbose=True)
    out = capsys.readouterr().out
    assert "No issues found" in out
    assert issues == []


# Module cleanup
def teardown_module():
    """Clean up after tests."""
    # Remove any test registrations
    for spec_cls in [SimpleSpec, LinearSpec, FailingSpec]:
        SpecMeta._realisers.pop(spec_cls, None)

    # Reset configuration to defaults
    configure_realisation(
        cache=ModuleCache(),
        plugins=[],
        strict=True,
        warnings=True,
        auto_import=True,
        optimizations=True,
        max_recursion=100,
    )
