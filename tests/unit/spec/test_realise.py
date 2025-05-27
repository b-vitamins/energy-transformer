"""Tests for specification realisation."""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from energy_transformer.spec.combinators import (
    Identity,
    Lambda,
    parallel,
    seq,
)
from energy_transformer.spec.library import (
    CLSTokenSpec,
    ETSpec,
    LayerNormSpec,
    PatchEmbedSpec,
)
from energy_transformer.spec.primitives import Context, SpecMeta
from energy_transformer.spec.realise import (
    ModuleCache,
    ParallelModule,
    RealisationError,
    Realiser,
    configure_realisation,
    from_yaml,
    realise,
    register,
    register_typed,
    to_yaml,
    visualize,
)


class TestModuleCache:
    """Test module caching functionality."""

    def test_basic_caching(self):
        cache = ModuleCache(max_size=2)

        spec = LayerNormSpec()
        ctx = Context(dimensions={"embed_dim": 768})
        module = nn.LayerNorm(768)

        # Cache miss
        assert cache.get(spec, ctx) is None
        assert cache.hit_rate == 0.0

        # Put and hit
        cache.put(spec, ctx, module)
        assert cache.get(spec, ctx) is module
        assert cache.hit_rate == 0.5  # 1 hit, 1 miss

    def test_lru_eviction(self):
        cache = ModuleCache(max_size=2)

        specs = [
            LayerNormSpec(eps=1e-5),
            LayerNormSpec(eps=1e-6),
            LayerNormSpec(eps=1e-7),
        ]
        ctx = Context(dimensions={"embed_dim": 768})
        modules = [nn.LayerNorm(768, eps=s.eps) for s in specs]

        # Fill cache
        cache.put(specs[0], ctx, modules[0])
        cache.put(specs[1], ctx, modules[1])

        # Access first to make it more recent
        cache.get(specs[0], ctx)

        # Add third, should evict second
        cache.put(specs[2], ctx, modules[2])

        assert cache.get(specs[0], ctx) is modules[0]  # Still there
        assert cache.get(specs[1], ctx) is None  # Evicted
        assert cache.get(specs[2], ctx) is modules[2]  # New one

    def test_disabled_cache(self):
        cache = ModuleCache(enabled=False)

        spec = LayerNormSpec()
        ctx = Context()
        module = nn.Identity()

        cache.put(spec, ctx, module)
        assert cache.get(spec, ctx) is None


class TestRealisationError:
    """Test RealisationError functionality."""

    def test_error_formatting(self):
        spec = LayerNormSpec()
        ctx = Context()
        cause = ValueError("test cause")

        err = RealisationError(
            "Test error",
            spec=spec,
            context=ctx,
            cause=cause,
            suggestion="Try this instead",
        )

        str_repr = str(err)
        assert "Test error" in str_repr
        assert "LayerNormSpec" in str_repr
        assert "ValueError: test cause" in str_repr
        assert "Try this instead" in str_repr


class TestRealiserPlugin:
    """Test plugin system."""

    def test_plugin_interface(self):
        class TestPlugin:
            def can_realise(self, spec):
                return isinstance(spec, LayerNormSpec)

            def realise(self, spec, context):
                return nn.LayerNorm(context.get_dim("embed_dim") or 768)

        plugin = TestPlugin()

        assert plugin.can_realise(LayerNormSpec())
        assert not plugin.can_realise(CLSTokenSpec())

        module = plugin.realise(
            LayerNormSpec(), Context(dimensions={"embed_dim": 512})
        )
        assert isinstance(module, nn.LayerNorm)
        assert module.normalized_shape == (512,)


class TestConfiguration:
    """Test realisation configuration."""

    def test_configure_realisation(self):
        # Configure cache
        new_cache = ModuleCache(max_size=256)
        configure_realisation(cache=new_cache)

        # Configure other options
        configure_realisation(
            strict=False,
            warnings=False,
            auto_import=False,
            optimizations=False,
            max_recursion=50,
        )

        # Test invalid option
        with pytest.raises(ValueError, match="Unknown configuration"):
            configure_realisation(invalid_option=True)


class TestRealiser:
    """Test main Realiser class."""

    def test_basic_realisation(self):
        # Register a simple realiser
        @register(LayerNormSpec)
        def realise_ln(spec, context):
            dim = context.get_dim("embed_dim")
            if dim is None:
                raise RealisationError("Missing embed_dim")
            return nn.LayerNorm(dim, eps=spec.eps)

        realiser = Realiser(Context(dimensions={"embed_dim": 768}))
        module = realiser.realise(LayerNormSpec())

        assert isinstance(module, nn.LayerNorm)
        assert module.normalized_shape == (768,)
        assert module.eps == 1e-5

    def test_sequential_realisation(self):
        # Mock realisers
        @register(PatchEmbedSpec)
        def realise_patch(spec, context):
            # Simple mock that sets context
            context.set_dim("embed_dim", spec.embed_dim)
            context.set_dim("token_count", 4)
            return nn.Linear(spec.in_chans, spec.embed_dim)

        @register(CLSTokenSpec)
        def realise_cls(spec, context):
            if context.get_dim("embed_dim") is None:
                raise RealisationError("Missing embed_dim")
            context.set_dim("token_count", context.get_dim("token_count") + 1)
            return nn.Identity()

        @register(LayerNormSpec)
        def realise_ln(spec, context):
            dim = context.get_dim("embed_dim")
            if dim is None:
                raise RealisationError("Missing embed_dim")
            return nn.LayerNorm(dim)

        # Build sequence
        spec = seq(
            PatchEmbedSpec(img_size=32, patch_size=16, embed_dim=768),
            CLSTokenSpec(),
            LayerNormSpec(),
        )

        realiser = Realiser()
        module = realiser.realise(spec)

        assert isinstance(module, nn.Sequential)
        assert len(module) == 3
        assert isinstance(module[0], nn.Linear)
        assert isinstance(module[1], nn.Identity)
        assert isinstance(module[2], nn.LayerNorm)

    def test_parallel_realisation(self):
        # Register realiser
        @register(LayerNormSpec)
        def realise_ln(spec, context):
            return nn.LayerNorm(768)

        spec = parallel(LayerNormSpec(), LayerNormSpec(), merge="add")

        realiser = Realiser(Context(dimensions={"embed_dim": 768}))
        module = realiser.realise(spec)

        assert isinstance(module, ParallelModule)
        assert len(module.branches) == 2
        assert module.merge == "add"

    def test_builtin_realisers(self):
        realiser = Realiser()

        # Identity
        identity_module = realiser.realise(Identity())
        assert isinstance(identity_module, nn.Identity)

        # Lambda
        lambda_spec = Lambda(lambda x, ctx: x * 2, name="double")
        lambda_module = realiser.realise(lambda_spec)
        assert hasattr(lambda_module, "fn")

    def test_circular_dependency_detection(self):
        # This would require a more complex setup with actual circular deps
        pass

    def test_optimization(self):
        # Create sequence with identity nodes
        spec = seq(Identity(), LayerNormSpec(), Identity(), CLSTokenSpec())

        realiser = Realiser(Context(dimensions={"embed_dim": 768}))
        optimized = realiser._optimize_spec(spec)

        # Should remove identity nodes
        assert len(optimized.parts) == 2
        assert isinstance(optimized.parts[0], LayerNormSpec)
        assert isinstance(optimized.parts[1], CLSTokenSpec)


class TestPublicAPI:
    """Test public realisation API."""

    def test_realise_function(self):
        @register(LayerNormSpec)
        def realise_ln(spec, context):
            dim = context.get_dim("embed_dim")
            return nn.LayerNorm(dim or 768)

        # With context
        module = realise(LayerNormSpec(), embed_dim=512)
        assert isinstance(module, nn.LayerNorm)
        assert module.normalized_shape == (512,)

        # With explicit context
        ctx = Context(dimensions={"embed_dim": 256})
        module2 = realise(LayerNormSpec(), ctx)
        assert module2.normalized_shape == (256,)

    def test_register_decorator(self):
        @register(LayerNormSpec)
        def my_realiser(spec, context):
            return nn.LayerNorm(768)

        assert SpecMeta.get_realiser(LayerNormSpec) is my_realiser

    def test_register_typed(self):
        @register_typed
        def realise_layer_norm(
            spec: LayerNormSpec, context: Context
        ) -> nn.Module:
            return nn.LayerNorm(768)

        # Note: This would need proper type hint extraction in real implementation

    def test_visualize(self):
        spec = seq(PatchEmbedSpec(32, 16, 768), CLSTokenSpec())

        dot = visualize(spec)
        assert "digraph" in dot
        assert "PatchEmbedSpec" in dot
        assert "CLSTokenSpec" in dot
        assert "->" in dot  # Has edges

    def test_yaml_serialization(self):
        spec = LayerNormSpec(eps=1e-6)

        # To YAML
        yaml_str = to_yaml(spec)
        assert "_type: LayerNormSpec" in yaml_str
        assert "eps: 1" in yaml_str  # Contains eps value

        # From YAML
        spec2 = from_yaml(yaml_str)
        assert isinstance(spec2, LayerNormSpec)
        assert spec2.eps == 1e-6


class TestParallelModule:
    """Test ParallelModule implementation."""

    def test_merge_modes(self):
        branches = [nn.Identity(), nn.Identity()]

        # Concat
        pm_concat = ParallelModule(branches, merge="concat")
        x = torch.randn(2, 10)
        out = pm_concat(x)
        assert out.shape == (2, 20)  # Doubled last dim

        # Add
        pm_add = ParallelModule(branches, merge="add")
        out = pm_add(x)
        assert torch.allclose(out, x * 2)

        # Multiply
        pm_mul = ParallelModule(branches, merge="multiply")
        out = pm_mul(x)
        assert torch.allclose(out, x * x)

        # Mean
        pm_mean = ParallelModule(branches, merge="mean")
        out = pm_mean(x)
        assert torch.allclose(out, x)

        # Max
        pm_max = ParallelModule(branches, merge="max")
        out = pm_max(x)
        assert torch.allclose(out, x)

    def test_weighted_add(self):
        branches = [nn.Identity(), nn.Identity()]
        weights = (0.3, 0.7)

        pm = ParallelModule(branches, merge="add", weights=weights)
        x = torch.ones(2, 5)
        out = pm(x)

        assert torch.allclose(out, torch.ones_like(out))  # 0.3 + 0.7 = 1.0

    def test_unknown_merge_mode(self):
        pm = ParallelModule([nn.Identity()], merge="unknown")

        with pytest.raises(ValueError, match="Unknown merge mode"):
            pm(torch.randn(1, 1))


class TestAutoImport:
    """Test automatic module import."""

    @patch("importlib.import_module")
    def test_auto_import_success(self, mock_import):
        # Mock the import
        mock_module = Mock()
        mock_module.LayerNorm = nn.LayerNorm
        mock_import.return_value = mock_module

        # Enable auto import
        configure_realisation(auto_import=True)

        # Try to realise without registered realiser
        # This would trigger auto-import
        # (Would need more setup to test properly)

    def test_auto_import_failure(self):
        # Test graceful failure when import fails
        pass


# Clean up registered realisers after tests
def teardown_module():
    """Clean up test registrations."""
    # Remove test realisers
    for spec_cls in [LayerNormSpec, PatchEmbedSpec, CLSTokenSpec, ETSpec]:
        SpecMeta._realisers.pop(spec_cls, None)
