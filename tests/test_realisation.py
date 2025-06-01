"""Test realisation system robustness."""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from energy_transformer.spec import (
    Context,
    RealisationError,
    Sequential,
    Spec,
    configure_realisation,
    loop,
    param,
    realise,
    seq,
)
from energy_transformer.spec.primitives import SpecMeta
from energy_transformer.spec.realise import (
    ModuleCache,
    Realiser,
    _config,
    register,
)


@dataclass(frozen=True)
class SimpleSpec(Spec):
    """Simple spec for testing."""

    value: int = param(default=1)  # noqa: RUF009


@dataclass(frozen=True)
class DeepSpec(Spec):
    """Spec that creates deep nesting."""

    depth: int = param(default=10)  # noqa: RUF009

    def children(self) -> list[Spec]:
        if self.depth > 0:
            return [DeepSpec(depth=self.depth - 1)]
        return []


class TestRecursionDepth:
    """Test recursion depth handling with caching."""

    def test_cache_check_before_recursion_limit(self):
        configure_realisation(max_recursion=5)

        @register(SimpleSpec)
        def realise_simple(_spec, _context):
            if _spec.value == 0:
                return nn.Identity()
            return nn.Linear(_spec.value, _spec.value)

        specs = [SimpleSpec(value=i) for i in range(20)]
        deep_spec = seq(*specs)

        ctx = Context()
        model1 = realise(deep_spec, context=ctx)
        assert model1 is not None

        model2 = realise(deep_spec, context=ctx)
        assert model2 is not None

        assert _config.cache.hit_rate > 0

    def test_recursion_error_includes_context(self):
        configure_realisation(max_recursion=3, optimizations=False)

        @register(SimpleSpec)
        def realise_simple(spec, _context):
            return nn.Linear(spec.value, spec.value)

        spec = SimpleSpec(1)
        for _ in range(10):
            spec = Sequential(parts=(spec,))

        with pytest.raises(RealisationError) as exc_info:
            realise(spec)
        error = str(exc_info.value)
        assert "Maximum recursion depth (3) exceeded" in error
        assert "Current stack depth:" in error
        assert "Consider:" in error
        SpecMeta._realisers.pop(SimpleSpec, None)

    def test_circular_dependency_detection(self):
        @dataclass(frozen=True)
        class CircularSpec(Spec):
            name: str = param()  # noqa: RUF009

            def children(self) -> list[Spec]:
                if self.name == "A":
                    return [CircularSpec(name="B")]
                if self.name == "B":
                    return [CircularSpec(name="A")]
                return []

        realiser = Realiser()

        @register(CircularSpec)
        def realise_circular(spec, _context):
            return (
                realiser.realise(spec.children()[0])
                if spec.children()
                else nn.Identity()
            )

        configure_realisation(strict=False)

        circular = CircularSpec(name="A")
        with pytest.raises(RealisationError) as exc_info:
            realiser.realise(circular)
        assert "Circular dependency" in str(exc_info.value)
        SpecMeta._realisers.pop(CircularSpec, None)

    def test_verify_recursion_fix_script_behavior(self):
        """Test exact behavior from verify_recursion_fix.py."""
        from energy_transformer.spec import configure_realisation, realise, seq
        from energy_transformer.spec.library import IdentitySpec
        from energy_transformer.spec.realise import _config

        configure_realisation(max_recursion=5)
        deep_model_spec = seq(*[IdentitySpec() for _ in range(20)])

        model1 = realise(deep_model_spec)
        assert model1 is not None, "First realisation should succeed"

        initial_hits = _config.cache._hit_count
        model2 = realise(deep_model_spec)
        assert model2 is not None, "Second realisation should succeed"

        final_hits = _config.cache._hit_count
        assert final_hits > initial_hits, "Cache should have been used"
        assert _config.cache.hit_rate > 0, "Hit rate should be positive"


class TestCacheStateRestoration:
    """Test cache state is properly restored on errors."""

    def test_cache_restored_after_exception(self):
        configure_realisation(cache=ModuleCache(enabled=True))
        assert _config.cache.enabled

        @register(SimpleSpec)
        def realise_simple(_spec, _context):
            return nn.Identity()

        @dataclass(frozen=True)
        class FailingSpec(Spec):
            fail_on_iteration: int = param(default=2)  # noqa: RUF009

        attempt_count = 0

        @register(FailingSpec)
        def realise_failing(spec, _context):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == spec.fail_on_iteration:
                raise ValueError("Intentional failure")  # noqa: TRY003
            return nn.Identity()

        loop_spec = loop(
            FailingSpec(fail_on_iteration=2),
            times=3,
            unroll=True,
            share_weights=False,
        )

        with pytest.raises(RealisationError):
            realise(loop_spec)

        assert _config.cache.enabled

        simple = SimpleSpec()
        realise(simple)
        realise(simple)
        assert _config.cache.hit_rate > 0
        SpecMeta._realisers.pop(SimpleSpec, None)

    def test_cache_restored_with_nested_errors(self):
        configure_realisation(cache=ModuleCache(enabled=True))

        @dataclass(frozen=True)
        class OuterSpec(Spec):
            inner: Spec = param()  # noqa: RUF009

            def children(self) -> list[Spec]:
                return [self.inner]

        @dataclass(frozen=True)
        class InnerFailSpec(Spec):
            pass

        @register(InnerFailSpec)
        def realise_inner_fail(_spec, _context):
            raise RuntimeError("Inner failure")  # noqa: TRY003

        nested = OuterSpec(
            inner=loop(
                InnerFailSpec(),
                times=2,
                unroll=True,
                share_weights=False,
            ),
        )

        initial_state = _config.cache.enabled
        with pytest.raises(RealisationError):
            realise(nested)
        assert _config.cache.enabled == initial_state

    def test_multiple_cache_errors_handled(self):
        with patch.object(_config, "cache") as mock_cache:
            type(mock_cache).enabled = property(
                lambda _self: True,
                lambda _self, _value: (_ for _ in ()).throw(
                    RuntimeError("Cache restore failed"),
                ),
            )

            @dataclass(frozen=True)
            class BadSpec(Spec):
                pass

            realiser = Realiser()
            with pytest.raises(RuntimeError) as exc_info:
                realiser._realise_unrolled_independent(
                    loop(BadSpec(), times=1),
                    times=1,
                )
            assert "Multiple errors" in str(exc_info.value)

    def test_verify_cache_restoration_script_behavior(self):
        """Test exact behavior from verify_cache_restoration.py."""
        from dataclasses import dataclass

        from torch import nn

        from energy_transformer.spec import (
            Spec,
            configure_realisation,
            loop,
            realise,
        )
        from energy_transformer.spec.realise import (
            ModuleCache,
            _config,
            register,
        )

        @dataclass(frozen=True)
        class FailingSpec(Spec):
            pass

        call_count = 0

        @register(FailingSpec)
        def realise_failing(_spec, _context):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Intentional failure")  # noqa: TRY003
            return nn.Identity()

        configure_realisation(cache=ModuleCache(enabled=True))
        initial_state = _config.cache.enabled
        assert initial_state

        with pytest.raises(
            RealisationError,
            match="Intentional failure",
        ):
            realise(
                loop(FailingSpec(), times=3, unroll=True, share_weights=False),
            )

        assert _config.cache.enabled == initial_state, (
            "Cache state was not restored!"
        )

        from energy_transformer.spec.primitives import SpecMeta

        SpecMeta._realisers.pop(FailingSpec, None)


class TestCacheKeyGeneration:
    """Test deep cache key generation."""

    def test_nested_dict_ordering(self):
        cache = ModuleCache()
        spec = SimpleSpec()
        ctx1 = Context(
            dimensions={"a": 1, "b": 2},
            metadata={
                "config": {"y": 2, "x": 1},
                "nested": {"inner": {"b": 2, "a": 1}},
            },
        )
        ctx2 = Context(
            dimensions={"b": 2, "a": 1},
            metadata={
                "config": {"x": 1, "y": 2},
                "nested": {"inner": {"a": 1, "b": 2}},
            },
        )
        key1 = cache._make_key(spec, ctx1)
        key2 = cache._make_key(spec, ctx2)
        assert key1 == key2

    def test_type_preservation_in_keys(self):
        cache = ModuleCache()
        spec = SimpleSpec()
        ctx_int = Context(dimensions={"value": 1})
        ctx_float = Context(dimensions={"value": 1.0})
        key_int = cache._make_key(spec, ctx_int)
        key_float = cache._make_key(spec, ctx_float)
        assert key_int != key_float

    def test_cycle_detection_in_cache_keys(self):
        cache = ModuleCache()
        spec = SimpleSpec()
        circular_dict = {"a": 1}
        circular_dict["self"] = circular_dict
        ctx = Context(metadata={"circular": circular_dict})
        key = cache._make_key(spec, ctx)
        assert key is not None
        assert "<cycle:" in str(key)

    def test_cache_invalidation_on_version_change(self):
        cache = ModuleCache()
        cache.version = 1
        spec = SimpleSpec()
        ctx = Context()
        key1 = cache._make_key(spec, ctx)
        cache.version = 2
        key2 = cache._make_key(spec, ctx)
        assert key1 != key2

    def test_complex_nested_structures(self):
        cache = ModuleCache()

        @dataclass(frozen=True)
        class ComplexSpec(Spec):
            data: dict = param(default_factory=dict)  # noqa: RUF009

        spec = ComplexSpec(
            data={
                "lists": [[1, 2], [3, 4]],
                "sets": {1, 2, 3},
                "mixed": {"a": [{"x": 1}, {"y": 2}], "b": {(1, 2), (3, 4)}},
            },
        )
        ctx = Context()
        key = cache._make_key(spec, ctx)
        assert key is not None
        spec2 = ComplexSpec(
            data={
                "mixed": {"b": {(3, 4), (1, 2)}, "a": [{"x": 1}, {"y": 2}]},
                "sets": {3, 2, 1},
                "lists": [[1, 2], [3, 4]],
            },
        )
        key2 = cache._make_key(spec2, ctx)
        assert key == key2

    def test_verify_cache_keys_script_behavior(self):
        """Test exact behavior from verify_cache_keys.py."""
        from energy_transformer.spec import Context, realise
        from energy_transformer.spec.library import IdentitySpec
        from energy_transformer.spec.realise import _config

        ctx1 = Context(
            dimensions={"a": 1, "b": 2},
            metadata={"nested": {"x": 1, "y": 2}},
        )
        ctx2 = Context(
            dimensions={"b": 2, "a": 1},
            metadata={"nested": {"y": 2, "x": 1}},
        )

        spec = IdentitySpec()

        _config.cache.clear()
        _config.cache._hit_count = 0
        _config.cache._miss_count = 0

        model1 = realise(spec, context=ctx1)
        assert _config.cache._hit_count == 0
        assert _config.cache._miss_count > 0

        miss_after_first = _config.cache._miss_count

        model2 = realise(spec, context=ctx2)
        assert _config.cache._hit_count >= 1
        assert _config.cache._miss_count == miss_after_first
        assert model1 is model2


class TestAutoImportLogging:
    """Test auto-import with proper logging."""

    def test_import_failure_logged(self, caplog):
        configure_realisation(warnings=True)
        caplog.set_level(
            logging.DEBUG,
            logger="energy_transformer.spec.realise",
        )

        @dataclass(frozen=True)
        class UnknownSpec(Spec):
            pass

        realiser = Realiser()
        result = realiser._try_auto_import(UnknownSpec())
        assert result is None
        assert len(caplog.records) > 0
        assert "No auto-import mapping" in caplog.text

    def test_missing_module_logged(self, caplog):
        configure_realisation(warnings=True)
        caplog.set_level(
            logging.WARNING,
            logger="energy_transformer.spec.realise",
        )
        with patch.dict(
            "energy_transformer.spec.realise.module_mappings",
            {"TestSpec": ("non_existent_module", "TestClass")},
        ):

            @dataclass(frozen=True)
            class TestSpec(Spec):
                pass

            realiser = Realiser()
            result = realiser._try_auto_import(TestSpec())
            assert result is None
            assert "Failed to import non_existent_module" in caplog.text
            assert "pip install" in caplog.text

    def test_missing_class_logged(self, caplog):
        configure_realisation(warnings=True)
        caplog.set_level(
            logging.WARNING,
            logger="energy_transformer.spec.realise",
        )
        with patch.dict(
            "energy_transformer.spec.realise.module_mappings",
            {"TestSpec": ("torch.nn", "NonExistentClass")},
        ):

            @dataclass(frozen=True)
            class TestSpec(Spec):
                pass

            realiser = Realiser()
            result = realiser._try_auto_import(TestSpec())
            assert result is None
            assert "has no attribute NonExistentClass" in caplog.text
            assert "Available attributes:" in caplog.text

    def test_instantiation_failure_logged(self, caplog):
        configure_realisation(warnings=True)
        caplog.set_level(
            logging.WARNING,
            logger="energy_transformer.spec.realise",
        )
        with patch.dict(
            "energy_transformer.spec.realise.module_mappings",
            {"TestSpec": ("torch.nn", "Linear")},
        ):

            @dataclass(frozen=True)
            class TestSpec(Spec):
                pass

            realiser = Realiser()
            result = realiser._try_auto_import(TestSpec())
            assert result is None
            assert "Failed to instantiate Linear" in caplog.text
            assert (
                "missing" in caplog.text.lower()
                or "required" in caplog.text.lower()
            )

    def test_successful_import_logged(self, caplog):
        configure_realisation(warnings=True)
        caplog.set_level(logging.INFO, logger="energy_transformer.spec.realise")
        with patch.dict(
            "energy_transformer.spec.realise.module_mappings",
            {"TestSpec": ("torch.nn", "Identity")},
        ):

            @dataclass(frozen=True)
            class TestSpec(Spec):
                pass

            realiser = Realiser()
            result = realiser._try_auto_import(TestSpec())
            assert result is not None
            assert isinstance(result, nn.Identity)
            assert "Successfully auto-imported" in caplog.text


class TestRealisationErrorContext:
    """Test error context enhancement."""

    def test_error_context_preserved(self):
        @dataclass(frozen=True)
        class FailSpec(Spec):
            message: str = param(default="Test failure")  # noqa: RUF009

        @register(FailSpec)
        def realise_fail(spec, _context):
            raise ValueError(spec.message)

        spec = FailSpec(message="Original error message")
        with pytest.raises(RealisationError) as exc_info:
            realise(spec)
        error = exc_info.value
        assert error.spec == spec
        assert error.cause is not None
        assert isinstance(error.cause, ValueError)
        assert "Original error message" in str(error)
        assert "ValueError" in str(error)

    def test_nested_error_context(self):
        @dataclass(frozen=True)
        class Level1Spec(Spec):
            child: Spec = param()  # noqa: RUF009

            def children(self) -> list[Spec]:
                return [self.child]

        @dataclass(frozen=True)
        class Level2Spec(Spec):
            pass

        @register(Level2Spec)
        def realise_level2(_spec, _context):
            raise RuntimeError("Deep error")  # noqa: TRY003

        @register(Level1Spec)
        def realise_level1(spec, _context):
            return realise(spec.child)

        nested = Level1Spec(child=Level2Spec())
        with pytest.raises(RealisationError) as exc_info:
            realise(nested)
        error = str(exc_info.value)
        assert "Deep error" in error
        assert "RuntimeError" in error
        SpecMeta._realisers.pop(Level1Spec, None)
        SpecMeta._realisers.pop(Level2Spec, None)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
