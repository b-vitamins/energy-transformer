"""Test realisation system robustness."""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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
pytestmark = pytest.mark.integration


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


