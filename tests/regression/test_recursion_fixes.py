"""Test realisation system robustness."""

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from energy_transformer.spec import (
    Context,
    RealisationError,
    Sequential,
    Spec,
    configure_realisation,
    param,
    realise,
    seq,
)
from energy_transformer.spec.primitives import SpecMeta
from energy_transformer.spec.realise import (
    Realiser,
    _get_config,
    register,
)

pytestmark = pytest.mark.regression


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

        assert _get_config().cache.hit_rate > 0

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
        from energy_transformer.spec.realise import _get_config

        configure_realisation(max_recursion=5)
        deep_model_spec = seq(*[IdentitySpec() for _ in range(20)])

        model1 = realise(deep_model_spec)
        assert model1 is not None, "First realisation should succeed"

        initial_hits = _get_config().cache._hit_count
        model2 = realise(deep_model_spec)
        assert model2 is not None, "Second realisation should succeed"

        final_hits = _get_config().cache._hit_count
        assert final_hits > initial_hits, "Cache should have been used"
        assert _get_config().cache.hit_rate > 0, "Hit rate should be positive"
