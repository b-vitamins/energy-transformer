"""Test realisation system robustness."""

import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from energy_transformer.spec import (
    RealisationError,
    Spec,
    configure_realisation,
    loop,
    param,
    realise,
)
from energy_transformer.spec.primitives import SpecMeta
from energy_transformer.spec.realise import (
    ModuleCache,
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


class TestCacheStateRestoration:
    """Test cache state is properly restored on errors."""

    def test_cache_restored_after_exception(self):
        configure_realisation(cache=ModuleCache(enabled=True))
        assert _get_config().cache.enabled

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
                raise ValueError("Intentional failure")
            return nn.Identity()

        loop_spec = loop(
            FailingSpec(fail_on_iteration=2),
            times=3,
            unroll=True,
            share_weights=False,
        )

        with pytest.raises(RealisationError):
            realise(loop_spec)

        assert _get_config().cache.enabled

        simple = SimpleSpec()
        realise(simple)
        realise(simple)
        assert _get_config().cache.hit_rate > 0
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
            raise RuntimeError("Inner failure")

        nested = OuterSpec(
            inner=loop(
                InnerFailSpec(),
                times=2,
                unroll=True,
                share_weights=False,
            ),
        )

        initial_state = _get_config().cache.enabled
        with pytest.raises(RealisationError):
            realise(nested)
        assert _get_config().cache.enabled == initial_state

    def test_multiple_cache_errors_handled(self):
        with patch.object(_get_config(), "cache") as mock_cache:
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
            _get_config,
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
                raise ValueError("Intentional failure")
            return nn.Identity()

        configure_realisation(cache=ModuleCache(enabled=True))
        initial_state = _get_config().cache.enabled
        assert initial_state

        with pytest.raises(
            RealisationError,
            match="Intentional failure",
        ):
            realise(
                loop(FailingSpec(), times=3, unroll=True, share_weights=False),
            )

        assert _get_config().cache.enabled == initial_state, (
            "Cache state was not restored!"
        )

        from energy_transformer.spec.primitives import SpecMeta

        SpecMeta._realisers.pop(FailingSpec, None)
