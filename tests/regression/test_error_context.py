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
            raise RuntimeError("Deep error")

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
