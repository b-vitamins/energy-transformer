"""Extra tests for primitives focusing on dimension resolution and validation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from energy_transformer.spec.primitives import (
    REQUIRED,
    Context,
    Dimension,
    Spec,
    ValidationError,
    param,
)


class TestDimensionResolve:
    """Tests for :class:`Dimension` formula evaluation."""

    def test_resolve_formula(self) -> None:
        ctx = Context(dimensions={"a": 2, "b": 8})
        dim = Dimension("d", formula="a * 2 + b / 4")
        assert dim.resolve(ctx) == 6

    def test_resolve_errors_return_none(self) -> None:
        ctx = Context(dimensions={"a": 1})
        # Division by zero
        dim = Dimension("d", formula="a / 0")
        assert dim.resolve(ctx) is None
        # Unknown variable
        dim2 = Dimension("d", formula="missing + 1")
        assert dim2.resolve(ctx) is None
        # Mismatched parentheses
        dim3 = Dimension("d", formula="(a + 2")
        assert dim3.resolve(ctx) is None


def test_required_parameter_validation() -> None:
    """Missing required parameters raise :class:`ValidationError`."""

    @dataclass(frozen=True)
    class ReqSpec(Spec):
        value: int = param()  # noqa: RUF009

    with pytest.raises(ValidationError):
        ReqSpec(value=REQUIRED)
