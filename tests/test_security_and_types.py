"""Test security fixes and type safety.

This module verifies:
1. eval() exploit prevention
2. Type union crash fixes
3. Choice validation type safety
"""

import pytest
import sys
import os
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from energy_transformer.spec.primitives import Dimension, Context, ValidationError
from energy_transformer.spec.library import (
    validate_positive,
    validate_probability,
    validate_dimension,
)


class TestSecurityFixes:
    """Test that eval() exploits are prevented."""

    def test_dimension_blocks_code_execution(self):
        """Ensure malicious code cannot be executed through formulas."""
        ctx = Context(dimensions={"x": 10, "y": 20})

        dangerous_formulas = [
            "__import__('os').system('echo pwned')",
            "__import__('subprocess').call(['ls'])",
            "exec('print(1)')",
            "eval('1+1')",
            "__builtins__['eval']('1+1')",
            "globals()['__builtins__']['exec']('x=1')",
            "[i for i in range(10**10)]",
            "10**10**10",
            "x.__class__.__bases__[0].__subclasses__()",
            "().__class__.__bases__[0].__subclasses__()",
            "print(x)",
            "len([1,2,3])",
            "max(1,2,3)",
        ]

        for formula in dangerous_formulas:
            dim = Dimension("test", formula=formula)
            result = dim.resolve(ctx)
            assert result is None, f"Formula {formula!r} should fail to parse"

    def test_dimension_allows_safe_math(self):
        """Ensure legitimate mathematical formulas still work."""
        ctx = Context(dimensions={"width": 224, "patch": 16, "heads": 12})

        safe_formulas = [
            ("width / patch", 14),
            ("width * 2", 448),
            ("width + patch", 240),
            ("width - patch", 208),
            ("(width / patch) * heads", 168),
            ("width / patch / 2", 7),
            ("-patch", -16),
            ("patch * -1", -16),
        ]

        for formula, expected in safe_formulas:
            dim = Dimension("test", formula=formula)
            result = dim.resolve(ctx)
            assert result == expected

    def test_dimension_handles_missing_variables(self):
        ctx = Context(dimensions={"x": 10})
        dim = Dimension("test", formula="x + y")
        assert dim.resolve(ctx) is None

    def test_dimension_handles_none_values(self):
        ctx = Context(dimensions={"x": 10, "y": None})
        dim = Dimension("test", formula="x + y")
        assert dim.resolve(ctx) is None

    @patch('os.system')
    def test_no_actual_execution(self, mock_system):
        ctx = Context(dimensions={})
        dim = Dimension("test", formula="__import__('os').system('echo test')")
        result = dim.resolve(ctx)
        mock_system.assert_not_called()
        assert result is None


class TestTypeUnionFixes:
    """Test that isinstance() works correctly with type unions."""

    def test_validate_positive_with_numbers(self):
        assert validate_positive(5) is True
        assert validate_positive(0) is False
        assert validate_positive(-5) is False
        assert validate_positive(5.5) is True
        assert validate_positive(0.0) is False
        assert validate_positive(-5.5) is False
        assert validate_positive(float('inf')) is True
        assert validate_positive(float('-inf')) is False

    def test_validate_positive_with_tuples(self):
        assert validate_positive((1, 2, 3)) is True
        assert validate_positive((1.5, 2.5)) is True
        assert validate_positive((1, 2.5, 3)) is True
        assert validate_positive((1, 0, 3)) is False
        assert validate_positive((1, -2, 3)) is False
        assert validate_positive(()) is True

    def test_validate_positive_with_wrong_types(self):
        assert validate_positive("5") is False
        assert validate_positive([1, 2, 3]) is False
        assert validate_positive(None) is False
        assert validate_positive({"x": 5}) is False

    def test_validate_probability(self):
        assert validate_probability(0.0) is True
        assert validate_probability(0.5) is True
        assert validate_probability(1.0) is True
        assert validate_probability(-0.1) is False
        assert validate_probability(1.1) is False

    def test_validate_dimension(self):
        assert validate_dimension(1) is True
        assert validate_dimension(768) is True
        assert validate_dimension(65536) is True
        assert validate_dimension(0) is False
        assert validate_dimension(-1) is False
        assert validate_dimension(65537) is False


class TestChoiceValidation:
    """Test that choice validation includes type checking."""

    def test_choices_with_consistent_types(self):
        from energy_transformer.spec.primitives import param, Spec
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class TestSpec(Spec):
            mode: str = param(default="auto", choices=["auto", "manual", "hybrid"])
            size: int = param(default=1, choices=[1, 2, 4, 8])

        spec = TestSpec(mode="manual", size=4)
        assert spec.mode == "manual"
        assert spec.size == 4

    def test_choices_with_mixed_types_warns(self, caplog):
        from energy_transformer.spec.primitives import param, Spec
        from dataclasses import dataclass
        import logging

        logging.basicConfig(level=logging.WARNING)

        @dataclass(frozen=True)
        class BadSpec(Spec):
            value: str | None = param(default="a", choices=["a", "b", None])

        BadSpec(value=None)

    def test_choices_type_mismatch_error(self):
        from energy_transformer.spec.primitives import param, Spec, ValidationError
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class TestSpec(Spec):
            size: int = param(choices=[1, 2, 4, 8])

        with pytest.raises(ValidationError):
            TestSpec(size="2")


class TestIntegration:
    """Integration tests combining multiple fixes."""

    def test_complex_formula_with_validation(self):
        from energy_transformer.spec.primitives import Context, Dimension

        ctx = Context(dimensions={"embed_dim": 768, "num_heads": 12, "mlp_ratio": 4})

        dim1 = Dimension("head_dim", formula="embed_dim / num_heads")
        assert dim1.resolve(ctx) == 64

        dim2 = Dimension("mlp_hidden", formula="embed_dim * mlp_ratio")
        assert dim2.resolve(ctx) == 3072

        dim3 = Dimension("exploit", formula="exec('x=1') or embed_dim")
        assert dim3.resolve(ctx) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
