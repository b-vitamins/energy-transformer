import pytest
from unittest.mock import patch
from energy_transformer.spec.primitives import Context, Dimension

pytestmark = pytest.mark.security

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

    def test_verify_security_script_behavior(self):
        """Test the exact behavior from verify_security.py."""
        from energy_transformer.spec.primitives import Context, Dimension

        # This should NOT print anything or execute code
        ctx = Context(dimensions={"x": 10})
        evil = Dimension("evil", formula="print('HACKED!') or x")
        result = evil.resolve(ctx)

        # Evil formula should return None
        assert result is None, "Evil formula should fail to parse"

        # Good formula should work correctly
        good = Dimension("good", formula="x * 2")
        result = good.resolve(ctx)
        assert result == 20, "Good formula should evaluate correctly"

    def test_no_output_on_evil_formulas(self, capsys):
        """Ensure evil formulas produce no output (no code execution)."""
        from energy_transformer.spec.primitives import Context, Dimension

        ctx = Context(dimensions={"x": 10})
        evil_formulas = [
            "print('HACKED!')",
            "print('HACKED!') or x",
            "__import__('sys').stdout.write('HACKED')",
        ]

        for formula in evil_formulas:
            dim = Dimension("evil", formula=formula)
            result = dim.resolve(ctx)

            captured = capsys.readouterr()
            assert captured.out == "", f"Formula {formula!r} produced output!"
            assert captured.err == "", f"Formula {formula!r} produced errors!"
            assert result is None

    @patch("os.system")
    def test_no_actual_execution(self, mock_system):
        ctx = Context(dimensions={})
        dim = Dimension("test", formula="__import__('os').system('echo test')")
        result = dim.resolve(ctx)
        mock_system.assert_not_called()
        assert result is None
