import pytest

from energy_transformer.spec.library import (
    validate_dimension,
    validate_positive,
    validate_probability,
)
from energy_transformer.spec.primitives import ValidationError

pytestmark = pytest.mark.security


class TestTypeUnionFixes:
    """Test that isinstance() works correctly with type unions."""

    def test_validate_positive_with_numbers(self):
        assert validate_positive(5) is True
        assert validate_positive(0) is False
        assert validate_positive(-5) is False
        assert validate_positive(5.5) is True
        assert validate_positive(0.0) is False
        assert validate_positive(-5.5) is False
        assert validate_positive(float("inf")) is True
        assert validate_positive(float("-inf")) is False

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

    def test_verify_types_script_behavior(self):
        """Test the exact behavior from verify_types.py."""
        from energy_transformer.spec.library import validate_positive

        assert validate_positive(5)
        assert validate_positive(5.5)
        assert validate_positive((2, 3))
        assert not validate_positive(-1)

        test_cases = [
            (5, True),
            (5.5, True),
            ((2, 3), True),
            ((1, -1), False),
            (-1, False),
            ("5", False),
            (None, False),
        ]

        for value, expected in test_cases:
            try:
                result = validate_positive(value)
                assert result == expected, (
                    f"validate_positive({value!r}) returned {result}, expected {expected}"
                )
            except TypeError:
                pytest.fail(
                    f"validate_positive({value!r}) raised TypeError - type union fix failed!",
                )

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
        from dataclasses import dataclass

        from energy_transformer.spec.primitives import Spec, param

        @dataclass(frozen=True)
        class TestSpec(Spec):
            mode: str = param(  # noqa: RUF009
                default="auto",
                choices=["auto", "manual", "hybrid"],
            )
            size: int = param(default=1, choices=[1, 2, 4, 8])  # noqa: RUF009

        spec = TestSpec(mode="manual", size=4)
        assert spec.mode == "manual"
        assert spec.size == 4

    @pytest.mark.usefixtures("caplog")
    def test_choices_with_mixed_types_warns(self):
        import logging
        from dataclasses import dataclass

        from energy_transformer.spec.primitives import Spec, param

        logging.basicConfig(level=logging.WARNING)

        @dataclass(frozen=True)
        class BadSpec(Spec):
            value: str | None = param(default="a", choices=["a", "b", None])  # noqa: RUF009

        BadSpec(value=None)

    def test_choices_type_mismatch_error(self):
        from dataclasses import dataclass

        from energy_transformer.spec.primitives import (
            Spec,
            param,
        )

        @dataclass(frozen=True)
        class TestSpec(Spec):
            size: int = param(choices=[1, 2, 4, 8])  # noqa: RUF009

        with pytest.raises(ValidationError):
            TestSpec(size="2")
