"""Tests for specification primitives."""

from dataclasses import dataclass

import pytest

from energy_transformer.spec.primitives import (
    REQUIRED,
    Context,
    Dimension,
    Spec,
    SpecMeta,
    ValidationError,
    modifies,
    param,
    provides,
    requires,
    spec,
    validate_field,
)

pytestmark = pytest.mark.unit


# Test specs for validation
@dataclass(frozen=True)
@requires("embed_dim")
@provides("hidden_dim")
class ExampleSpec(Spec):
    """Test specification with dimension requirements."""

    size: int = param(  # noqa: RUF009
        default=REQUIRED,
        validator=lambda x: x > 0,
        description="Size must be positive",
    )
    ratio: float = param(  # noqa: RUF009
        default=1.0,
        validator=lambda x: 0 < x <= 1,
        description="Ratio between 0 and 1",
    )
    mode: str = param(default="auto", choices=["auto", "manual", "hybrid"])  # noqa: RUF009
    hidden_dim: Dimension = param(  # noqa: RUF009
        default_factory=lambda: Dimension(
            "hidden_dim",
            formula="embed_dim * 4",
        ),
        dimension=True,
    )


class TestContext:
    """Test Context functionality."""

    def test_basic_operations(self):
        ctx = Context()

        # Test get/set
        assert ctx.get_dim("foo") is None
        ctx.set_dim("foo", 42)
        assert ctx.get_dim("foo") == 42

        # Test update
        ctx.update(bar=10, baz=20)
        assert ctx.get_dim("bar") == 10
        assert ctx.get_dim("baz") == 20

    def test_hierarchy(self):
        parent = Context(dimensions={"a": 1, "b": 2})
        child = parent.child(b=3, c=4)

        assert child.get_dim("a") == 1  # Inherited
        assert child.get_dim("b") == 3  # Overridden
        assert child.get_dim("c") == 4  # New
        assert parent.get_dim("c") is None  # Not in parent

    def test_caching(self):
        ctx = Context()
        assert ctx.cache_get("key") is None

        ctx.cache_set("key", "value")
        assert ctx.cache_get("key") == "value"

        # Cache cleared on dimension update
        ctx.set_dim("foo", 1)
        assert ctx.cache_get("key") is None


class TestDimension:
    """Test Dimension functionality."""

    def test_static_value(self):
        dim = Dimension("test", value=42)
        ctx = Context()
        assert dim.resolve(ctx) == 42

    def test_reference(self):
        dim = Dimension("test", value="other")
        ctx = Context(dimensions={"other": 100})
        assert dim.resolve(ctx) == 100

    def test_formula(self):
        dim = Dimension("hidden", formula="embed_dim * 4")
        ctx = Context(dimensions={"embed_dim": 768})
        assert dim.resolve(ctx) == 3072

    def test_constraints(self):
        dim = Dimension(
            "test",
            value=10,
            constraints=[lambda x: x > 0, lambda x: x % 2 == 0],
        )

        dim.validate(10)  # Should pass

        with pytest.raises(ValidationError):
            dim.validate(-1)  # Fails first constraint

        with pytest.raises(ValidationError):
            dim.validate(3)  # Fails second constraint


class TestValidationError:
    """Test ValidationError functionality."""

    def test_error_formatting(self):
        spec = ExampleSpec(size=10)
        ctx = Context()

        err = ValidationError(
            "Test error",
            spec=spec,
            field_name="size",
            suggestion="Try a different value",
            context=ctx,
        )

        str_repr = str(err)
        assert "Test error" in str_repr
        assert "ExampleSpec" in str_repr
        assert "size" in str_repr
        assert "Try a different value" in str_repr


class TestParam:
    """Test param field factory."""

    def test_required_param(self):
        @dataclass(frozen=True)
        class RequiredSpec(Spec):
            value: int = param(default=REQUIRED)  # noqa: RUF009

        with pytest.raises(TypeError, match="missing 1 required"):
            RequiredSpec()

    def test_validator(self):
        @dataclass(frozen=True)
        class ValidatedSpec(Spec):
            value: int = param(validator=lambda x: x > 0)  # noqa: RUF009

        ValidatedSpec(value=1)  # Should pass

        with pytest.raises(ValidationError, match="Validation failed"):
            ValidatedSpec(value=-1)

    def test_choices(self):
        @dataclass(frozen=True)
        class ChoiceSpec(Spec):
            mode: str = param(choices=["a", "b", "c"])  # noqa: RUF009

        ChoiceSpec(mode="a")  # Should pass

        with pytest.raises(ValidationError, match="not in allowed choices"):
            ChoiceSpec(mode="d")


class TestSpecMeta:
    """Test SpecMeta metaclass functionality."""

    def test_auto_registration(self):
        # ExampleSpec should be auto-registered
        assert "ExampleSpec" in SpecMeta._registry
        assert SpecMeta._registry["ExampleSpec"] is ExampleSpec

    def test_realiser_registration(self):
        @dataclass(frozen=True)
        class CustomSpec(Spec):
            pass

        @SpecMeta.register_realiser(CustomSpec)
        def realise_custom(_spec, _context):
            return "custom_module"

        assert SpecMeta.get_realiser(CustomSpec) is realise_custom

        # Test inheritance lookup
        @dataclass(frozen=True)
        class DerivedSpec(CustomSpec):
            pass

        assert SpecMeta.get_realiser(DerivedSpec) is realise_custom


class TestSpecClass:
    """Test base Spec functionality."""

    def test_validation_integration(self):
        spec = ExampleSpec(size=10, ratio=0.5)
        ctx = Context(dimensions={"embed_dim": 768})

        issues = spec.validate(ctx)
        assert len(issues) == 0

        # Missing required dimension
        ctx2 = Context()
        issues = spec.validate(ctx2)
        assert any("embed_dim" in issue for issue in issues)

    def test_apply_context(self):
        spec = ExampleSpec(size=10)
        ctx = Context(dimensions={"embed_dim": 768})

        new_ctx = spec.apply_context(ctx)
        assert new_ctx.get_dim("hidden_dim") == 3072

    def test_to_from_dict(self):
        spec = ExampleSpec(size=10, ratio=0.8, mode="manual")

        # Convert to dict
        data = spec.to_dict()
        assert data["_type"] == "ExampleSpec"
        assert data["size"] == 10
        assert data["ratio"] == 0.8
        assert data["mode"] == "manual"

        # Reconstruct from dict
        spec2 = Spec.from_dict(data)
        assert isinstance(spec2, ExampleSpec)
        assert spec2.size == 10
        assert spec2.ratio == 0.8
        assert spec2.mode == "manual"

    def test_tree_operations(self):
        @dataclass(frozen=True)
        class ParentSpec(Spec):
            child: Spec

        @dataclass(frozen=True)
        class LeafSpec(Spec):
            value: int = 1

        tree = ParentSpec(child=LeafSpec())

        # Test children
        assert len(tree.children()) == 1
        assert isinstance(tree.children()[0], LeafSpec)

        # Test map
        results = tree.map(lambda s: s.__class__.__name__)
        assert results == ["ParentSpec", "LeafSpec"]

        # Test find
        leaves = tree.find(lambda s: isinstance(s, LeafSpec))
        assert len(leaves) == 1
        assert isinstance(leaves[0], LeafSpec)


class TestDecorators:
    """Test spec decorators."""

    def test_requires_provides_modifies(self):
        @requires("a", "b")
        @provides("c", "d")
        @modifies("e")
        @dataclass(frozen=True)
        class DecoratedSpec(Spec):
            pass

        assert DecoratedSpec._requires == {"a", "b"}
        assert DecoratedSpec._provides == {"c", "d"}
        assert DecoratedSpec._modifies == {"e"}

    def test_spec_decorator(self):
        @spec(
            requires={"input_dim"},
            provides={"output_dim"},
            modifies={"hidden_state"},
        )
        @dataclass(frozen=True)
        class SpecDecoratedSpec(Spec):
            pass

        assert SpecDecoratedSpec._requires == {"input_dim"}
        assert SpecDecoratedSpec._provides == {"output_dim"}
        assert SpecDecoratedSpec._modifies == {"hidden_state"}

    def test_validate_field(self):
        @validate_field("value", lambda x: x > 0, "Must be positive")
        @dataclass(frozen=True)
        class FieldValidatedSpec(Spec):
            value: int

        assert hasattr(FieldValidatedSpec, "_field_validators")
        validator, message = FieldValidatedSpec._field_validators["value"]

        assert validator(10) is True
        assert validator(-1) is False
        assert message == "Must be positive"


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
