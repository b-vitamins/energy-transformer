"""Specification primitives for declarative model construction.

This module provides the foundational specification system for defining
machine learning models declaratively. Specifications are immutable
descriptions of model components that can be validated, composed, and
transformed into executable modules.

The system uses Python's type system and metaclasses to provide automatic
validation, registration, and introspection capabilities.
"""

from __future__ import annotations

import inspect
from abc import ABC, ABCMeta
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import (
    Any,
    ClassVar,
    TypeAlias,
    TypeVar,
    get_args,
    get_type_hints,
)

import torch.nn as nn

__all__ = [
    "Spec",
    "SpecMeta",
    "AsyncSpec",
    "spec",
    "param",
    "Context",
    "Dimension",
    "DimensionDef",
    "DimensionLike",
    "ValidationError",
    "requires",
    "provides",
    "modifies",
    "validate_field",
    "REQUIRED",
]

# Type aliases
T = TypeVar("T")
DimensionLike: TypeAlias = int | str | None
ModuleFactory = Callable[[Any, "Context"], nn.Module]


# Sentinel for required dimensions
class _Required:
    def __repr__(self) -> str:
        return "REQUIRED"


REQUIRED = _Required()


@dataclass
class DimensionDef:
    """Definition of a dimension with type information and validation.

    Parameters
    ----------
    name : str
        Name of the dimension
    type : type[int] | type[float], optional
        Expected type of the dimension value, default is int
    validator : Callable[[Any], bool], optional
        Validation function for dimension values
    description : str, optional
        Human-readable description of the dimension
    """

    name: str
    type: type[int] | type[float] = int
    validator: Callable[[Any], bool] | None = None
    description: str | None = None


class Dimension:
    """Symbolic dimension that can be resolved from context.

    Dimensions represent values that may not be known at specification time
    but can be resolved from context during realization. They support
    formulas for computed dimensions and validation constraints. Formulas
    support basic arithmetic (+, -, *, /), parentheses, and variables from
    the context. Function calls and attribute access are disallowed to
    prevent code execution.

    Parameters
    ----------
    name : str
        Name of the dimension
    value : DimensionLike, optional
        Static value or reference to another dimension
    formula : str, optional
        Formula to compute dimension value (e.g., "embed_dim * 4")
    constraints : list[Callable[[int], bool]], optional
        Validation constraints for the dimension
    """

    def __init__(
        self,
        name: str,
        value: DimensionLike = None,
        formula: str | None = None,
        constraints: list[Callable[[int], bool]] | None = None,
    ):
        """Initialize Dimension with name, value, formula, and constraints.

        Parameters
        ----------
        name : str
            Name of the dimension
        value : DimensionLike, optional
            Static value or reference to another dimension
        formula : str, optional
            Formula to compute dimension value
        constraints : list[Callable[[int], bool]], optional
            Validation constraints for the dimension
        """
        self.name = name
        self.value = value
        self.formula = formula
        self.constraints = constraints or []

    def _tokenize(self, formula: str) -> list[tuple[str, str]]:
        """Tokenize mathematical formula into (type, value) pairs."""
        import re

        token_regex = re.compile(
            r"(?P<NUMBER>\d+(?:\.\d+)?)|"
            r"(?P<IDENT>[a-zA-Z_][a-zA-Z0-9_]*)|"
            r"(?P<PLUS>\+)|"
            r"(?P<MINUS>-)|"
            r"(?P<TIMES>\*)|"
            r"(?P<DIVIDE>/)|"
            r"(?P<LPAREN>\()|"
            r"(?P<RPAREN>\))|"
            r"(?P<WHITESPACE>\s+)|"
            r"(?P<INVALID>.)",
            re.VERBOSE,
        )

        tokens = []
        for match in token_regex.finditer(formula):
            kind = match.lastgroup
            value = match.group()
            assert kind is not None
            if kind == "WHITESPACE":
                continue
            elif kind == "INVALID":
                raise ValueError(f"Invalid character in formula: {value!r}")
            tokens.append((kind, value))

        return tokens

    def _parse_expression(
        self,
        tokens: list[tuple[str, str]],
        pos: int,
        variables: dict[str, int | None],
    ) -> tuple[float, int]:
        """Parse mathematical expression recursively."""
        left, pos = self._parse_term(tokens, pos, variables)

        while pos < len(tokens) and tokens[pos][0] in ("PLUS", "MINUS"):
            op = tokens[pos][0]
            pos += 1
            right, pos = self._parse_term(tokens, pos, variables)
            if op == "PLUS":
                left = left + right
            else:
                left = left - right

        return left, pos

    def _parse_term(
        self,
        tokens: list[tuple[str, str]],
        pos: int,
        variables: dict[str, int | None],
    ) -> tuple[float, int]:
        """Parse multiplication/division term."""
        left, pos = self._parse_factor(tokens, pos, variables)

        while pos < len(tokens) and tokens[pos][0] in ("TIMES", "DIVIDE"):
            op = tokens[pos][0]
            pos += 1
            right, pos = self._parse_factor(tokens, pos, variables)
            if op == "TIMES":
                left = left * right
            else:
                if right == 0:
                    raise ValueError("Division by zero")
                left = left / right

        return left, pos

    def _parse_factor(
        self,
        tokens: list[tuple[str, str]],
        pos: int,
        variables: dict[str, int | None],
    ) -> tuple[float, int]:
        """Parse number, variable, or parenthesized expression."""
        if pos >= len(tokens):
            raise ValueError("Unexpected end of expression")

        token_type, token_value = tokens[pos]

        if token_type == "NUMBER":
            return float(token_value), pos + 1

        elif token_type == "IDENT":
            if token_value not in variables:
                raise ValueError(f"Unknown variable: {token_value}")
            value = variables[token_value]
            if value is None:
                raise ValueError(f"Variable {token_value} has no value")
            return float(value), pos + 1

        elif token_type == "LPAREN":
            pos += 1  # Skip '('
            expr_value, pos = self._parse_expression(tokens, pos, variables)
            if pos >= len(tokens) or tokens[pos][0] != "RPAREN":
                raise ValueError("Missing closing parenthesis")
            return expr_value, pos + 1  # Skip ')'

        elif token_type == "MINUS":
            pos += 1  # Skip unary minus
            factor_val, pos = self._parse_factor(tokens, pos, variables)
            return -factor_val, pos

        else:
            raise ValueError(f"Unexpected token: {token_value}")

    def _safe_eval_formula(
        self, formula: str, variables: dict[str, int | None]
    ) -> float | None:
        """Safely evaluate mathematical formula without eval()."""
        try:
            tokens = self._tokenize(formula)
            if not tokens:
                return None

            result, pos = self._parse_expression(tokens, 0, variables)

            if pos < len(tokens):
                raise ValueError(
                    f"Unexpected token after expression: {tokens[pos][1]}"
                )

            return result
        except Exception as e:  # pragma: no cover - debugging
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(f"Formula evaluation failed for {formula!r}: {e}")
            return None

    def resolve(self, context: Context) -> int | None:
        """Resolve dimension value from context.

        Parameters
        ----------
        context : Context
            Context containing dimension values

        Returns
        -------
        int | None
            Resolved dimension value or None if unresolvable
        """
        if self.value is not None:
            if isinstance(self.value, str):
                return context.get_dim(self.value)
            return self.value

        if self.formula:
            local_vars = {
                name: context.get_dim(name) for name in context.dimensions
            }
            try:
                result = self._safe_eval_formula(self.formula, local_vars)
                return int(result) if result is not None else None
            except Exception:
                return None

        return context.get_dim(self.name)

    def validate(self, value: int) -> None:
        """Validate resolved dimension value against constraints.

        Parameters
        ----------
        value : int
            Dimension value to validate

        Raises
        ------
        ValidationError
            If value fails any constraint
        """
        for constraint in self.constraints:
            if not constraint(value):
                raise ValidationError(
                    f"Dimension {self.name}={value} failed constraint"
                )


class Context:
    """Execution context containing dimensions and metadata.

    Contexts form a hierarchy allowing scoped dimension resolution
    and metadata storage. They support caching for performance and
    provide a clean interface for dimension management.

    Parameters
    ----------
    dimensions : dict[str, int | None], optional
        Dimension name to value mappings
    metadata : dict[str, Any], optional
        Additional metadata storage
    parent : Context, optional
        Parent context for hierarchical lookup
    """

    def __init__(
        self,
        dimensions: dict[str, int | None] | None = None,
        metadata: dict[str, Any] | None = None,
        parent: Context | None = None,
    ):
        """Initialize Context with dimensions, metadata, and parent.

        Parameters
        ----------
        dimensions : dict[str, int | None], optional
            Dimension name to value mappings
        metadata : dict[str, Any], optional
            Additional metadata storage
        parent : Context, optional
            Parent context for hierarchical lookup
        """
        self.dimensions = dimensions or {}
        self.metadata = metadata or {}
        self.parent = parent
        self._cache: dict[str, Any] = {}

    def get_dim(self, name: str, default: int | None = None) -> int | None:
        """Get dimension value by name with hierarchical lookup.

        Parameters
        ----------
        name : str
            Dimension name to look up
        default : int | None, optional
            Default value if dimension not found

        Returns
        -------
        int | None
            Dimension value, default, or None if not found
        """
        if name in self.dimensions:
            return self.dimensions[name]
        if self.parent:
            return self.parent.get_dim(name, default)
        return default

    def set_dim(self, name: str, value: int) -> None:
        """Set dimension value in this context.

        Parameters
        ----------
        name : str
            Dimension name
        value : int
            Dimension value
        """
        self.dimensions[name] = value
        # Invalidate cache entries that might depend on this
        self._cache.clear()

    def update(self, **kwargs: int) -> None:
        """Update multiple dimensions at once.

        Parameters
        ----------
        **kwargs : int
            Dimension name to value mappings
        """
        self.dimensions.update(kwargs)
        self._cache.clear()

    def child(self, **updates: Any) -> Context:
        """Create child context with updates.

        Parameters
        ----------
        **updates : Any
            Dimension updates for child context

        Returns
        -------
        Context
            New child context
        """
        child = Context(
            dimensions=self.dimensions.copy(),
            metadata=self.metadata.copy(),
            parent=self,
        )
        child.update(**updates)
        return child

    def cache_get(self, key: str) -> Any | None:
        """Get cached value by key.

        Parameters
        ----------
        key : str
            Cache key

        Returns
        -------
        Any | None
            Cached value or None
        """
        return self._cache.get(key)

    def cache_set(self, key: str, value: Any) -> None:
        """Set cached value.

        Parameters
        ----------
        key : str
            Cache key
        value : Any
            Value to cache
        """
        self._cache[key] = value

    def __repr__(self) -> str:
        """Return detailed string representation of the context.

        Returns
        -------
        str
            Detailed representation including dimensions and metadata
        """
        return (
            f"Context(dimensions={self.dimensions!r}, "
            f"metadata={self.metadata!r})"
        )


class ValidationError(ValueError):
    """Validation error with structured debugging information.

    Parameters
    ----------
    message : str
        Primary error message
    spec : Spec, optional
        Specification that failed validation
    field_name : str, optional
        Field that caused the error
    suggestion : str, optional
        Helpful suggestion for fixing the error
    context : Context, optional
        Context during validation
    """

    def __init__(
        self,
        message: str,
        spec: Spec | None = None,
        field_name: str | None = None,
        suggestion: str | None = None,
        context: Context | None = None,
    ):
        """Initialize ValidationError with debugging information.

        Parameters
        ----------
        message : str
            Primary error message
        spec : Spec, optional
            Specification that failed validation
        field_name : str, optional
            Field that caused the error
        suggestion : str, optional
            Helpful suggestion for fixing the error
        context : Context, optional
            Context during validation
        """
        self.spec = spec
        self.field_name = field_name
        self.suggestion = suggestion
        self.context = context

        parts = [message]
        if spec:
            parts.append(f"in {spec.__class__.__name__}")
        if field_name:
            parts.append(f"field '{field_name}'")
        if context:
            parts.append(f"with context {context}")
        if suggestion:
            parts.append(f"\nSuggestion: {suggestion}")

        super().__init__(" ".join(parts))


def param(
    default: Any = REQUIRED,
    *,
    default_factory: Callable[[], Any] | None = None,
    validator: Callable[[Any], bool] | None = None,
    description: str | None = None,
    dimension: bool = False,
    choices: list[Any] | None = None,
) -> Any:
    """Define a specification parameter with validation and metadata.

    Parameters
    ----------
    default : Any
        Default value or REQUIRED for required parameters
    default_factory : Callable[[], Any], optional
        Factory function for default value (for mutable defaults)
    validator : Callable[[Any], bool], optional
        Validation function returning True if valid
    description : str, optional
        Parameter description for documentation
    dimension : bool, optional
        Whether this parameter represents a dimension
    choices : list[Any], optional
        Allowed values for the parameter

    Returns
    -------
    Field
        Dataclass field with metadata

    Examples
    --------
    >>> @dataclass(frozen=True)
    ... class MySpec(Spec):
    ...     size: int = param(
    ...         validator=lambda x: x > 0,
    ...         description="Size must be positive"
    ...     )
    ...     mode: str = param(default="auto", choices=["auto", "manual"])
    ...     items: list = param(default_factory=list)
    """
    metadata = {
        "validator": validator,
        "description": description,
        "dimension": dimension,
        "choices": choices,
        "required": default is REQUIRED and default_factory is None,
    }

    if default_factory is not None:
        return field(default_factory=default_factory, metadata=metadata)
    elif default is REQUIRED:
        return field(metadata=metadata)
    else:
        return field(default=default, metadata=metadata)


class SpecMeta(ABCMeta):
    """Metaclass for automatic spec registration and validation setup."""

    _registry: dict[str, type[Spec]] = {}
    _realisers: dict[type[Spec], ModuleFactory] = {}

    def __new__(
        cls: type[SpecMeta],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type[Spec]:
        """Create new spec class with automatic registration."""
        new_class = super().__new__(cls, name, bases, namespace, **kwargs)

        # Auto-register non-abstract specs
        if not inspect.isabstract(new_class) and name != "Spec":
            SpecMeta._registry[name] = new_class  # type: ignore[assignment]

        return new_class  # type: ignore[return-value]

    @classmethod
    def register_realiser(
        cls, spec_cls: type[Spec]
    ) -> Callable[[ModuleFactory], ModuleFactory]:
        """Register a realiser function for a spec type.

        Parameters
        ----------
        spec_cls : type[Spec]
            Spec class to register realiser for

        Returns
        -------
        Callable
            Decorator function
        """

        def decorator(fn: ModuleFactory) -> ModuleFactory:
            cls._realisers[spec_cls] = fn
            return fn

        return decorator

    @classmethod
    def get_realiser(cls, spec_cls: type[Spec]) -> ModuleFactory | None:
        """Get registered realiser for a spec type.

        Parameters
        ----------
        spec_cls : type[Spec]
            Spec class to get realiser for

        Returns
        -------
        ModuleFactory | None
            Realiser function or None
        """
        # Check exact match first
        if spec_cls in cls._realisers:
            return cls._realisers[spec_cls]

        # Check parent classes
        for base in spec_cls.__mro__[1:]:
            if base in cls._realisers:
                return cls._realisers[base]

        return None


@dataclass(frozen=True)
class Spec(ABC, metaclass=SpecMeta):
    """Base specification with automatic validation and registration.

    Specifications are immutable descriptions of model components that
    can be validated, composed, and transformed into executable modules.
    They support automatic parameter validation, dimension tracking,
    and hierarchical composition.

    Class Attributes
    ----------------
    _requires : set[str]
        Required dimensions from context
    _provides : set[str]
        Dimensions provided to context
    _modifies : set[str]
        Dimensions modified in context
    _version : str
        Specification version for compatibility
    _compatible_versions : set[str]
        Compatible specification versions
    """

    # Class-level configuration
    _requires: ClassVar[set[str]] = set()
    _provides: ClassVar[set[str]] = set()
    _modifies: ClassVar[set[str]] = set()
    _version: ClassVar[str] = "1.0"
    _compatible_versions: ClassVar[set[str]] = {"1.0"}

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self._validate_all_fields()

    def _validate_all_fields(self) -> None:  # noqa: C901
        """Validate all fields using metadata and type hints."""
        hints = get_type_hints(self.__class__)

        for field_info in fields(self):
            value = getattr(self, field_info.name)
            field_type = hints.get(field_info.name, Any)

            # Skip if None and Optional
            if value is None and type(None) in get_args(field_type):
                continue

            # Check required fields
            if field_info.metadata.get("required") and value is REQUIRED:
                raise ValidationError(
                    "Required parameter not provided",
                    spec=self,
                    field_name=field_info.name,
                )

            # Run field validator
            if validator := field_info.metadata.get("validator"):
                if not validator(value):
                    raise ValidationError(
                        f"Validation failed for {value!r}",
                        spec=self,
                        field_name=field_info.name,
                    )

            # Check choices
            if choices := field_info.metadata.get("choices"):
                if value not in choices:
                    raise ValidationError(
                        f"Value {value!r} not in allowed choices {choices}",
                        spec=self,
                        field_name=field_info.name,
                    )

                if choices:
                    choice_types = {type(c) for c in choices}
                    if len(choice_types) > 1:
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"Field {field_info.name} has choices with mixed types: {choice_types}"
                        )

                    value_type = type(value)
                    if value_type not in choice_types and value in choices:
                        raise ValidationError(
                            f"Value type {value_type.__name__} doesn't match choice types {choice_types}",
                            spec=self,
                            field_name=field_info.name,
                            suggestion="Ensure all choices have consistent types",
                        )

    def validate(self, context: Context) -> list[str]:
        """Validate spec against context, returning issues.

        Parameters
        ----------
        context : Context
            Context to validate against

        Returns
        -------
        list[str]
            List of validation issues (empty if valid)
        """
        issues: list[str] = []

        # Step 1: check version compatibility with original context
        issues.extend(self._validate_version(context))

        # Step 2: ensure required dimensions exist before updates
        issues.extend(self._validate_required_dimensions(context))

        # Step 3: create child context with this spec's updates
        child_context = context.child()
        try:
            updated_context = self.apply_context(child_context)
        except Exception as e:
            issues.append(
                f"Failed to apply context updates: {type(e).__name__}: {e}"
            )
            return issues

        # Step 4: validate dimension parameters with updated context
        issues.extend(self._validate_dimension_parameters(updated_context))

        # Step 5: validate children with updated context
        for i, child in enumerate(self.children()):
            try:
                child_issues = child.validate(updated_context)
                for issue in child_issues:
                    issues.append(
                        f"{self.__class__.__name__}.children[{i}] "
                        f"({child.__class__.__name__}): {issue}"
                    )
            except Exception as e:
                issues.append(
                    f"{self.__class__.__name__}.children[{i}]: "
                    f"Validation crashed with {type(e).__name__}: {e}"
                )

        return issues

    def _validate_version(self, context: Context) -> list[str]:
        """Validate version compatibility.

        Parameters
        ----------
        context : Context
            Context to validate against

        Returns
        -------
        list[str]
            Version-related validation issues
        """
        if hasattr(context, "spec_version"):
            if context.spec_version not in self._compatible_versions:
                return [
                    f"Version mismatch: context has {context.spec_version}, "
                    f"spec supports {self._compatible_versions}"
                ]
        return []

    def _validate_required_dimensions(self, context: Context) -> list[str]:
        """Validate required dimensions are present.

        Parameters
        ----------
        context : Context
            Context to validate against

        Returns
        -------
        list[str]
            Missing dimension issues
        """
        issues = []
        for dim in self._requires:
            if context.get_dim(dim) is None:
                issues.append(f"Missing required dimension: {dim}")
        return issues

    def _validate_dimension_parameters(self, context: Context) -> list[str]:
        """Validate dimension parameters.

        Parameters
        ----------
        context : Context
            Context to validate against

        Returns
        -------
        list[str]
            Dimension parameter validation issues
        """
        issues = []
        for field_info in fields(self):
            if field_info.metadata.get("dimension"):
                value = getattr(self, field_info.name)
                if isinstance(value, Dimension):
                    try:
                        if resolved := value.resolve(context):
                            value.validate(resolved)
                    except ValidationError as e:
                        issues.append(str(e))
        return issues

    def _validate_children(self, context: Context) -> list[str]:
        """Validate child specifications.

        DEPRECATED: This method is now integrated into validate() for proper
        context propagation. Kept for backward compatibility only.

        Parameters
        ----------
        context : Context
            Context to validate against

        Returns
        -------
        list[str]
            Empty list (validation now done in main validate method)
        """
        return []

    def apply_context(self, context: Context) -> Context:
        """Apply this spec's effects to context.

        Parameters
        ----------
        context : Context
            Context to update

        Returns
        -------
        Context
            Updated context
        """
        # Update context with provided dimensions
        for dim in self._provides:
            if hasattr(self, dim):
                value = getattr(self, dim)
                if isinstance(value, int):
                    context.set_dim(dim, value)
                elif isinstance(value, Dimension):
                    if resolved := value.resolve(context):
                        context.set_dim(dim, resolved)

        return context

    def children(self) -> list[Spec]:
        """Return child specs for traversal.

        Returns
        -------
        list[Spec]
            Direct child specifications
        """
        children = []
        for field_info in fields(self):
            value = getattr(self, field_info.name)
            if isinstance(value, Spec):
                children.append(value)
            elif isinstance(value, list | tuple):
                children.extend(v for v in value if isinstance(v, Spec))
        return children

    def map(self, fn: Callable[[Spec], T]) -> list[T]:
        """Map function over spec tree.

        Parameters
        ----------
        fn : Callable[[Spec], T]
            Function to apply to each spec

        Returns
        -------
        list[T]
            Results from applying function
        """
        results = [fn(self)]
        for child in self.children():
            results.extend(child.map(fn))
        return results

    def find(self, predicate: Callable[[Spec], bool]) -> list[Spec]:
        """Find all specs matching predicate.

        Parameters
        ----------
        predicate : Callable[[Spec], bool]
            Function to test each spec

        Returns
        -------
        list[Spec]
            Specs that match predicate
        """
        return [spec for spec in self.map(lambda s: s) if predicate(spec)]

    def to_dict(self) -> dict[str, Any]:
        """Convert spec to dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary representation
        """
        data = {
            "_type": self.__class__.__name__,
            "_version": self._version,
        }

        for field_info in fields(self):
            value = getattr(self, field_info.name)
            if isinstance(value, Spec):
                value = value.to_dict()
            elif isinstance(value, list | tuple):
                value = [
                    v.to_dict() if isinstance(v, Spec) else v for v in value
                ]
            data[field_info.name] = value

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Spec:
        """Create spec from dictionary representation.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary representation

        Returns
        -------
        Spec
            Reconstructed specification
        """
        spec_type = data.pop("_type")
        version = data.pop("_version", "1.0")
        spec_cls = SpecMeta._registry.get(spec_type)

        if not spec_cls:
            raise ValueError(f"Unknown spec type: {spec_type}")

        # Check version compatibility
        if version not in spec_cls._compatible_versions:
            raise ValueError(
                f"Version {version} not compatible with {spec_cls.__name__}"
            )

        # Recursively convert nested specs
        kwargs = {}
        for key, value in data.items():
            if isinstance(value, dict) and "_type" in value:
                value = Spec.from_dict(value)
            elif isinstance(value, list):
                value = [
                    Spec.from_dict(v)
                    if isinstance(v, dict) and "_type" in v
                    else v
                    for v in value
                ]
            kwargs[key] = value

        return spec_cls(**kwargs)

    def __repr__(self) -> str:
        """Return string representation."""
        params = []
        for field_info in fields(self):
            value = getattr(self, field_info.name)
            if value != field_info.default:
                params.append(f"{field_info.name}={value!r}")

        return f"{self.__class__.__name__}({', '.join(params)})"

    def __rshift__(self, other: Spec) -> Any:
        """Chain specs using >> operator to create Sequential."""
        from .combinators import Sequential

        return Sequential(parts=(self, other))

    def __lshift__(self, other: Spec) -> Any:
        """Prepend spec using << operator to create Sequential."""
        from .combinators import Sequential

        return Sequential(parts=(other, self))

    def __or__(self, other: Spec) -> Any:
        """Create parallel composition using | operator."""
        from .combinators import Parallel

        return Parallel(branches=(self, other))


@dataclass(frozen=True)
class AsyncSpec(Spec):
    """Base specification with async support.

    Extends Spec to support asynchronous context application and
    validation for streaming or async model components.
    """

    async def apply_context_async(self, context: Context) -> Context:
        """Apply spec effects to context asynchronously.

        Parameters
        ----------
        context : Context
            Context to update

        Returns
        -------
        Context
            Updated context
        """
        return self.apply_context(context)

    async def validate_async(self, context: Context) -> list[str]:
        """Validate spec asynchronously.

        Parameters
        ----------
        context : Context
            Context to validate against

        Returns
        -------
        list[str]
            List of validation issues
        """
        return self.validate(context)


# Decorators for dimension requirements
def requires(*dims: str) -> Callable[[type[T]], type[T]]:
    """Declare required dimensions for a spec class.

    Parameters
    ----------
    *dims : str
        Dimension names required by the spec

    Returns
    -------
    Callable
        Class decorator

    Examples
    --------
    >>> @requires("embed_dim", "num_heads")
    ... class AttentionSpec(Spec):
    ...     pass
    """

    def decorator(cls: type[T]) -> type[T]:
        if hasattr(cls, "_requires"):
            cls._requires = set(dims)  # type: ignore[attr-defined]
        return cls

    return decorator


def provides(*dims: str) -> Callable[[type[T]], type[T]]:
    """Declare dimensions provided by a spec class.

    Parameters
    ----------
    *dims : str
        Dimension names provided by the spec

    Returns
    -------
    Callable
        Class decorator
    """

    def decorator(cls: type[T]) -> type[T]:
        if hasattr(cls, "_provides"):
            cls._provides = set(dims)  # type: ignore[attr-defined]
        return cls

    return decorator


def modifies(*dims: str) -> Callable[[type[T]], type[T]]:
    """Declare dimensions modified by a spec class.

    Parameters
    ----------
    *dims : str
        Dimension names modified by the spec

    Returns
    -------
    Callable
        Class decorator
    """

    def decorator(cls: type[T]) -> type[T]:
        if hasattr(cls, "_modifies"):
            cls._modifies = set(dims)  # type: ignore[attr-defined]
        return cls

    return decorator


def spec(
    *,
    requires: set[str] | None = None,
    provides: set[str] | None = None,
    modifies: set[str] | None = None,
) -> Callable[[type[T]], type[T]]:
    """Declare dimension requirements for a spec class.

    Parameters
    ----------
    requires : set[str], optional
        Required dimension names
    provides : set[str], optional
        Provided dimension names
    modifies : set[str], optional
        Modified dimension names

    Returns
    -------
    Callable
        Class decorator
    """

    def decorator(cls: type[T]) -> type[T]:
        if requires and hasattr(cls, "_requires"):
            cls._requires = requires  # type: ignore[attr-defined]
        if provides and hasattr(cls, "_provides"):
            cls._provides = provides  # type: ignore[attr-defined]
        if modifies and hasattr(cls, "_modifies"):
            cls._modifies = modifies  # type: ignore[attr-defined]
        return cls

    return decorator


def validate_field(
    field_name: str,
    validator: Callable[[Any], bool],
    message: str | None = None,
) -> Callable[[type[T]], type[T]]:
    """Add field validator to spec class.

    Parameters
    ----------
    field_name : str
        Name of field to validate
    validator : Callable[[Any], bool]
        Validation function
    message : str, optional
        Custom error message

    Returns
    -------
    Callable
        Class decorator
    """

    def decorator(cls: type[T]) -> type[T]:
        if not hasattr(cls, "_field_validators"):
            cls._field_validators = {}  # type: ignore[attr-defined]
        cls._field_validators[field_name] = (  # type: ignore[attr-defined]
            validator,
            message,
        )
        return cls

    return decorator
