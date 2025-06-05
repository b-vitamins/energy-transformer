# ruff: noqa: TRY003
"""Specification realisation system for creating PyTorch modules.

This module converts abstract specifications into concrete PyTorch modules
through a plugin-based architecture. The realisation system supports caching,
automatic module discovery, and extensibility through plugins.

The system maintains a clear separation between specification (what to build)
and realisation (how to build it), enabling multiple implementations of the
same specification.
"""
# ruff: noqa: TRY003

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import logging
import threading
import time
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeVar,
    cast,
    get_type_hints,
)

import torch
from torch import nn

from .combinators import (
    Conditional,
    Graph,
    Identity,
    Lambda,
    Loop,
    Parallel,
    Residual,
    Sequential,
    Switch,
)
from .metrics import MetricsCollector
from .primitives import Context, Spec, SpecMeta, ValidationError


@dataclass
class RealisationConstants:
    """Configuration constants for the realisation system."""

    # Stack and recursion limits
    MAX_RECURSION: int = 100
    MAX_STACK_PREVIEW: int = 5

    # Graph processing
    EDGE_TUPLE_SIZE: int = 2
    FULL_EDGE_SIZE: int = 3

    # Loop unrolling
    UNROLL_LIMIT: int = 12

    # Cache settings
    DEFAULT_CACHE_SIZE: int = 128


# Default mappings for auto-importing modules based on Spec names
module_mappings = {
    "LayerNormSpec": ("energy_transformer.layers", "EnergyLayerNorm"),
    "PatchEmbedSpec": (
        "energy_transformer.layers.embeddings",
        "ConvPatchEmbed",
    ),
    "PosEmbedSpec": (
        "energy_transformer.layers.embeddings",
        "PosEmbed2D",
    ),
    "MHEASpec": (
        "energy_transformer.layers.attention",
        "MultiheadEnergyAttention",
    ),
    "MHASpec": ("torch.nn", "MultiheadAttention"),
    "HNSpec": ("energy_transformer.layers.hopfield", "HopfieldNetwork"),
    "SHNSpec": (
        "energy_transformer.layers.simplicial",
        "SimplicialHopfieldNetwork",
    ),
    "ClassificationHeadSpec": (
        "energy_transformer.layers.heads",
        "ClassifierHead",
    ),
    "MLPSpec": ("energy_transformer.layers.mlp", "MLP"),
    "DropoutSpec": ("torch.nn", "Dropout"),
    "IdentitySpec": ("torch.nn", "Identity"),
}


class AutoImporter:
    """Handles automatic module importing for specs."""

    def __init__(self, context: Context, warnings_enabled: bool = True) -> None:
        self.context = context
        self.warnings_enabled = warnings_enabled
        self.logger = logging.getLogger(__name__)

    def try_import(self, spec: Spec) -> nn.Module | None:
        """Try to automatically import and instantiate a module."""
        spec_name = spec.__class__.__name__
        mapping = module_mappings.get(spec_name)

        if not mapping:
            if self.warnings_enabled:
                self.logger.debug(
                    "No auto-import mapping for %s. Available mappings: %s",
                    spec_name,
                    list(module_mappings.keys()),
                )
            return None

        module_path, class_name = mapping
        module = self._import_module(module_path)
        if not module:
            return None

        cls = self._get_class(module, module_path, class_name)
        if not cls:
            return None

        kwargs = self._extract_kwargs(spec, spec_name)
        if kwargs is None:
            return None

        return self._instantiate(cls, class_name, kwargs)

    def _import_module(self, module_path: str) -> ModuleType | None:
        """Import a module by path."""
        try:
            return importlib.import_module(module_path)
        except ImportError as e:
            if self.warnings_enabled:
                self.logger.warning(
                    "Failed to import %s for auto-import: %s. "
                    "Is the module installed? Try: pip install energy-transformer[all]",
                    module_path,
                    e,
                )
            return None
        except Exception as e:  # pragma: no cover - unexpected
            if self.warnings_enabled:
                self.logger.exception(
                    "Unexpected error importing %s: %s",
                    module_path,
                    type(e).__name__,
                )
            return None

    def _get_class(
        self, module: ModuleType, module_path: str, class_name: str
    ) -> type[nn.Module] | None:
        """Get a class from a module."""
        try:
            return cast(type[nn.Module], getattr(module, class_name))
        except AttributeError:
            if self.warnings_enabled:
                available = [a for a in dir(module) if not a.startswith("_")]
                self.logger.warning(
                    "Module %s has no attribute %s. Available attributes: %s",
                    module_path,
                    class_name,
                    available[:10],
                )
            return None

    def _extract_kwargs(
        self, spec: Spec, spec_name: str
    ) -> dict[str, Any] | None:
        """Extract kwargs from spec based on spec type."""
        try:
            kwargs = self._get_base_kwargs(spec)
            self._apply_spec_specific_logic(spec, spec_name, kwargs)
            return self._clean_kwargs(kwargs)
        except Exception as e:  # pragma: no cover - unexpected
            if self.warnings_enabled:
                self.logger.exception(
                    "Failed to extract kwargs from %s: %s",
                    spec_name,
                    type(e).__name__,
                )
            return None

    def _get_base_kwargs(self, spec: Spec) -> dict[str, Any]:
        """Extract base kwargs from dataclass fields."""
        kwargs: dict[str, Any] = {}

        if hasattr(spec, "__dataclass_fields__"):
            for field_name, field_info in spec.__dataclass_fields__.items():
                if field_name.startswith("_"):
                    continue
                value = getattr(spec, field_name)
                if self._is_default_value(value, field_info):
                    continue
                kwargs[field_name] = value

        return kwargs

    def _is_default_value(
        self, value: object, field_info: dataclasses.Field[object]
    ) -> bool:
        """Check if a value is the default for a field."""
        if (
            hasattr(field_info, "default_factory")
            and field_info.default_factory is not dataclasses.MISSING
        ):
            return False

        if (
            hasattr(field_info, "default")
            and field_info.default is not dataclasses.MISSING
        ):
            return value == field_info.default

        return False

    def _apply_spec_specific_logic(
        self, spec: Spec, spec_name: str, kwargs: dict[str, Any]
    ) -> None:
        """Apply spec-specific parameter transformation logic."""
        handler_name = f"_handle_{spec_name.lower()}"
        handler = getattr(self, handler_name, None)
        if handler:
            handler(spec, kwargs)

    def _handle_mheaspec(self, spec: MHEASpec, kwargs: dict[str, Any]) -> None:
        embed_dim = self.context.get_dim("embed_dim")
        if embed_dim is None:
            embed_dim = spec.num_heads * spec.head_dim
        kwargs["embed_dim"] = embed_dim
        kwargs.pop("head_dim", None)
        kwargs.pop("bias", None)

    def _handle_mhaspec(self, _spec: Spec, kwargs: dict[str, Any]) -> None:
        if embed_dim := self.context.get_dim("embed_dim"):
            kwargs["embed_dim"] = embed_dim

    def _handle_hnspec(self, spec: HNSpec, kwargs: dict[str, Any]) -> None:
        if embed_dim := self.context.get_dim("embed_dim"):
            kwargs["embed_dim"] = embed_dim
            if spec.hidden_dim is None and hasattr(spec, "multiplier"):
                kwargs["hidden_dim"] = int(embed_dim * spec.multiplier)
                kwargs.pop("multiplier", None)
        if energy_fn := kwargs.pop("energy_fn", None):
            if energy_fn == "relu_squared":
                kwargs["activation"] = "relu"
            elif energy_fn == "softmax":
                kwargs["activation"] = "softmax"

    def _handle_shnspec(self, spec: SHNSpec, kwargs: dict[str, Any]) -> None:
        if embed_dim := self.context.get_dim("embed_dim"):
            kwargs["in_dim"] = embed_dim
            if spec.hidden_dim is None and hasattr(spec, "multiplier"):
                pass

        if (
            "num_vertices" not in kwargs
            and spec.num_vertices is None
            and (vertices := self.context.get_dim("simplicial_vertices"))
        ):
            kwargs["num_vertices"] = vertices

    def _clean_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        cleaned = {
            k: v
            for k, v in kwargs.items()
            if v is not None and k not in ["_type", "_version"]
        }

        if self.warnings_enabled and self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Auto-import kwargs after cleaning: %s", cleaned)

        return cleaned

    def _instantiate(
        self, cls: type, class_name: str, kwargs: dict[str, Any]
    ) -> nn.Module | None:
        try:
            instance = cls(**kwargs)
        except TypeError as e:
            if self.warnings_enabled:
                self.logger.warning(
                    "Failed to instantiate %s: %s. Provided kwargs: %s. "
                    "This usually means the spec and module have incompatible parameters.",
                    class_name,
                    str(e),
                    list(kwargs.keys()),
                )
            return None
        except Exception as e:  # pragma: no cover - unexpected
            if self.warnings_enabled:
                self.logger.exception(
                    "Failed to instantiate %s: %s",
                    class_name,
                    type(e).__name__,
                )
            return None
        else:
            if not isinstance(instance, nn.Module):
                if self.warnings_enabled:
                    self.logger.warning(
                        "Auto-imported %s is not an nn.Module, got %s",
                        class_name,
                        type(instance),
                    )
                return None

            if self.warnings_enabled:
                self.logger.info(
                    "Successfully auto-imported %s as %s",
                    class_name,
                    instance.__class__.__name__,
                )

            return instance


if TYPE_CHECKING:
    from .debug import DebugTracer
    from .library import HNSpec, MHEASpec, SHNSpec

__all__ = [
    "ModuleCache",
    "RealisationError",
    "Realiser",
    "RealiserPlugin",
    "configure_realisation",
    "from_yaml",
    "optimize_spec",
    "realise",
    "register",
    "to_yaml",
    "visualize",
]

T = TypeVar("T", bound=nn.Module)


class RealisationError(Exception):
    """Realisation error with debugging information.

    Provides detailed error context including the specification,
    context state, and suggestions for resolution.

    Parameters
    ----------
    message : str
        Primary error message
    spec : Spec, optional
        Specification that failed realisation
    context : Context, optional
        Context during realisation
    cause : Exception, optional
        Underlying exception
    suggestion : str, optional
        Helpful suggestion for resolution
    """

    def __init__(
        self,
        message: str,
        spec: Spec | None = None,
        context: Context | None = None,
        cause: Exception | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Initialize RealisationError with debugging information.

        Parameters
        ----------
        message : str
            Primary error message
        spec : Spec | None
            Specification that failed realisation
        context : Context | None
            Context during realisation
        cause : Exception | None
            Underlying exception
        suggestion : str | None
            Helpful suggestion for resolution
        """
        self.spec = spec
        self.context = context
        self.cause = cause
        self.suggestion = suggestion

        parts = [message]
        if spec:
            parts.append(f"\nSpec: {spec}")
        if context:
            parts.append(f"\nContext: {context}")
        if cause:
            parts.append(f"\nCause: {type(cause).__name__}: {cause}")
        if suggestion:
            parts.append(f"\nSuggestion: {suggestion}")

        super().__init__("\n".join(parts))


class ModuleCache:
    """Cache for realised modules with LRU eviction.

    Caches realised modules to avoid redundant construction,
    with configurable size limits and eviction policies.

    Parameters
    ----------
    max_size : int
        Maximum number of cached modules
    enabled : bool
        Whether caching is enabled
    """

    def __init__(self, max_size: int = 128, enabled: bool = True) -> None:
        """Initialize module cache with LRU eviction.

        Parameters
        ----------
        max_size : int
            Maximum number of cached modules
        enabled : bool
            Whether caching is enabled
        """
        if hasattr(_thread_local, "config"):
            config = _get_config()
            default_size = config.constants.DEFAULT_CACHE_SIZE
        else:
            default_size = RealisationConstants.DEFAULT_CACHE_SIZE

        self.max_size = (
            max_size
            if max_size != RealisationConstants.DEFAULT_CACHE_SIZE
            else default_size
        )
        self.enabled = enabled
        self._cache: dict[tuple[Any, ...], nn.Module] = {}
        self._access_order: list[tuple[Any, ...]] = []
        self._hit_count = 0
        self._miss_count = 0

    def _make_key(self, spec: Spec, context: Context) -> tuple[Any, ...]:
        """Create a cache key using a hash-based approach."""
        if spec.__class__.__name__ == "ETBlockSpec":
            return ("nocache", id(spec))
        try:
            meta_str = json.dumps(context.metadata, sort_keys=True, default=str)
        except ValueError:
            meta_str = str(context.metadata)

        key_dict = {
            "spec_type": type(spec).__name__,
            "spec_params": self._serialize_spec(spec),
            "context_dims": dict(context.dimensions),
            "context_meta": meta_str,
            "version": getattr(self, "version", 1),
        }

        key_str = json.dumps(key_dict, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()
        return (type(spec).__name__, key_hash)

    def _serialize_spec(
        self, spec: Spec, max_depth: int = 10
    ) -> dict[str, Any]:
        """Serialize a spec to a dictionary for hashing."""
        if max_depth <= 0:
            return {"__truncated__": True}

        result: dict[str, Any] = {"__type__": type(spec).__name__}

        if hasattr(spec, "__dataclass_fields__"):
            for field_name, _field in spec.__dataclass_fields__.items():
                value = getattr(spec, field_name)
                if isinstance(value, Spec):
                    result[field_name] = self._serialize_spec(
                        value, max_depth - 1
                    )
                elif isinstance(value, list | tuple):
                    result[field_name] = [
                        self._serialize_spec(v, max_depth - 1)
                        if isinstance(v, Spec)
                        else v
                        for v in value
                    ]
                elif isinstance(value, dict):
                    result[field_name] = {
                        k: self._serialize_spec(v, max_depth - 1)
                        if isinstance(v, Spec)
                        else v
                        for k, v in value.items()
                    }
                elif hasattr(value, "__dict__"):
                    result[field_name] = str(value)
                else:
                    result[field_name] = value

        return result

    def get(self, spec: Spec, context: Context) -> nn.Module | None:
        """Get cached module if available.

        Parameters
        ----------
        spec : Spec
            Specification to look up
        context : Context
            Realisation context

        Returns
        -------
        nn.Module | None
            Cached module or None
        """
        if not self.enabled:
            return None

        key = self._make_key(spec, context)
        if key in self._cache:
            self._hit_count += 1
            # Update access order for LRU
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]

        self._miss_count += 1
        return None

    def put(self, spec: Spec, context: Context, module: nn.Module) -> None:
        """Cache a realised module.

        Parameters
        ----------
        spec : Spec
            Specification that was realised
        context : Context
            Realisation context
        module : nn.Module
            Realised module
        """
        if not self.enabled:
            return

        key = self._make_key(spec, context)

        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = module
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
        self._hit_count = 0
        self._miss_count = 0

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0


class RealiserPlugin(Protocol):
    """Protocol for realiser plugins.

    Plugins extend the realisation system to support custom
    specifications without modifying core code.
    """

    def can_realise(self, spec: Spec) -> bool:
        """Check if this plugin can realise the given spec.

        Parameters
        ----------
        spec : Spec
            Specification to check

        Returns
        -------
        bool
            True if plugin can handle this spec
        """
        ...

    def realise(self, spec: Spec, context: Context) -> nn.Module:
        """Realise the spec into a module.

        Parameters
        ----------
        spec : Spec
            Specification to realise
        context : Context
            Realisation context

        Returns
        -------
        nn.Module
            Realised module
        """
        ...


@dataclass
class RealiserConfig:
    """Configuration for the realisation system.

    Controls caching, validation, plugin loading, and other
    realisation behaviors.
    """

    cache: ModuleCache = field(default_factory=ModuleCache)
    plugins: list[RealiserPlugin] = field(default_factory=list)
    strict: bool = True
    warnings: bool = True
    auto_import: bool = True
    optimizations: bool = True
    max_recursion: int = 100  # Keep for backward compatibility
    constants: RealisationConstants = field(
        default_factory=RealisationConstants
    )
    metrics_collector: MetricsCollector = field(
        default_factory=MetricsCollector
    )
    debug_tracer: "DebugTracer | None" = None  # noqa: UP037


# Thread-local configuration
_thread_local = threading.local()


def _get_config() -> RealiserConfig:
    """Get thread-local configuration instance."""
    if not hasattr(_thread_local, "config"):
        _thread_local.config = RealiserConfig()
    return cast(RealiserConfig, _thread_local.config)


def configure_realisation(**kwargs: Any) -> None:
    """Configure the realisation system.

    Parameters
    ----------
    **kwargs : Any
        Configuration options to set. Can include:
        - cache: ModuleCache instance
        - strict: bool for strict validation
        - warnings: bool for warning messages
        - auto_import: bool for automatic imports
        - optimizations: bool for spec optimizations
        - max_recursion: int for recursion limit
        - constants: RealisationConstants for all constants
        - enable_metrics: bool to enable metrics collection

    Raises
    ------
    ValueError
        If unknown configuration option
    """
    config = _get_config()

    if "enable_metrics" in kwargs:
        config.metrics_collector.enabled = kwargs.pop("enable_metrics")

    # Handle constants specially
    if "constants" in kwargs:
        config.constants = kwargs.pop("constants")

    constant_overrides = {}
    for key in list(kwargs.keys()):
        if hasattr(config.constants, key):
            constant_overrides[key] = kwargs.pop(key)
    if "max_recursion" in kwargs:
        constant_overrides["MAX_RECURSION"] = kwargs.pop("max_recursion")

    if constant_overrides:
        current = config.constants
        new_constants = RealisationConstants(
            **{k: getattr(current, k) for k in current.__dataclass_fields__}
        )
        for k, v in constant_overrides.items():
            setattr(new_constants, k, v)
        config.constants = new_constants

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration option: {key}")


class Realiser:
    """Main realiser with plugin and optimization support.

    Converts specifications into PyTorch modules, handling
    dimension propagation, caching, and plugin dispatch.

    Parameters
    ----------
    context : Context, optional
        Initial realisation context
    _recursion_depth : int, optional
        Current recursion depth (for internal use)
    """

    def __init__(
        self,
        context: Context | None = None,
        _recursion_depth: int = 0,
    ) -> None:
        """Initialize realiser with optional context.

        Parameters
        ----------
        context : Context | None
            Initial realisation context
        _recursion_depth : int
            Current recursion depth for tracking
        """
        self.context = context or Context()
        self._realiser_stack: list[Spec] = []
        self._recursion_depth = _recursion_depth

        # Dispatch table for built-in combinators
        self._builtin_realisers: dict[type, Callable[[Any], nn.Module]] = {
            Sequential: self._realise_sequential,
            Parallel: self._realise_parallel,
            Conditional: self._realise_conditional,
            Residual: self._realise_residual,
            Graph: self._realise_graph,
            Loop: self._realise_loop,
            Switch: self._realise_switch,
            Identity: lambda _: nn.Identity(),
            Lambda: lambda spec: LambdaModule(spec.fn, spec.name),
        }

    def realise(self, spec: Spec) -> nn.Module:  # noqa: C901, PLR0912, PLR0915
        """Realise a spec into a PyTorch module with metrics and debugging."""
        config = _get_config()
        metrics = config.metrics_collector
        tracer = config.debug_tracer

        if tracer:
            tracer.trace_enter(spec, self._recursion_depth, self.context)

        start_time = time.perf_counter() if metrics.enabled else 0

        with metrics.timer("cache_lookup_time"):
            cached = config.cache.get(spec, self.context)

        if cached:
            metrics.record_cache_hit()
            if tracer:
                tracer.trace_cache_hit(spec, self._recursion_depth)
            if metrics.enabled:
                config.metrics_collector._metrics.total_time += (
                    time.perf_counter() - start_time
                )
            return cached

        metrics.record_cache_miss()
        metrics.update_max_depth(self._recursion_depth)

        if config.optimizations:
            spec = self._optimize_spec(spec)
            if cached := config.cache.get(spec, self.context):
                metrics.record_cache_hit()
                if tracer:
                    tracer.trace_cache_hit(spec, self._recursion_depth)
                if metrics.enabled:
                    config.metrics_collector._metrics.total_time += (
                        time.perf_counter() - start_time
                    )
                return cached

        # Enforce recursion limit only for uncached specs
        if self._recursion_depth >= config.constants.MAX_RECURSION:
            stack_summary = self._get_stack_summary()
            raise RealisationError(
                f"Maximum recursion depth ({config.constants.MAX_RECURSION}) exceeded",
                spec=spec,
                context=self.context,
                suggestion=(
                    f"Current stack depth: {self._recursion_depth}\n"
                    f"Stack summary: {stack_summary}\n"
                    "Consider:\n"
                    "1. Increasing max_recursion in configure_realisation()\n"
                    "2. Using loop() instead of deep nesting\n"
                    "3. Checking for circular dependencies"
                ),
            )

        # Detect cycles
        if spec in self._realiser_stack:
            cycle_path = self._get_cycle_path(spec)
            raise RealisationError(
                "Circular dependency detected",
                spec=spec,
                context=self.context,
                suggestion=f"Dependency cycle: {' -> '.join(cycle_path)}",
            )

        # Track stack and depth
        self._realiser_stack.append(spec)
        self._recursion_depth += 1

        try:
            with metrics.spec_timer(spec.__class__.__name__):
                module = self._realise_impl(spec)

            config.cache.put(spec, self.context, module)

            if tracer:
                tracer.trace_exit(spec, self._recursion_depth, module)
        except Exception as e:
            if tracer:
                tracer.trace_error(spec, self._recursion_depth, e)
            if not isinstance(e, RealisationError):
                raise RealisationError(
                    f"Realisation failed: {type(e).__name__}: {e}",
                    spec=spec,
                    context=self.context,
                    cause=e,
                    suggestion=f"Error at depth {self._recursion_depth}",
                ) from e
            e.suggestion = (
                f"{e.suggestion}\nFailed at depth {self._recursion_depth}"
                if e.suggestion
                else f"Failed at depth {self._recursion_depth}"
            )
            raise
        else:
            return module
        finally:
            self._realiser_stack.pop()
            self._recursion_depth -= 1
            if metrics.enabled:
                config.metrics_collector._metrics.total_time += (
                    time.perf_counter() - start_time
                )

    def _get_stack_summary(self) -> str:
        """Get a brief summary of the current realisation stack."""
        if not self._realiser_stack:
            return "Empty"

        config = _get_config()
        recent = self._realiser_stack[-config.constants.MAX_STACK_PREVIEW :]
        summary = " -> ".join(spec.__class__.__name__ for spec in recent)
        if len(self._realiser_stack) > config.constants.MAX_STACK_PREVIEW:
            summary = f"... ({len(self._realiser_stack) - config.constants.MAX_STACK_PREVIEW} more) -> {summary}"
        return summary

    def _get_cycle_path(self, target_spec: Spec) -> list[str]:
        """Return the dependency cycle path for ``target_spec``."""
        try:
            start_idx = self._realiser_stack.index(target_spec)
            cycle_specs = self._realiser_stack[start_idx:] + [target_spec]
            return [spec.__class__.__name__ for spec in cycle_specs]
        except ValueError:  # pragma: no cover - should not happen
            return ["Unknown cycle"]

    def _realise_impl(self, spec: Spec) -> nn.Module:
        """Implement realisation logic."""
        config = _get_config()
        # Try registered realisers first
        module = self._try_registered_realiser(spec)
        if module is not None:
            return module

        # Handle built-in combinators
        module = self._try_builtin_realiser(spec)
        if module is not None:
            return module

        # Try plugins
        module = self._try_plugin_realiser(spec)
        if module is not None:
            return module

        # Try auto-import if enabled
        if config.auto_import:
            module = self._try_auto_import(spec)
            if module is not None:
                return module

        # No realiser found
        raise RealisationError(
            f"No realiser registered for {spec.__class__.__name__}",
            spec=spec,
            context=self.context,
            suggestion=f"Use @register({spec.__class__.__name__}) to "
            f"register a realiser function",
        )

    def _optimize_spec(self, spec: Spec) -> Spec:
        """Apply optimization passes to specification."""
        # Simple optimizations
        # 1. Remove identity nodes in sequential
        if isinstance(spec, Sequential):
            parts = [p for p in spec.parts if not isinstance(p, Identity)]
            if len(parts) != len(spec.parts):
                spec = Sequential(parts=tuple(parts))

        # 2. Flatten nested sequentials
        if isinstance(spec, Sequential):
            parts = []
            for part in spec.parts:
                if isinstance(part, Sequential):
                    parts.extend(part.parts)
                else:
                    parts.append(part)
            spec = Sequential(parts=tuple(parts))

        return spec

    def _try_registered_realiser(self, spec: Spec) -> nn.Module | None:
        """Try to realise using registered realiser."""
        if realiser_fn := SpecMeta.get_realiser(spec.__class__):
            try:
                return realiser_fn(spec, self.context)
            except Exception as e:
                raise RealisationError(
                    f"Realiser failed for {spec.__class__.__name__}",
                    spec=spec,
                    context=self.context,
                    cause=e,
                ) from e
        return None

    def _try_builtin_realiser(self, spec: Spec) -> nn.Module | None:
        """Try to realise using built-in combinator realisers."""
        spec_type = type(spec)
        if realiser_fn := self._builtin_realisers.get(spec_type):
            return realiser_fn(spec)
        return None

    def _try_plugin_realiser(self, spec: Spec) -> nn.Module | None:
        """Try to realise using plugins."""
        config = _get_config()
        for plugin in config.plugins:
            if plugin.can_realise(spec):
                try:
                    return plugin.realise(spec, self.context)
                except Exception as e:  # noqa: BLE001
                    if config.warnings:
                        warnings.warn(
                            f"Plugin {plugin} failed for {spec}: {e}",
                            stacklevel=2,
                        )
        return None

    def _try_auto_import(self, spec: Spec) -> nn.Module | None:
        """Try to automatically import and instantiate a module."""
        config = _get_config()
        importer = AutoImporter(self.context, config.warnings)
        return importer.try_import(spec)

    def _realise_sequential(self, spec: Sequential) -> nn.Module:
        """Realise sequential composition."""
        modules = []
        current_context = self.context

        for i, part in enumerate(spec.parts):
            # Create child realiser with updated context and inherited depth
            child_realiser = Realiser(current_context, self._recursion_depth)
            try:
                module = child_realiser.realise(part)
                modules.append(module)
                # Update context for next part
                current_context = part.apply_context(current_context)
            except RealisationError as e:
                raise RealisationError(
                    f"Failed to realise part {i} of sequential",
                    spec=spec,
                    context=self.context,
                    cause=e,
                ) from e

        return nn.Sequential(*modules)

    def _realise_parallel(self, spec: Parallel) -> nn.Module:
        """Realise parallel composition."""
        branches = []
        for i, branch in enumerate(spec.branches):
            # Each branch gets independent context but inherits depth
            child_realiser = Realiser(
                self.context.child(),
                self._recursion_depth,
            )
            try:
                module = child_realiser.realise(branch)
                branches.append(module)
            except RealisationError as e:
                raise RealisationError(
                    f"Failed to realise branch {i} of parallel",
                    spec=spec,
                    context=self.context,
                    cause=e,
                ) from e

        return ParallelModule(branches, spec.merge, spec.weights)

    def _realise_conditional(self, spec: Conditional) -> nn.Module:
        """Realise conditional execution."""
        # Evaluate condition at realisation time
        if spec.condition(self.context):
            return self.realise(spec.if_true)
        if spec.if_false:
            return self.realise(spec.if_false)
        return nn.Identity()

    def _realise_residual(self, spec: Residual) -> nn.Module:
        """Realise residual connection."""
        inner = self.realise(spec.inner)
        return ResidualModule(inner, spec.merge, spec.scale)

    def _realise_graph(self, spec: Graph) -> nn.Module:
        """Realise graph structure into executable module."""
        nodes: dict[str, nn.Module] = {}
        for name, node_spec in spec.nodes.items():
            node_context = self.context.child()
            child_realiser = Realiser(node_context, self._recursion_depth)
            try:
                nodes[name] = child_realiser.realise(node_spec)
            except RealisationError as e:
                raise RealisationError(
                    f"Failed to realise graph node '{name}'",
                    spec=spec,
                    context=self.context,
                    cause=e,
                    suggestion=(
                        f"Check node '{name}' specification and context requirements"
                    ),
                ) from e

        return GraphModule(
            nodes=nodes,
            edges=spec.edges,
            inputs=spec.inputs,
            outputs=spec.outputs,
        )

    def _realise_loop(self, spec: Loop) -> nn.Module:
        """Realise loop structure with proper cache handling."""
        # Resolve loop count from context if needed
        if isinstance(spec.times, str):
            times = self.context.get_dim(spec.times)
            if times is None:
                raise RealisationError(
                    f"Unknown dimension for loop count: {spec.times}",
                    spec=spec,
                    context=self.context,
                    suggestion=(
                        f"Available dimensions: {list(self.context.dimensions.keys())}"
                    ),
                )
        else:
            times = spec.times

        if times <= 0:
            raise RealisationError(
                f"Invalid loop count: {times}",
                spec=spec,
                context=self.context,
                suggestion="Loop count must be positive",
            )

        from . import library  # Local import to avoid circular dependency

        config = _get_config()

        if (
            not spec.unroll
            and isinstance(spec.body, library.ETBlockSpec)
            and isinstance(times, int)
            and times <= config.constants.UNROLL_LIMIT
        ):
            return self._realise_unrolled_independent(spec, times)

        if spec.unroll:
            if spec.share_weights:
                body = self.realise(spec.body)
                modules = [body for _ in range(times)]
                return nn.Sequential(*modules)
            return self._realise_unrolled_independent(spec, times)
        body = self.realise(spec.body)
        return LoopModule(body, times)

    def _realise_unrolled_independent(
        self,
        spec: Loop,
        times: int,
    ) -> nn.Module:
        """Realise unrolled loop with independent weights."""
        config = _get_config()
        bodies: list[nn.Module] = []
        original_cache_enabled = config.cache.enabled
        cache_error: Exception | None = None

        try:
            config.cache.enabled = False
            for i in range(times):
                try:
                    child_realiser = Realiser(
                        self.context.child(),
                        self._recursion_depth,
                    )
                    child_realiser.context.metadata["loop_iteration"] = i
                    body = child_realiser.realise(spec.body)
                    bodies.append(body)
                except Exception as e:
                    if isinstance(e, RealisationError):
                        e.suggestion = (
                            f"Failed at loop iteration {i + 1}/{times}\n{e.suggestion}"
                            if e.suggestion
                            else f"Failed at loop iteration {i + 1}/{times}"
                        )
                        raise
                    raise RealisationError(
                        f"Loop iteration {i + 1}/{times} failed",
                        spec=spec.body,
                        context=self.context,
                        cause=e,
                    ) from e
        except Exception as e:  # noqa: BLE001
            cache_error = e
        finally:
            try:
                config.cache.enabled = original_cache_enabled
            except Exception as restore_error:
                import logging

                logger = logging.getLogger(__name__)
                logger.critical(
                    "Failed to restore cache state: %s",
                    restore_error,
                    exc_info=True,
                )
                if cache_error:
                    raise RuntimeError(
                        "Multiple errors: cache restore failed after realisation error",
                    ) from cache_error
                raise

        if cache_error:
            raise cache_error

        return nn.Sequential(*bodies)

    def _realise_switch(self, spec: Switch) -> nn.Module:
        """Realise switch structure."""
        # Evaluate switch at realisation time
        if callable(spec.key):
            key_value = spec.key(self.context)
        else:
            # Look in metadata first, then dimensions
            key_value = self.context.metadata.get(spec.key)
            if key_value is None:
                key_value = self.context.get_dim(spec.key)

        if key_value in spec.cases:
            return self.realise(spec.cases[key_value])
        if spec.default:
            return self.realise(spec.default)
        return nn.Identity()


# Module implementations


class ParallelModule(nn.Module):  # type: ignore[misc]
    """Module for parallel execution with merging.

    Executes multiple branches in parallel and combines outputs
    according to the specified merge strategy.
    """

    def __init__(
        self,
        branches: list[nn.Module],
        merge: str = "concat",
        weights: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.merge = merge
        self.weights = weights

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Execute branches and merge outputs."""
        outputs: list[torch.Tensor] = [branch(x) for branch in self.branches]

        if self.merge == "concat":
            return torch.cat(outputs, dim=-1)
        if self.merge == "add":
            if self.weights:
                result = self.weights[0] * outputs[0]
                for w, out in zip(self.weights[1:], outputs[1:], strict=False):
                    result = result + w * out
                return result
            result = outputs[0]
            for out in outputs[1:]:
                result = result + out
            return result
        if self.merge == "multiply":
            result = outputs[0]
            for out in outputs[1:]:
                result = result * out
            return result
        if self.merge == "mean":
            return torch.stack(outputs).mean(dim=0)
        if self.merge == "max":
            return torch.stack(outputs).max(dim=0)[0]
        raise ValueError(f"Unknown merge mode: {self.merge}")


class ResidualModule(nn.Module):  # type: ignore[misc]
    """Module for residual connections.

    Wraps a module with residual connection and flexible merging.
    """

    def __init__(
        self,
        inner: nn.Module,
        merge: str = "add",
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.merge = merge
        self.scale = scale

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Apply inner module with residual connection."""
        residual = x
        out: torch.Tensor = self.inner(x)

        if self.merge == "add":
            return residual + self.scale * out
        if self.merge == "concat":
            return torch.cat([residual, out], dim=-1)
        if self.merge == "gate":
            # Learned gating would require parameters
            # For now, use simple average gating
            gate = torch.sigmoid(out.mean(dim=-1, keepdim=True))
            result: torch.Tensor = residual * (1 - gate) + out * gate
            return result
        raise ValueError(f"Unknown merge mode: {self.merge}")


class GraphModule(nn.Module):  # type: ignore[misc]
    """Module for graph execution.

    Executes a computation graph with named nodes and edges.
    """

    def __init__(
        self,
        nodes: dict[str, nn.Module],
        edges: list[tuple[str, str, str | None]],
        inputs: list[str],
        outputs: list[str],
    ) -> None:
        super().__init__()
        self.nodes = nn.ModuleDict(nodes)
        self.edges = edges
        self.inputs = inputs
        self.outputs = outputs

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Execute graph computation with proper data flow.

        This method executes nodes in topological order, ensuring each node
        receives the correct inputs based on the graph edges.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor from the specified output nodes

        Raises
        ------
        RuntimeError
            If graph execution fails or outputs are not available
        """
        adjacency, in_degree, incoming_edges = self._prepare_graph()
        values = self._collect_input_values(x)
        order = self._topological_sort(adjacency, in_degree)
        self._execute_nodes(order, incoming_edges, values)
        return self._gather_outputs(values)

    def _prepare_graph(
        self,
    ) -> tuple[
        defaultdict[str, list[tuple[str, str | None]]],
        defaultdict[str, int],
        defaultdict[str, list[tuple[str, str | None]]],
    ]:
        """Construct adjacency and dependency data for the graph."""
        from collections import defaultdict

        config = _get_config()

        adjacency: defaultdict[str, list[tuple[str, str | None]]] = defaultdict(
            list,
        )
        in_degree: defaultdict[str, int] = defaultdict(int)
        incoming_edges: defaultdict[str, list[tuple[str, str | None]]] = (
            defaultdict(list)
        )

        for edge in self.edges:
            source = edge[0]
            target = edge[1]
            transform = (
                edge[2]
                if len(edge) == config.constants.FULL_EDGE_SIZE
                else None
            )

            if source in self.nodes or source in self.inputs:
                adjacency[source].append((target, transform))

            if target in self.nodes:
                incoming_edges[target].append((source, transform))
                if source in self.nodes:
                    in_degree[target] += 1

        return adjacency, in_degree, incoming_edges

    def _collect_input_values(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Gather values for graph input nodes."""
        values: dict[str, torch.Tensor] = {}

        if isinstance(x, dict):
            for name, tensor in x.items():
                if name in self.inputs:
                    values[name] = tensor
        else:
            for input_name in self.inputs:
                values[input_name] = x

        return values

    def _topological_sort(
        self,
        adjacency: defaultdict[str, list[tuple[str, str | None]]],
        in_degree: defaultdict[str, int],
    ) -> list[str]:
        """Return nodes in topological execution order."""
        queue = deque([n for n in self.nodes if in_degree[n] == 0])
        order: list[str] = []

        while queue:
            current = queue.popleft()
            order.append(current)
            for target, _ in adjacency.get(current, []):
                if target in self.nodes:
                    in_degree[target] -= 1
                    if in_degree[target] == 0:
                        queue.append(target)

        if len(order) != len(self.nodes):
            unprocessed = set(self.nodes) - set(order)
            raise RuntimeError(
                f"Graph contains cycles or unreachable nodes: {unprocessed}",
            )

        return order

    def _execute_nodes(
        self,
        order: list[str],
        incoming_edges: defaultdict[str, list[tuple[str, str | None]]],
        values: dict[str, torch.Tensor],
    ) -> None:
        """Run graph nodes following the resolved order."""
        for node_name in order:
            node_inputs: list[torch.Tensor] = []
            for source, transform in incoming_edges[node_name]:
                if source not in values:
                    raise RuntimeError(
                        f"Input '{source}' not available for node '{node_name}'. "
                        f"Available values: {list(values.keys())}",
                    )
                value = values[source]
                if transform is not None:
                    value = self._apply_edge_transform(value, transform)
                node_inputs.append(value)

            if not node_inputs:
                raise RuntimeError(f"Node '{node_name}' has no inputs")
            if len(node_inputs) == 1:
                values[node_name] = self.nodes[node_name](node_inputs[0])
            else:
                combined = torch.cat(node_inputs, dim=-1)
                values[node_name] = self.nodes[node_name](combined)

    def _gather_outputs(
        self,
        values: dict[str, torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Collect tensors for the output nodes."""
        output_tensors = []
        for output_name in self.outputs:
            if output_name not in values:
                raise RuntimeError(
                    f"Output '{output_name}' not computed. "
                    f"Available values: {list(values.keys())}",
                )
            output_tensors.append(values[output_name])

        if len(output_tensors) == 0:
            raise RuntimeError("No outputs specified for graph")
        if len(output_tensors) == 1:
            return output_tensors[0]
        return tuple(output_tensors)

    def _apply_edge_transform(  # noqa: C901, PLR0911, PLR0912
        self,
        tensor: torch.Tensor,
        transform: str,
    ) -> torch.Tensor:
        """Apply transformation to tensor safely without eval()."""
        if not transform:
            return tensor

        if transform == "detach":
            return tensor.detach()

        if transform.startswith("[") and transform.endswith("]"):
            index_str = transform[1:-1]
            if index_str.isdigit():
                return tensor[int(index_str)]
            if index_str == "...":
                return tensor[...]
            if ":" in index_str:
                parts = index_str.split(":")
                if len(parts) == 2:  # noqa: PLR2004
                    start = int(parts[0]) if parts[0] else None
                    end = int(parts[1]) if parts[1] else None
                    return tensor[start:end]
                if len(parts) == 3:  # noqa: PLR2004
                    start = int(parts[0]) if parts[0] else None
                    end = int(parts[1]) if parts[1] else None
                    step = int(parts[2]) if parts[2] else None
                    return tensor[start:end:step]
            if "," in index_str:
                indices: list[Any] = []
                for idx in index_str.split(","):
                    part = idx.strip()
                    if part == "...":
                        indices.append(...)
                    elif part.isdigit():
                        indices.append(int(part))
                    elif ":" in part:
                        slice_parts = part.split(":")
                        if len(slice_parts) == 2:  # noqa: PLR2004
                            start = (
                                int(slice_parts[0]) if slice_parts[0] else None
                            )
                            end = (
                                int(slice_parts[1]) if slice_parts[1] else None
                            )
                            indices.append(slice(start, end))
                        elif len(slice_parts) == 3:  # noqa: PLR2004
                            start = (
                                int(slice_parts[0]) if slice_parts[0] else None
                            )
                            end = (
                                int(slice_parts[1]) if slice_parts[1] else None
                            )
                            step = (
                                int(slice_parts[2]) if slice_parts[2] else None
                            )
                            indices.append(slice(start, end, step))
                        else:
                            raise ValueError(f"Invalid index format: {part}")
                    else:
                        raise ValueError(f"Invalid index format: {part}")
                return tensor[tuple(indices)]
            raise ValueError(f"Unsupported indexing format: {transform}")

        transform_registry: dict[
            str, Callable[[torch.Tensor], torch.Tensor]
        ] = {
            "sigmoid": torch.sigmoid,
            "relu": torch.relu,
            "tanh": torch.tanh,
            "softmax": lambda t: torch.softmax(t, dim=-1),
            "abs": torch.abs,
            "neg": torch.neg,
            "normalize": lambda t: torch.nn.functional.normalize(t, dim=-1),
            "stop_gradient": torch.Tensor.detach,
        }

        if transform in transform_registry:
            func = transform_registry[transform]
            return func(tensor)

        raise ValueError(f"Unknown transformation: {transform}")


class LoopModule(nn.Module):  # type: ignore[misc]
    """Module for dynamic loops.

    Applies a module multiple times in sequence.
    """

    def __init__(self, body: nn.Module, times: int) -> None:
        super().__init__()
        self.body = body
        self.times = times

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply body module multiple times."""
        for _ in range(self.times):
            x = self.body(x)
        return x


class LambdaModule(nn.Module):  # type: ignore[misc]
    """Module for lambda functions.

    Wraps a custom function as a module.
    """

    Fn = Callable[[torch.Tensor, Context], torch.Tensor]

    def __init__(self, fn: Fn, name: str = "lambda") -> None:
        """Initialize lambda module with function and name.

        Parameters
        ----------
        fn : Callable[[torch.Tensor, Context], torch.Tensor]
            Function to wrap
        name : str
            Name for the lambda module
        """
        super().__init__()
        self.fn = fn
        self._name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply lambda function."""
        # For simplicity, create empty context
        return self.fn(x, Context())

    def __repr__(self) -> str:
        return f"LambdaModule({self._name})"


# Public API


def realise(
    spec: Spec,
    context: Context | None = None,
    **context_updates: Any,
) -> nn.Module:
    """Realise a specification into a PyTorch module.

    Parameters
    ----------
    spec : Spec
        The specification to realise
    context : Context, optional
        Initial context for realisation
    **context_updates : Any
        Additional context dimensions to set

    Returns
    -------
    nn.Module
        The realised PyTorch module

    Raises
    ------
    ValidationError
        If specification validation fails
    RealisationError
        If realisation fails

    Examples
    --------
    >>> spec = seq(PatchEmbedSpec(...), CLSTokenSpec(), ETSpec())
    >>> model = realise(spec, embed_dim=768, num_heads=12)
    """
    if context is None:
        context = Context()

    # Apply context updates
    for key, value in context_updates.items():
        context.set_dim(key, value)

    # Validate spec first
    config = _get_config()
    if config.strict:
        issues = spec.validate(context)
        if issues:
            raise ValidationError(
                "Spec validation failed",
                spec=spec,
                context=context,
                suggestion="\n".join(issues),
            )

    # Realise
    realiser = Realiser(context)
    return realiser.realise(spec)


def register(spec_cls: type[Spec]) -> Callable[[Any], Any]:
    """Register a realiser function for a spec class.

    Parameters
    ----------
    spec_cls : type[Spec]
        Specification class to register for

    Returns
    -------
    Callable
        Decorator function

    Examples
    --------
    >>> @register(MySpec)
    ... def realise_my_spec(spec: MySpec, context: Context) -> nn.Module:
    ...     return MyModule(spec.param1, spec.param2)
    """
    return SpecMeta.register_realiser(spec_cls)


def register_typed(
    fn: Callable[[Any, Context], nn.Module],
) -> Callable[[Any, Context], nn.Module]:
    """Register a realiser using type hints.

    Parameters
    ----------
    fn : Callable
        Realiser function with type hints

    Returns
    -------
    Callable
        The same function

    Examples
    --------
    >>> @register_typed
    ... def realise_my_spec(spec: MySpec, context: Context) -> nn.Module:
    ...     return MyModule(spec.param1)
    """
    hints = get_type_hints(fn)
    spec_type = next(iter(hints.values()))
    SpecMeta._realisers[spec_type] = fn
    return fn


def visualize(spec: Spec, _out_format: str = "svg") -> str:
    """Generate visual representation of specification.

    Parameters
    ----------
    spec : Spec
        Specification to visualize
    out_format : str
        Output format (svg, png, dot)

    Returns
    -------
    str
        Visualization in requested format
    """
    # Simplified - would use graphviz or similar
    nodes = []
    edges = []

    def traverse(s: Spec, parent: str | None = None) -> str:
        node_id = f"{s.__class__.__name__}_{id(s)}"
        nodes.append(f'{node_id} [label="{s.__class__.__name__}"]')

        if parent:
            edges.append(f"{parent} -> {node_id}")

        for child in s.children():
            traverse(child, node_id)

        return node_id

    traverse(spec)

    dot = "digraph {\n"
    dot += "\n".join(nodes)
    dot += "\n"
    dot += "\n".join(edges)
    dot += "\n}"

    return dot


def optimize_spec(spec: Spec) -> Spec:
    """Apply optimization passes to specification.

    Parameters
    ----------
    spec : Spec
        Specification to optimize

    Returns
    -------
    Spec
        Optimized specification
    """
    # This would implement various optimization passes
    # For now, just delegate to realiser's optimization
    realiser = Realiser()
    return realiser._optimize_spec(spec)


def to_yaml(spec: Spec) -> str:
    """Serialize specification to YAML.

    Parameters.
    ----------
    spec : Spec
        Specification to serialize

    Returns
    -------
    str
        YAML representation
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required for YAML serialization. "
            "Install with: pip install PyYAML",
        ) from e
    return str(yaml.dump(spec.to_dict(), default_flow_style=False))


def from_yaml(yaml_str: str) -> Spec:
    """Load specification from YAML.

    Parameters.
    ----------
    yaml_str : str
        YAML representation

    Returns
    -------
    Spec
        Reconstructed specification
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required for YAML deserialization. "
            "Install with: pip install PyYAML",
        ) from e
    data = yaml.safe_load(yaml_str)
    return Spec.from_dict(data)


from . import library  # noqa: E402


@register(library.ETBlockSpec)
def realise_et_block(spec: library.ETBlockSpec, context: Context) -> nn.Module:
    """Realise ``ETBlockSpec`` into an :class:`EnergyTransformer`."""
    from energy_transformer.models import EnergyTransformer
    from energy_transformer.utils.optimizers import SGD

    realiser = Realiser(context)

    context = spec.layer_norm.apply_context(context)
    layer_norm = realiser.realise(spec.layer_norm)

    context = spec.attention.apply_context(context)
    attention = realiser.realise(spec.attention)

    context = spec.hopfield.apply_context(context)
    hopfield = realiser.realise(spec.hopfield)

    return EnergyTransformer(
        layer_norm=layer_norm,
        attention=attention,
        hopfield=hopfield,
        steps=spec.steps,
        optimizer=SGD(alpha=spec.alpha),
    )


@register(library.CLSTokenSpec)
def realise_cls_token(
    _spec: library.CLSTokenSpec,
    context: Context,
) -> nn.Module:
    """Realise a CLS token using context embed dimension."""
    embed_dim = context.get_dim("embed_dim")
    assert embed_dim is not None

    class _CLSToken(nn.Module):
        def __init__(self, embed_dim: int) -> None:
            super().__init__()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            return torch.cat([cls_tokens, x], dim=1)

    return _CLSToken(embed_dim)


@register(library.PosEmbedSpec)
def realise_pos_embed(
    spec: library.PosEmbedSpec,
    context: Context,
) -> nn.Module:
    """Realise positional embedding with context dimensions."""
    from energy_transformer.layers.embeddings import PosEmbed2D

    num_patches = context.get_dim("num_patches")
    if spec.include_cls:
        num_patches = (num_patches or 0) - 1
    embed_dim = context.get_dim("embed_dim")
    assert embed_dim is not None
    assert num_patches is not None
    return PosEmbed2D(
        num_patches,
        embed_dim,
        cls_token=spec.include_cls,
    )


@register(library.LayerNormSpec)
def realise_layer_norm(
    spec: library.LayerNormSpec,
    context: Context,
) -> nn.Module:
    """Realise layer normalization using custom implementation."""
    from energy_transformer.layers import EnergyLayerNorm

    embed_dim = context.get_dim("embed_dim")
    if embed_dim is None:
        raise RealisationError(
            "LayerNormSpec requires 'embed_dim' in context",
            spec=spec,
            context=context,
        )

    return EnergyLayerNorm(embed_dim, eps=spec.eps)


@register(library.ClassificationHeadSpec)
def realise_cls_head(
    spec: library.ClassificationHeadSpec,
    context: Context,
) -> nn.Module:
    """Realise classification head module."""
    from energy_transformer.layers.heads import ClassifierHead

    embed_dim = context.get_dim("embed_dim")
    assert embed_dim is not None
    return ClassifierHead(
        embed_dim,
        num_classes=spec.num_classes,
        pool_type=spec.pool_type,
        drop_rate=spec.drop_rate,
        use_conv=spec.use_conv,
        bias=spec.bias,
    )


@register(library.MHASpec)
def realise_mha(
    spec: library.MHASpec,
    context: Context,
) -> nn.Module:
    """Realise ``MHASpec`` into a standard ``nn.MultiheadAttention``."""
    embed_dim = context.get_dim("embed_dim")
    if embed_dim is None:
        raise RealisationError(
            "MHASpec requires 'embed_dim' in context",
            spec=spec,
            context=context,
        )

    return nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=spec.num_heads,
        dropout=spec.attn_drop,
        bias=spec.qkv_bias,
        batch_first=True,
    )


@register(library.MHEASpec)
def realise_mhea(
    spec: library.MHEASpec,
    context: Context,
) -> nn.Module:
    """Realise ``MHEASpec`` into :class:`MultiheadEnergyAttention`."""
    from energy_transformer.layers.attention import MultiheadEnergyAttention

    embed_dim = context.get_dim("embed_dim")
    if embed_dim is None:
        embed_dim = spec.num_heads * spec.head_dim

    if spec.bias:
        warnings.warn(
            "bias parameter is ignored in MultiheadEnergyAttention",
            stacklevel=2,
        )

    return MultiheadEnergyAttention(
        embed_dim=embed_dim,
        num_heads=spec.num_heads,
        beta=spec.beta,
        init_std=spec.init_std,
    )


@register(library.MLPSpec)
def realise_mlp(spec: library.MLPSpec, context: Context) -> nn.Module:
    """Realise MLP specification."""
    from energy_transformer.layers.mlp import MLP

    embed_dim = context.get_dim("embed_dim")
    if embed_dim is None:
        raise RealisationError(
            "MLPSpec requires 'embed_dim' in context",
            spec=spec,
            context=context,
        )

    hidden_features = spec.hidden_features or embed_dim * 4
    out_features = spec.out_features or embed_dim

    activation_map: dict[str, Callable[..., nn.Module]] = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "swish": nn.SiLU,
        "silu": nn.SiLU,
    }
    act_layer = activation_map.get(spec.activation, nn.GELU)

    return MLP(
        in_features=embed_dim,
        hidden_features=hidden_features,
        out_features=out_features,
        act_layer=act_layer,
        drop=spec.drop,
    )


@register(library.TransformerBlockSpec)
def realise_transformer_block(
    spec: library.TransformerBlockSpec,
    context: Context,
) -> nn.Module:
    """Realise standard transformer block."""
    from torch import nn

    realiser = Realiser(context)
    embed_dim_val = context.get_dim("embed_dim")
    if embed_dim_val is None:
        raise RealisationError(
            "TransformerBlockSpec requires 'embed_dim' in context",
            spec=spec,
            context=context,
        )
    embed_dim: int = embed_dim_val

    attention = realiser.realise(spec.attention)
    mlp = realiser.realise(spec.mlp)

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = attention
            self.mlp = mlp
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            if spec.norm_first:
                y = self.norm1(x)
                x = x + self.attn(y, y, y)[0]
                y = self.norm2(x)
                x = x + self.mlp(y)
                return x  # noqa: RET504

            x = self.attn(x, x, x)[0] + x
            x = self.norm1(x)
            x = self.mlp(x) + x
            x = self.norm2(x)
            return x  # noqa: RET504

    return _Block()
