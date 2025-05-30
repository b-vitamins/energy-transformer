"""Specification realisation system for creating PyTorch modules.

This module converts abstract specifications into concrete PyTorch modules
through a plugin-based architecture. The realisation system supports caching,
automatic module discovery, and extensibility through plugins.

The system maintains a clear separation between specification (what to build)
and realisation (how to build it), enabling multiple implementations of the
same specification.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, get_type_hints

import torch
import torch.nn as nn

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
from .primitives import Context, Spec, SpecMeta, ValidationError

if TYPE_CHECKING:
    pass

__all__ = [
    "realise",
    "Realiser",
    "register",
    "RealisationError",
    "ModuleCache",
    "RealiserPlugin",
    "configure_realisation",
    "visualize",
    "optimize_spec",
    "to_yaml",
    "from_yaml",
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
    ):
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

    def __init__(self, max_size: int = 128, enabled: bool = True):
        """Initialize module cache with LRU eviction.

        Parameters
        ----------
        max_size : int
            Maximum number of cached modules
        enabled : bool
            Whether caching is enabled
        """
        self.max_size = max_size
        self.enabled = enabled
        self._cache: dict[tuple[Any, ...], nn.Module] = {}
        self._access_order: list[tuple[Any, ...]] = []
        self._hit_count = 0
        self._miss_count = 0

    def _make_key(self, spec: Spec, context: Context) -> tuple[Any, ...]:
        """Create cache key from spec and context."""

        def make_hashable(obj: Any) -> Any:
            if isinstance(obj, dict):
                return tuple(
                    sorted((k, make_hashable(v)) for k, v in obj.items())
                )
            elif isinstance(obj, list):
                return tuple(make_hashable(item) for item in obj)
            elif isinstance(obj, set):
                return tuple(sorted(make_hashable(item) for item in obj))
            else:
                return obj

        spec_dict = spec.to_dict()
        spec_key = (spec.__class__.__name__, make_hashable(spec_dict))
        ctx_dims_key = tuple(sorted(context.dimensions.items()))
        ctx_meta_key = make_hashable(context.metadata)
        return (spec_key, ctx_dims_key, ctx_meta_key)

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
    max_recursion: int = 100


# Global configuration
_config = RealiserConfig()


def configure_realisation(**kwargs: Any) -> None:
    """Configure the realisation system.

    Parameters
    ----------
    **kwargs : Any
        Configuration options to set

    Raises
    ------
    ValueError
        If unknown configuration option
    """
    for key, value in kwargs.items():
        if hasattr(_config, key):
            setattr(_config, key, value)
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
        self, context: Context | None = None, _recursion_depth: int = 0
    ):
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

    def realise(self, spec: Spec) -> nn.Module:
        """Realise a spec into a PyTorch module.

        Parameters
        ----------
        spec : Spec
            Specification to realise

        Returns
        -------
        nn.Module
            Realised module

        Raises
        ------
        RealisationError
            If realisation fails
        """
        # Check recursion limit using depth counter
        if self._recursion_depth >= _config.max_recursion:
            raise RealisationError(
                "Maximum recursion depth exceeded",
                spec=spec,
                context=self.context,
                suggestion="Check for circular dependencies in specifications",
            )

        # Apply optimizations if enabled
        if _config.optimizations:
            spec = self._optimize_spec(spec)

        # Check cache
        if cached := _config.cache.get(spec, self.context):
            return cached

        if spec in self._realiser_stack:
            raise RealisationError(
                "Circular dependency detected",
                spec=spec,
                context=self.context,
            )

        self._realiser_stack.append(spec)
        # Increment depth for this realisation
        self._recursion_depth += 1
        try:
            module = self._realise_impl(spec)
            _config.cache.put(spec, self.context, module)
            return module
        finally:
            self._realiser_stack.pop()
            self._recursion_depth -= 1

    def _realise_impl(self, spec: Spec) -> nn.Module:
        """Implement realisation logic."""
        # Try registered realisers first
        if module := self._try_registered_realiser(spec):
            return module

        # Handle built-in combinators
        if module := self._try_builtin_realiser(spec):
            return module

        # Try plugins
        if module := self._try_plugin_realiser(spec):
            return module

        # Try auto-import if enabled
        if _config.auto_import:
            if module := self._try_auto_import(spec):
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
        for plugin in _config.plugins:
            if plugin.can_realise(spec):
                try:
                    return plugin.realise(spec, self.context)
                except Exception as e:
                    if _config.warnings:
                        warnings.warn(
                            f"Plugin {plugin} failed for {spec}: {e}",
                            stacklevel=2,
                        )
        return None

    def _try_auto_import(self, spec: Spec) -> nn.Module | None:
        """Try to automatically import and create module."""
        # Map spec names to module paths
        spec_name = spec.__class__.__name__

        # Common mappings
        module_mappings = {
            "LayerNormSpec": ("energy_transformer.layers", "LayerNorm"),
            "PatchEmbedSpec": (
                "energy_transformer.layers.embeddings",
                "PatchEmbedding",
            ),
            "CLSTokenSpec": ("energy_transformer.layers.tokens", "CLSToken"),
            "MHEASpec": (
                "energy_transformer.layers.attention",
                "MultiHeadEnergyAttention",
            ),
            "HNSpec": (
                "energy_transformer.layers.hopfield",
                "HopfieldNetwork",
            ),
        }

        mapping = module_mappings.get(spec_name)
        if mapping:
            module_path, class_name = mapping
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)

                # Extract constructor args from spec
                kwargs = {}
                for field_info in spec.__dataclass_fields__.values():
                    value = getattr(spec, field_info.name)
                    kwargs[field_info.name] = value

                instance = cls(**kwargs)
                assert isinstance(instance, nn.Module)
                return instance
            except Exception:
                pass

        return None

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
                self.context.child(), self._recursion_depth
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
        elif spec.if_false:
            return self.realise(spec.if_false)
        else:
            return nn.Identity()

    def _realise_residual(self, spec: Residual) -> nn.Module:
        """Realise residual connection."""
        inner = self.realise(spec.inner)
        return ResidualModule(inner, spec.merge, spec.scale)

    def _realise_graph(self, spec: Graph) -> nn.Module:
        """Realise graph structure."""
        nodes = {}
        for name, node_spec in spec.nodes.items():
            nodes[name] = self.realise(node_spec)

        return GraphModule(nodes, spec.edges, spec.inputs, spec.outputs)

    def _realise_loop(self, spec: Loop) -> nn.Module:
        """Realise loop structure."""
        # Resolve loop count
        if isinstance(spec.times, str):
            times = self.context.get_dim(spec.times)
            if times is None:
                raise RealisationError(
                    f"Unknown dimension for loop count: {spec.times}",
                    spec=spec,
                    context=self.context,
                )
        else:
            times = spec.times

        if spec.unroll:
            # Unrolled loop
            if spec.share_weights:
                # Share weights across iterations
                body = self.realise(spec.body)
                modules = [body for _ in range(times)]  # Same instance
                return nn.Sequential(*modules)
            else:
                # Independent weights - don't share module instances
                bodies = []
                # Temporarily disable caching to ensure fresh instances
                original_cache_enabled = _config.cache.enabled
                _config.cache.enabled = False
                try:
                    for _ in range(times):
                        # Create child realiser with inherited depth
                        child_realiser = Realiser(
                            self.context, self._recursion_depth
                        )
                        bodies.append(child_realiser.realise(spec.body))
                finally:
                    _config.cache.enabled = original_cache_enabled
                return nn.Sequential(*bodies)
        else:
            # Dynamic loop
            body = self.realise(spec.body)
            return LoopModule(body, times)

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
        elif spec.default:
            return self.realise(spec.default)
        else:
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
    ):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.merge = merge
        self.weights = weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute branches and merge outputs."""
        outputs: list[torch.Tensor] = [branch(x) for branch in self.branches]

        if self.merge == "concat":
            return torch.cat(outputs, dim=-1)
        elif self.merge == "add":
            if self.weights:
                result = self.weights[0] * outputs[0]
                for w, out in zip(self.weights[1:], outputs[1:], strict=False):
                    result = result + w * out
                return result
            else:
                result = outputs[0]
                for out in outputs[1:]:
                    result = result + out
                return result
        elif self.merge == "multiply":
            result = outputs[0]
            for out in outputs[1:]:
                result = result * out
            return result
        elif self.merge == "mean":
            return torch.stack(outputs).mean(dim=0)
        elif self.merge == "max":
            return torch.stack(outputs).max(dim=0)[0]
        else:
            raise ValueError(f"Unknown merge mode: {self.merge}")


class ResidualModule(nn.Module):  # type: ignore[misc]
    """Module for residual connections.

    Wraps a module with residual connection and flexible merging.
    """

    def __init__(
        self, inner: nn.Module, merge: str = "add", scale: float = 1.0
    ):
        super().__init__()
        self.inner = inner
        self.merge = merge
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply inner module with residual connection."""
        residual = x
        out: torch.Tensor = self.inner(x)

        if self.merge == "add":
            return residual + self.scale * out
        elif self.merge == "concat":
            return torch.cat([residual, out], dim=-1)
        elif self.merge == "gate":
            # Learned gating would require parameters
            # For now, use simple average gating
            gate = torch.sigmoid(out.mean(dim=-1, keepdim=True))
            result: torch.Tensor = residual * (1 - gate) + out * gate
            return result
        else:
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
    ):
        super().__init__()
        self.nodes = nn.ModuleDict(nodes)
        self.edges = edges
        self.inputs = inputs
        self.outputs = outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute graph computation."""
        # Build adjacency for topological sort
        in_degree: dict[str, int] = {name: 0 for name in self.nodes}
        adjacency: dict[str, list[tuple[str, str | None]]] = {
            name: [] for name in self.nodes
        }

        for edge in self.edges:
            # Handle both 2-tuple and 3-tuple edges
            if len(edge) == 2:
                from_node, to_node = edge
                transform = None
            else:
                from_node, to_node, transform = edge
            if to_node in self.nodes:
                in_degree[to_node] += 1
                if from_node in self.nodes:
                    adjacency[from_node].append((to_node, None))

        # Topological sort
        queue = [node for node, degree in in_degree.items() if degree == 0]
        topo_order = []

        while queue:
            node = queue.pop(0)
            topo_order.append(node)
            for next_node, _ in adjacency[node]:
                in_degree[next_node] -= 1
                if in_degree[next_node] == 0:
                    queue.append(next_node)

        # Execute in topological order
        values = {"input": x}
        for node in topo_order:
            # For simplicity, assume single input
            node_input = x
            values[node] = self.nodes[node](node_input)

        # Return first output
        return values.get(self.outputs[0], x) if self.outputs else x


class LoopModule(nn.Module):  # type: ignore[misc]
    """Module for dynamic loops.

    Applies a module multiple times in sequence.
    """

    def __init__(self, body: nn.Module, times: int):
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

    def __init__(self, fn: Fn, name: str = "lambda"):
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
    if _config.strict:
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


def visualize(spec: Spec, format: str = "svg") -> str:
    """Generate visual representation of specification.

    Parameters
    ----------
    spec : Spec
        Specification to visualize
    format : str
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
            "Install with: pip install PyYAML"
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
            "Install with: pip install PyYAML"
        ) from e
    data = yaml.safe_load(yaml_str)
    return Spec.from_dict(data)
