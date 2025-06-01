"""Specification combinators for composing model architectures.

This module provides composition operators for building complex model
architectures from primitive specifications. Combinators handle dimension
propagation, validation, and support various architectural patterns including
sequential, parallel, conditional, and graph-based compositions.

The combinator system enables building sophisticated architectures through
simple, composable operations while maintaining type safety and validation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, overload

from .primitives import Context, Spec, ValidationError

__all__ = [
    "Conditional",
    "Graph",
    "Identity",
    "Lambda",
    "Loop",
    "Parallel",
    "Residual",
    "Sequential",
    "Switch",
    "cond",
    "graph",
    "loop",
    "mixture_of_experts",
    "multi_scale",
    "parallel",
    "residual",
    "seq",
    "switch",
    "transformer_block",
]

T = TypeVar("T", bound=Spec)
S = TypeVar("S", bound=Spec)


@dataclass(frozen=True)
class Sequential(Spec):
    """Sequential composition of specifications.

    Executes specifications in order, with outputs flowing from one
    specification to the next. Supports operator chaining and provides
    full sequence functionality including indexing and slicing.

    Parameters
    ----------
    parts : tuple[Spec, ...]
        Specifications to execute in sequence
    """

    parts: tuple[Spec, ...] = field(default_factory=tuple)

    def __rshift__(self, other: Spec) -> Sequential:
        """Chain specs using >> operator.

        Parameters
        ----------
        other : Spec
            Specification to append

        Returns
        -------
        Sequential
            New sequence with appended spec
        """
        if isinstance(other, Sequential):
            return Sequential(parts=self.parts + other.parts)
        return Sequential(parts=(*self.parts, other))

    def __lshift__(self, other: Spec) -> Sequential:
        """Prepend spec using << operator.

        Parameters
        ----------
        other : Spec
            Specification to prepend

        Returns
        -------
        Sequential
            New sequence with prepended spec
        """
        if isinstance(other, Sequential):
            return Sequential(parts=other.parts + self.parts)
        return Sequential(parts=(other, *self.parts))

    def __or__(self, other: Spec) -> Parallel:
        """Create parallel composition using | operator.

        Parameters
        ----------
        other : Spec
            Specification to run in parallel

        Returns
        -------
        Parallel
            Parallel composition of self and other
        """
        return Parallel(branches=(self, other))

    def __getitem__(self, idx: int | slice) -> Spec | Sequential:
        """Index or slice the sequence.

        Parameters
        ----------
        idx : int | slice
            Index or slice to access

        Returns
        -------
        Spec | Sequential
            Single spec or subsequence
        """
        if isinstance(idx, slice):
            return Sequential(parts=self.parts[idx])
        return self.parts[idx]

    def __len__(self) -> int:
        """Return number of parts."""
        return len(self.parts)

    def __iter__(self) -> Iterator[Spec]:
        """Iterate over parts."""
        return iter(self.parts)

    def children(self) -> list[Spec]:
        """Return all parts as children."""
        return list(self.parts)

    def validate(self, context: Context) -> list[str]:
        """Validate sequence with context propagation.

        Parameters
        ----------
        context : Context
            Initial context

        Returns
        -------
        list[str]
            Validation issues
        """
        issues: list[str] = []
        current_context = context

        for i, part in enumerate(self.parts):
            part_issues = part.validate(current_context)
            issues.extend(f"Part {i}: {issue}" for issue in part_issues)
            current_context = part.apply_context(current_context)

        return issues

    def apply_context(self, context: Context) -> Context:
        """Apply all parts in sequence to context.

        Parameters
        ----------
        context : Context
            Initial context

        Returns
        -------
        Context
            Final context after all parts
        """
        for part in self.parts:
            context = part.apply_context(context)
        return context


@dataclass(frozen=True)
class Parallel(Spec):
    """Parallel composition with flexible merging strategies.

    Executes multiple specifications in parallel and combines their
    outputs according to the specified merge strategy. Supports
    weighted combinations and dimension-specific merging.

    Parameters
    ----------
    branches : tuple[Spec, ...]
        Specifications to execute in parallel
    merge : str
        How to combine outputs: "concat", "add", "multiply", "mean", "max"
    merge_dim : str, optional
        Dimension name for merge validation
    weights : tuple[float, ...], optional
        Weights for weighted merge strategies
    """

    branches: tuple[Spec, ...] = field(default_factory=tuple)
    merge: Literal["concat", "add", "multiply", "mean", "max"] = "concat"
    merge_dim: str | None = None
    weights: tuple[float, ...] | None = None

    def __or__(self, other: Spec) -> Parallel:
        """Add branch using | operator.

        Parameters
        ----------
        other : Spec
            Specification to add as branch

        Returns
        -------
        Parallel
            New parallel with added branch
        """
        if isinstance(other, Parallel) and other.merge == self.merge:
            return Parallel(
                branches=self.branches + other.branches,
                merge=self.merge,
                merge_dim=self.merge_dim,
                weights=self.weights,
            )
        return Parallel(
            branches=(*self.branches, other),
            merge=self.merge,
            merge_dim=self.merge_dim,
            weights=self.weights,
        )

    def children(self) -> list[Spec]:
        """Return all branches as children."""
        return list(self.branches)

    def validate(self, context: Context) -> list[str]:  # noqa: C901
        """Validate all branches and merge compatibility.

        Parameters
        ----------
        context : Context
            Validation context

        Returns
        -------
        list[str]
            Validation issues
        """
        issues = super().validate(context)

        # Validate each branch independently
        for i, branch in enumerate(self.branches):
            branch_issues = branch.validate(context)
            issues.extend(f"Branch {i}: {issue}" for issue in branch_issues)

        # Validate merge compatibility
        if self.merge in ["add", "multiply", "mean", "max"]:
            merge_dim_name = self.merge_dim or "embed_dim"
            dims = []

            for i, branch in enumerate(self.branches):
                branch_ctx = branch.apply_context(context.child())
                dim_value = branch_ctx.get_dim(merge_dim_name)

                if dim_value is None:
                    issues.append(
                        f"Branch {i} does not provide required dimension "
                        f"'{merge_dim_name}' for {self.merge} merge",
                    )
                else:
                    dims.append((i, dim_value))

            if dims:
                unique_dims = {d[1] for d in dims}
                if len(unique_dims) > 1:
                    dim_info = ", ".join(f"Branch {i}: {d}" for i, d in dims)
                    issues.append(
                        f"Incompatible dimensions for {self.merge} merge: {dim_info}. "
                        f"All branches must output the same dimension.",
                    )

        elif self.merge == "concat":
            if self.merge_dim:
                concat_dims: list[int] = []
                for _i, branch in enumerate(self.branches):
                    branch_ctx = branch.apply_context(context.child())
                    dim_value = branch_ctx.get_dim(self.merge_dim)
                    if dim_value:
                        concat_dims.append(dim_value)

                if concat_dims and max(concat_dims) > min(concat_dims) * 10:
                    issues.append(
                        f"Warning: Large dimension disparity in concat merge: "
                        f"min={min(concat_dims)}, max={max(concat_dims)}",
                    )

        # Validate weights if provided
        if self.weights:
            if len(self.weights) != len(self.branches):
                issues.append(
                    f"Weight count ({len(self.weights)}) doesn't match "
                    f"branch count ({len(self.branches)})",
                )

            if not all(isinstance(w, int | float) for w in self.weights):
                issues.append("All weights must be numeric")

            if self.merge == "add" and abs(sum(self.weights) - 1.0) > 1e-6:
                issues.append(
                    f"Warning: Weights sum to {sum(self.weights)}, not 1.0. "
                    f"This may cause unexpected scaling.",
                )

        return issues


@dataclass(frozen=True)
class Conditional(Spec):
    """Conditional specification execution.

    Executes different specifications based on a runtime condition
    evaluated against the context. Supports dynamic architecture
    selection and optional fallback branches.

    Parameters
    ----------
    condition : Callable[[Context], bool]
        Function to evaluate condition
    if_true : Spec
        Specification to use if condition is true
    if_false : Spec, optional
        Specification to use if condition is false
    """

    condition: Callable[[Context], bool]
    if_true: Spec
    if_false: Spec | None = None

    def children(self) -> list[Spec]:
        """Return both branches as children."""
        children = [self.if_true]
        if self.if_false:
            children.append(self.if_false)
        return children

    def validate(self, context: Context) -> list[str]:
        """Validate the branch that would execute.

        Parameters
        ----------
        context : Context
            Validation context

        Returns
        -------
        list[str]
            Validation issues
        """
        issues: list[str] = []

        if self.condition(context):
            issues.extend(self.if_true.validate(context))
        elif self.if_false:
            issues.extend(self.if_false.validate(context))

        return issues

    def apply_context(self, context: Context) -> Context:
        """Apply the appropriate branch to context.

        Parameters
        ----------
        context : Context
            Input context

        Returns
        -------
        Context
            Updated context
        """
        if self.condition(context):
            return self.if_true.apply_context(context)
        if self.if_false:
            return self.if_false.apply_context(context)
        return context


@dataclass(frozen=True)
class Residual(Spec):
    """Residual connection with flexible merging.

    Wraps a specification with a residual connection, supporting
    various merge strategies including gating and scaling.

    Parameters
    ----------
    inner : Spec
        Specification to wrap with residual
    merge : str
        Merge strategy: "add", "concat", "gate"
    gate_dim : str, optional
        Dimension for gating (required for gate merge)
    scale : float
        Scale factor for residual branch
    """

    inner: Spec
    merge: Literal["add", "concat", "gate"] = "add"
    gate_dim: str | None = None
    scale: float = 1.0

    def children(self) -> list[Spec]:
        """Return inner spec as child."""
        return [self.inner]

    def validate(self, context: Context) -> list[str]:
        """Validate inner spec and merge configuration.

        Parameters
        ----------
        context : Context
            Validation context

        Returns
        -------
        list[str]
            Validation issues
        """
        issues = super().validate(context)
        issues.extend(self.inner.validate(context))

        if self.merge == "gate" and not self.gate_dim:
            issues.append("Gate merge requires gate_dim")

        if self.scale <= 0:
            issues.append(f"Scale must be positive, got {self.scale}")

        return issues


@dataclass(frozen=True)
class Graph(Spec):
    """Graph-based composition for complex architectures.

    Represents computation as a directed graph where nodes are
    specifications and edges define data flow. Supports arbitrary
    DAG architectures with optional edge transformations.

    Parameters
    ----------
    nodes : dict[str, Spec]
        Named specification nodes
    edges : list[tuple[str, str, str | None]]
        Edges as (source, target, transform)
    inputs : list[str]
        Input node names
    outputs : list[str]
        Output node names
    """

    nodes: dict[str, Spec] = field(default_factory=dict)
    edges: list[tuple[str, str, str | None]] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate graph structure immediately after creation."""
        super().__post_init__()
        try:
            if self._has_cycle():
                cycle_path = self._find_cycle_path()
                raise ValidationError(
                    "Graph contains cycles",
                    spec=self,
                    suggestion=f"Remove cyclic dependency in path: {' -> '.join(cycle_path)}",
                )
        except Exception as e:  # noqa: BLE001
            raise ValidationError(
                f"Graph validation failed: {e}",
                spec=self,
                suggestion="Check graph edge definitions",
            ) from e

    def add_node(self, name: str, spec: Spec) -> Graph:
        """Add a node to the graph.

        Parameters
        ----------
        name : str
            Node name
        spec : Spec
            Node specification

        Returns
        -------
        Graph
            New graph with added node
        """
        nodes = dict(self.nodes)
        nodes[name] = spec
        return Graph(
            nodes=nodes,
            edges=self.edges,
            inputs=self.inputs,
            outputs=self.outputs,
        )

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        transform: str | None = None,
    ) -> Graph:
        """Add an edge to the graph.

        Parameters
        ----------
        from_node : str
            Source node name
        to_node : str
            Target node name
        transform : str, optional
            Edge transformation

        Returns
        -------
        Graph
            New graph with added edge
        """
        edges = list(self.edges)
        edges.append((from_node, to_node, transform))
        return Graph(
            nodes=self.nodes,
            edges=edges,
            inputs=self.inputs,
            outputs=self.outputs,
        )

    def children(self) -> list[Spec]:
        """Return all node specs as children."""
        return list(self.nodes.values())

    def validate(self, context: Context) -> list[str]:
        """Validate graph structure and all nodes.

        Parameters
        ----------
        context : Context
            Validation context

        Returns
        -------
        list[str]
            Validation issues
        """
        issues = super().validate(context)

        # Check graph connectivity
        for from_node, to_node, _ in self.edges:
            if from_node not in self.nodes and from_node not in self.inputs:
                issues.append(f"Unknown source node: {from_node}")
            if to_node not in self.nodes and to_node not in self.outputs:
                issues.append(f"Unknown target node: {to_node}")

        # Check for cycles
        if self._has_cycle():
            issues.append("Graph contains cycles")

        # Validate nodes
        for name, spec in self.nodes.items():
            node_issues = spec.validate(context)
            issues.extend(f"Node {name}: {issue}" for issue in node_issues)

        return issues

    def _has_cycle(self) -> bool:
        """Check if graph contains cycles using DFS."""
        visited = set()
        rec_stack = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for from_node, to_node, _ in self.edges:
                if from_node == node:
                    if to_node not in visited:
                        if dfs(to_node):
                            return True
                    elif to_node in rec_stack:
                        return True

            rec_stack.remove(node)
            return False

        return any(node not in visited and dfs(node) for node in self.nodes)

    def _find_cycle_path(self) -> list[str]:
        """Find a cycle path for error reporting."""
        visited = set()
        rec_stack = set()
        path: list[str] = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for from_node, to_node, _ in self.edges:
                if from_node == node and to_node in self.nodes:
                    if to_node not in visited:
                        if dfs(to_node):
                            return True
                    elif to_node in rec_stack:
                        cycle_start = path.index(to_node)
                        del path[:cycle_start]
                        path.append(to_node)
                        return True

            rec_stack.remove(node)
            path.pop()
            return False

        for node in self.nodes:
            if node not in visited and dfs(node):
                return path

        return []


@dataclass(frozen=True)
class Loop(Spec):
    """Repeated application with optional unrolling.

    Applies a specification multiple times, with support for
    static unrolling and weight sharing control.

    Parameters
    ----------
    body : Spec
        Specification to repeat
    times : int | str
        Number of iterations or dimension name
    unroll : bool
        Whether to unroll at compile time
    share_weights : bool
        Whether to share weights across iterations
    """

    body: Spec
    times: int | str
    unroll: bool = False
    share_weights: bool = True

    def children(self) -> list[Spec]:
        """Return body spec, potentially multiple times if unrolled."""
        if self.unroll and isinstance(self.times, int):
            return [self.body] * self.times
        return [self.body]

    def validate(self, context: Context) -> list[str]:
        """Validate loop configuration and body.

        Parameters
        ----------
        context : Context
            Validation context

        Returns
        -------
        list[str]
            Validation issues
        """
        issues = super().validate(context)

        # Resolve times if it's a dimension
        if isinstance(self.times, str):
            if context.get_dim(self.times) is None:
                issues.append(f"Unknown loop count dimension: {self.times}")
        elif self.times <= 0:
            issues.append(f"Loop count must be positive, got {self.times}")

        issues.extend(self.body.validate(context))
        return issues


@dataclass(frozen=True)
class Switch(Spec):
    """Multi-way conditional based on context value.

    Selects from multiple specifications based on a key value,
    with optional default case for unmatched keys.

    Parameters
    ----------
    key : str | Callable[[Context], Any]
        Key name or function to compute key
    cases : dict[Any, Spec]
        Mapping from key values to specifications
    default : Spec, optional
        Default specification for unmatched keys
    """

    key: str | Callable[[Context], Any]
    cases: dict[Any, Spec] = field(default_factory=dict)
    default: Spec | None = None

    def children(self) -> list[Spec]:
        """Return all case specs and default."""
        children = list(self.cases.values())
        if self.default:
            children.append(self.default)
        return children

    def validate(self, context: Context) -> list[str]:
        """Validate all cases.

        Parameters
        ----------
        context : Context
            Validation context

        Returns
        -------
        list[str]
            Validation issues
        """
        issues = super().validate(context)

        for key, spec in self.cases.items():
            case_issues = spec.validate(context)
            issues.extend(f"Case {key}: {issue}" for issue in case_issues)

        if self.default:
            default_issues = self.default.validate(context)
            issues.extend(f"Default: {issue}" for issue in default_issues)

        return issues


@dataclass(frozen=True)
class Identity(Spec):
    """Identity specification that passes through unchanged.

    Useful as a no-op placeholder or for conditional branches
    that should preserve the input without modification.
    """

    def children(self) -> list[Spec]:
        """Return empty list as identity has no children."""
        return []


@dataclass(frozen=True)
class Lambda(Spec):
    """Custom transformation specification.

    Wraps a custom function for use in specification pipelines,
    enabling arbitrary transformations while maintaining the
    specification interface.

    Parameters
    ----------
    fn : Callable[[Any, Context], Any]
        Transformation function
    name : str
        Descriptive name for the transformation
    """

    fn: Callable[[Any, Context], Any]
    name: str = "lambda"

    def children(self) -> list[Spec]:
        """Return empty list as lambda has no children."""
        return []


# Factory functions with type inference


@overload
def seq() -> Sequential: ...


@overload
def seq(spec: T, /) -> T: ...


@overload
def seq(spec1: Spec, spec2: Spec, /, *specs: Spec) -> Sequential: ...


def seq(*specs: Spec) -> Sequential | Spec:
    """Create a sequential composition of specifications.

    Composes specifications to execute in order, with outputs from each
    specification flowing to the next. When given a single specification,
    returns it unchanged for seamless composition.

    Parameters
    ----------
    *specs : Spec
        Variable number of specifications to compose sequentially

    Returns
    -------
    Sequential | Spec
        Empty Sequential if no arguments, the single spec if one argument,
        or Sequential containing all specs if multiple arguments

    Examples
    --------
    >>> # Empty sequence
    >>> empty = seq()
    >>> assert isinstance(empty, Sequential)
    >>> assert len(empty) == 0

    >>> # Single spec pass-through
    >>> single = seq(LayerNormSpec())
    >>> assert isinstance(single, LayerNormSpec)

    >>> # Multiple specs composed
    >>> pipeline = seq(
    ...     PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768),
    ...     CLSTokenSpec(),
    ...     ETSpec(steps=4)
    ... )
    >>> assert len(pipeline) == 3
    """
    if not specs:
        return Sequential()

    if len(specs) == 1:
        return specs[0]

    # Flatten nested Sequential specs
    parts: list[Spec] = []
    for spec in specs:
        if isinstance(spec, Sequential):
            parts.extend(spec.parts)
        else:
            parts.append(spec)

    return Sequential(parts=tuple(parts))


def parallel(
    *branches: Spec,
    merge: Literal["concat", "add", "multiply", "mean", "max"] = "concat",
    merge_dim: str | None = None,
    weights: Sequence[float] | None = None,
) -> Parallel:
    """Create parallel composition of specifications.

    Executes multiple specifications in parallel and combines their
    outputs according to the merge strategy.

    Parameters
    ----------
    *branches : Spec
        Specifications to execute in parallel
    merge : str
        Output merge strategy
    merge_dim : str, optional
        Dimension name for merge validation
    weights : Sequence[float], optional
        Weights for weighted merge strategies

    Returns
    -------
    Parallel
        Parallel composition specification

    Raises
    ------
    ValueError
        If no branches provided
    """
    if not branches:
        raise ValueError("Parallel requires at least one branch")

    # Flatten nested Parallel specs with same merge
    flat_branches: list[Spec] = []
    for branch in branches:
        if isinstance(branch, Parallel) and branch.merge == merge:
            flat_branches.extend(branch.branches)
        else:
            flat_branches.append(branch)

    return Parallel(
        branches=tuple(flat_branches),
        merge=merge,
        merge_dim=merge_dim,
        weights=tuple(weights) if weights else None,
    )


def residual(
    inner: Spec,
    *,
    merge: Literal["add", "concat", "gate"] = "add",
    gate_dim: str | None = None,
    scale: float = 1.0,
) -> Residual:
    """Create residual connection around a specification.

    Parameters
    ----------
    inner : Spec
        Specification to wrap with residual
    merge : str
        How to merge residual with output
    gate_dim : str, optional
        Dimension for gating (required for gate merge)
    scale : float
        Scale factor for residual branch

    Returns
    -------
    Residual
        Residual connection specification
    """
    return Residual(inner=inner, merge=merge, gate_dim=gate_dim, scale=scale)


def cond(
    condition: Callable[[Context], bool] | str,
    if_true: Spec,
    if_false: Spec | None = None,
) -> Conditional:
    """Create conditional specification.

    Parameters
    ----------
    condition : Callable[[Context], bool] | str
        Condition function or dimension name to check
    if_true : Spec
        Specification for true condition
    if_false : Spec, optional
        Specification for false condition

    Returns
    -------
    Conditional
        Conditional specification
    """
    if isinstance(condition, str):
        # Simple dimension existence check
        dim_name = condition

        def condition_fn(ctx: Context) -> bool:
            return ctx.get_dim(dim_name) is not None

        return Conditional(
            condition=condition_fn,
            if_true=if_true,
            if_false=if_false,
        )

    return Conditional(condition=condition, if_true=if_true, if_false=if_false)


def loop(
    body: Spec,
    times: int | str,
    *,
    unroll: bool = False,
    share_weights: bool = True,
) -> Loop:
    """Create loop specification.

    Parameters
    ----------
    body : Spec
        Specification to repeat
    times : int | str
        Number of iterations or dimension name
    unroll : bool
        Whether to unroll at compile time
    share_weights : bool
        Whether to share weights across iterations

    Returns
    -------
    Loop
        Loop specification
    """
    return Loop(
        body=body,
        times=times,
        unroll=unroll,
        share_weights=share_weights,
    )


def switch(
    key: str | Callable[[Context], Any],
    cases: dict[Any, Spec],
    default: Spec | None = None,
) -> Switch:
    """Create switch specification.

    Parameters
    ----------
    key : str | Callable[[Context], Any]
        Key name or function to compute key
    cases : dict[Any, Spec]
        Mapping from key values to specifications
    default : Spec, optional
        Default specification

    Returns
    -------
    Switch
        Switch specification
    """
    return Switch(key=key, cases=cases, default=default)


def graph() -> Graph:
    """Create empty graph specification.

    Returns
    -------
    Graph
        Empty graph specification
    """
    return Graph()


# Architecture patterns


def transformer_block(
    *,
    norm_first: bool = True,
    attention: Spec,
    mlp: Spec,
    drop_path: float = 0.0,
) -> Spec:
    """Create standard transformer block pattern.

    Parameters
    ----------
    norm_first : bool
        Whether to use pre-normalization
    attention : Spec
        Attention specification
    mlp : Spec
        MLP specification
    drop_path : float
        Drop path rate for stochastic depth

    Returns
    -------
    Spec
        Transformer block specification
    """
    if norm_first:
        # Pre-norm architecture
        attn_block = seq(Identity(), residual(seq(Identity(), attention)))
        mlp_block = seq(Identity(), residual(seq(Identity(), mlp)))
    else:
        # Post-norm architecture
        attn_block = seq(residual(attention), Identity())
        mlp_block = seq(residual(mlp), Identity())

    _ = drop_path
    return seq(attn_block, mlp_block)


def multi_scale(
    spec_fn: Callable[[int], Spec],
    scales: list[int],
    merge: Literal["concat", "add", "multiply", "mean", "max"] = "concat",
) -> Parallel:
    """Create multi-scale processing pattern.

    Parameters
    ----------
    spec_fn : Callable[[int], Spec]
        Function to create spec for each scale
    scales : list[int]
        Scale values to use
    merge : str
        How to merge multi-scale outputs

    Returns
    -------
    Parallel
        Multi-scale specification
    """
    branches = [spec_fn(scale) for scale in scales]
    return parallel(*branches, merge=merge)


def mixture_of_experts(
    experts: list[Spec],
    router: Spec,
    top_k: int = 2,
) -> Graph:
    """Create mixture of experts pattern.

    Parameters
    ----------
    experts : list[Spec]
        Expert specifications
    router : Spec
        Router specification
    top_k : int
        Number of experts to activate

    Returns
    -------
    Graph
        Mixture of experts specification
    """
    _ = top_k
    g = graph()

    # Add router
    g = g.add_node("router", router)
    g = g.add_edge("input", "router")

    # Add experts
    for i, expert in enumerate(experts):
        g = g.add_node(f"expert_{i}", expert)
        g = g.add_edge("router", f"expert_{i}", f"gate_{i}")

    # Combine outputs
    g = g.add_node("combine", Identity())
    for i in range(len(experts)):
        g = g.add_edge(f"expert_{i}", "combine")

    return Graph(
        nodes=g.nodes,
        edges=g.edges,
        inputs=["input"],
        outputs=["combine"],
    )
