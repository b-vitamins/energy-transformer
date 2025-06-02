"""Type stubs for energy_transformer.spec module."""

from collections.abc import Callable
from typing import Any, TypeVar, overload

from torch import nn

from .combinators import (
    Conditional,
    Graph,
    Loop,
    Parallel,
    Residual,
    Sequential,
    Switch,
)
from .primitives import Context, Spec
from .realise import ModuleCache

T = TypeVar('T', bound=Spec)
S = TypeVar('S', bound=Spec)

# Core realisation function
def realise(
    spec: Spec,
    context: Context | None = None,
    **context_updates: Any
) -> nn.Module: ...

# Combinators
@overload
def seq() -> Sequential: ...
@overload
def seq(spec: T, /) -> T: ...
@overload
def seq(spec1: Spec, spec2: Spec, /, *specs: Spec) -> Sequential: ...


def parallel(
    *branches: Spec,
    merge: str = "concat",
    merge_dim: str | None = None,
    weights: tuple[float, ...] | None = None
) -> Parallel: ...


def loop(
    body: Spec,
    times: int | str,
    *,
    unroll: bool = False,
    share_weights: bool = True
) -> Loop: ...


def cond(
    condition: Callable[[Context], bool] | str,
    if_true: Spec,
    if_false: Spec | None = None
) -> Conditional: ...


def residual(
    inner: Spec,
    *,
    merge: str = "add",
    gate_dim: str | None = None,
    scale: float = 1.0
) -> Residual: ...


def switch(
    key: str | Callable[[Context], Any],
    cases: dict[Any, Spec],
    default: Spec | None = None
) -> Switch: ...


def graph() -> Graph: ...

# Registration
def register(spec_cls: type[Spec]) -> Callable[[Any], Any]: ...
def register_typed(
    fn: Callable[[Any, Context], nn.Module]
) -> Callable[[Any, Context], nn.Module]: ...

# Configuration
def configure_realisation(
    *,
    cache: ModuleCache | None = None,
    strict: bool | None = None,
    warnings: bool | None = None,
    auto_import: bool | None = None,
    optimizations: bool | None = None,
    max_recursion: int | None = None,
    enable_metrics: bool | None = None,
    # Constants overrides
    MAX_RECURSION: int | None = None,  # noqa: N803
    MAX_STACK_PREVIEW: int | None = None,  # noqa: N803
    UNROLL_LIMIT: int | None = None,  # noqa: N803
    DEFAULT_CACHE_SIZE: int | None = None,  # noqa: N803
    **kwargs: Any
) -> None: ...

def initialize_defaults() -> None: ...

# Metrics
def get_realisation_metrics() -> dict[str, Any]: ...
def reset_metrics() -> None: ...

# Utilities
def visualize(spec: Spec, out_format: str = "svg") -> str: ...
def to_yaml(spec: Spec) -> str: ...
def from_yaml(yaml_str: str) -> Spec: ...
def optimize_spec(spec: Spec) -> Spec: ...
def validate_spec_tree(spec: Spec, verbose: bool = False) -> list[str]: ...
def benchmark_realisation(
    spec: Spec,
    iterations: int = 100
) -> dict[str, float]: ...

# Pattern functions
def transformer_block(
    *,
    norm_first: bool = True,
    attention: Spec,
    mlp: Spec,
    drop_path: float = 0.0
) -> Spec: ...


def multi_scale(
    spec_fn: Callable[[int], Spec],
    scales: list[int],
    merge: str = "concat"
) -> Parallel: ...


def mixture_of_experts(
    experts: list[Spec],
    router: Spec,
    top_k: int = 2
) -> Graph: ...

# Export utilities
def export_patterns() -> dict[str, Callable[..., Spec]]: ...
def quickstart() -> None: ...

# Constants
REQUIRED: Any
__version__: str
__author__: str
__license__: str

# Type aliases
Seq = seq
Par = parallel
Res = residual
Rep = loop
SpecLike = Spec | Sequential | Parallel | Conditional | Residual
