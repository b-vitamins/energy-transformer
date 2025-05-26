"""Energy Transformer specification system.

This module provides a declarative specification system for defining
machine learning model architectures. Specifications are immutable
descriptions that can be validated, composed, and transformed into
executable PyTorch modules.

The system consists of three main components:

1. **Primitives**: Base specification types with validation and dimension
   tracking
2. **Combinators**: Composition operators for building complex architectures
3. **Realisation**: System for converting specifications into PyTorch modules

Example
-------
>>> # Define a simple vision transformer
>>> model_spec = seq(
...     PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768),
...     CLSTokenSpec(),
...     PosEmbedSpec(include_cls=True),
...     loop(
...         ETSpec(
...             attention=MHEASpec(num_heads=12, head_dim=64),
...             hopfield=HNSpec()
...         ),
...         times=12
...     ),
...     LayerNormSpec()
... )
>>>
>>> # Realise into PyTorch module
>>> model = realise(model_spec)
>>>
>>> # Use the model
>>> images = torch.randn(4, 3, 224, 224)
>>> output = model(images)
"""

from collections.abc import Callable
from typing import Any, TypeAlias

# Combinators
from .combinators import (
    Conditional,
    Graph,
    Identity,
    Lambda,
    Loop,
    Parallel,
    Residual,
    # Core combinators
    Sequential,
    Switch,
    cond,
    graph,
    loop,
    mixture_of_experts,
    multi_scale,
    parallel,
    residual,
    # Factory functions
    seq,
    switch,
    # Architectural patterns
    transformer_block,
)

# Core primitives
from .primitives import (
    # Constants and types
    REQUIRED,
    AsyncSpec,
    Context,
    Dimension,
    DimensionDef,
    DimensionLike,
    # Base classes
    Spec,
    # Errors
    ValidationError,
    modifies,
    # Decorators
    param,
    provides,
    requires,
    spec,
    validate_field,
)

# Realisation system
from .realise import (
    # Advanced features
    ModuleCache,
    # Errors
    RealisationError,
    Realiser,
    RealiserPlugin,
    # Configuration
    configure_realisation,
    from_yaml,
    optimize_spec,
    # Main API
    realise,
    register,
    register_typed,
    to_yaml,
    # Utilities
    visualize,
)

# Import concrete specifications if they exist
# These would be defined in a separate module like spec.library or spec.catalog
try:
    from .library import (
        CLSTokenSpec,
        ETSpec,
        HNSpec,
        # Core layer specs
        LayerNormSpec,
        MHEASpec,
        # Vision specs
        PatchEmbedSpec,
        PosEmbedSpec,
    )
except ImportError:
    # Library module not available yet
    pass

__all__ = [
    # --- Core Primitives ---
    "Spec",
    "AsyncSpec",
    "Context",
    "Dimension",
    "DimensionDef",
    "ValidationError",
    "param",
    "requires",
    "provides",
    "modifies",
    "spec",
    "validate_field",
    "REQUIRED",
    "DimensionLike",
    # --- Combinators ---
    # Classes
    "Sequential",
    "Parallel",
    "Conditional",
    "Residual",
    "Graph",
    "Loop",
    "Switch",
    "Identity",
    "Lambda",
    # Factory functions
    "seq",
    "parallel",
    "residual",
    "cond",
    "loop",
    "switch",
    "graph",
    # Patterns
    "transformer_block",
    "multi_scale",
    "mixture_of_experts",
    # --- Realisation ---
    "realise",
    "register",
    "register_typed",
    "configure_realisation",
    "RealisationError",
    "ModuleCache",
    "RealiserPlugin",
    "Realiser",
    "visualize",
    "optimize_spec",
    "to_yaml",
    "from_yaml",
]

# Add library specs to __all__ if available
try:
    from .library import __all__ as library_all

    __all__.extend(library_all)
except (ImportError, NameError):
    pass

# Version information
__version__ = "2.0.0"

# Convenience aliases for common patterns
Seq = seq
Par = parallel
Res = residual
Rep = loop

# Type aliases for better IDE support
SpecLike: TypeAlias = Spec | Sequential | Parallel | Conditional | Residual

# Configure default settings
configure_realisation(
    cache=ModuleCache(max_size=128, enabled=True),
    strict=True,
    warnings=True,
    auto_import=True,
    optimizations=True,
    max_recursion=100,
)


# Quick start guide in docstring
def quickstart() -> None:
    """Print quick start guide for the specification system.

    Example
    -------
    >>> from energy_transformer.spec import quickstart
    >>> quickstart()
    """
    print("""
Energy Transformer Specification System - Quick Start
====================================================

1. Define specifications:
   ```python
   from energy_transformer.spec import *

   model = seq(
       PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768),
       CLSTokenSpec(),
       loop(ETSpec(), times=12),
       LayerNormSpec()
   )
   ```

2. Realise into PyTorch module:
   ```python
   module = realise(model)
   ```

3. Use operators for composition:
   - Sequential: spec1 >> spec2 >> spec3
   - Parallel: spec1 | spec2 | spec3
   - Conditional: cond("use_cls", CLSTokenSpec())
   - Loop: loop(ETSpec(), times=12)

4. Register custom realisers:
   ```python
   @register(MySpec)
   def realise_my_spec(spec, context):
       return MyModule(spec.param1, spec.param2)
   ```

5. Configure the system:
   ```python
   configure_realisation(
       cache=ModuleCache(max_size=256),
       strict=False,
       auto_import=True
   )
   ```

For more information, see the module docstring or documentation.
""")


# Export pattern for specific use cases
def export_patterns() -> dict[str, Callable[..., Spec]]:
    """Return common architectural patterns ready to use.

    Returns
    -------
    dict
        Dictionary of pattern name to spec factory function
    """
    return {
        "vit_tiny": lambda **kwargs: seq(
            PatchEmbedSpec(
                img_size=224, patch_size=16, embed_dim=192, **kwargs
            ),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            loop(
                ETSpec(
                    attention=MHEASpec(num_heads=3, head_dim=64),
                    hopfield=HNSpec(),
                ),
                times=12,
            ),
            LayerNormSpec(),
        ),
        "vit_base": lambda **kwargs: seq(
            PatchEmbedSpec(
                img_size=224, patch_size=16, embed_dim=768, **kwargs
            ),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            loop(
                ETSpec(
                    attention=MHEASpec(num_heads=12, head_dim=64),
                    hopfield=HNSpec(),
                ),
                times=12,
            ),
            LayerNormSpec(),
        ),
    }


# Backward compatibility helpers
def migrate_from_v1(old_spec: Any) -> Spec:
    """Help migrate specifications from v1 to v2.

    Parameters
    ----------
    old_spec : Any
        Old specification object

    Returns
    -------
    Spec
        New specification object

    Raises
    ------
    NotImplementedError
        If migration is not yet implemented
    """
    raise NotImplementedError("Migration guide coming soon")


# Development utilities
def validate_spec_tree(spec: Spec, verbose: bool = False) -> list[str]:
    """Validate an entire specification tree.

    Parameters
    ----------
    spec : Spec
        Root specification to validate
    verbose : bool
        Whether to print detailed information

    Returns
    -------
    list[str]
        List of validation issues found
    """
    context = Context()
    issues = spec.validate(context)

    if verbose:
        print(f"Validating {spec.__class__.__name__}...")
        if issues:
            print("Issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("  ✓ No issues found")

        # Validate children recursively
        for child in spec.children():
            child_issues = validate_spec_tree(child, verbose=True)
            issues.extend(child_issues)

    return issues


def benchmark_realisation(
    spec: Spec, iterations: int = 100
) -> dict[str, float]:
    """Benchmark specification realisation performance.

    Parameters
    ----------
    spec : Spec
        Specification to benchmark
    iterations : int
        Number of iterations for timing

    Returns
    -------
    dict[str, float]
        Timing statistics
    """
    import time

    times = []

    # Warm up
    realise(spec)

    # Benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        realise(spec)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
        "total": sum(times),
        "iterations": iterations,
    }


# Module initialization message (can be disabled)
_INIT_MESSAGE_ENABLED = False

if _INIT_MESSAGE_ENABLED:
    print(f"Energy Transformer Spec System v{__version__} loaded")
    print("Use quickstart() for a quick introduction")
