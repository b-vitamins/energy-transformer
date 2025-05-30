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
...         ETBlockSpec(
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
from typing import TypeAlias

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

# Lazy loading for library specs
_LIBRARY_SPECS = [
    "LayerNormSpec",
    "PatchEmbedSpec",
    "CLSTokenSpec",
    "PosEmbedSpec",
    "MHASpec",
    "MHEASpec",
    "MLPSpec",
    "HNSpec",
    "SHNSpec",
    "ETBlockSpec",
    "ClassificationHeadSpec",
    "FeatureHeadSpec",
    "DropoutSpec",
    "IdentitySpec",
    "VisionEmbeddingSpec",
    "TransformerBlockSpec",
]

def __getattr__(name: str):
    """Lazy load library specifications."""
    if name in _LIBRARY_SPECS:
        from . import library
        try:
            return getattr(library, name)
        except AttributeError:  # pragma: no cover - unexpected
            raise AttributeError(
                f"energy_transformer.spec.library has no spec {name!r}"
            ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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



# Version information
__version__ = "0.2.0-alpha1"

# Convenience aliases for common patterns
Seq = seq
Par = parallel
Res = residual
Rep = loop

# Type aliases for better IDE support
SpecLike: TypeAlias = Spec | Sequential | Parallel | Conditional | Residual

# NO GLOBAL CONFIGURATION ON IMPORT!
# Users must explicitly configure if they want non-defaults
def initialize_defaults():
    """Initialize default configuration.

    This must be called explicitly by users who want default settings.

    Example
    -------
    >>> import energy_transformer.spec as et_spec
    >>> et_spec.initialize_defaults()  # Explicit initialization
    """
    from .realise import configure_realisation, ModuleCache

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

    This only prints information, it does not modify any global state.

    Example
    -------
    >>> from energy_transformer.spec import quickstart
    >>> quickstart()
    """
    guide = """Energy Transformer Specification System - Quick Start
====================================================
1. Initialize defaults (optional):
   ```python
   import energy_transformer.spec as et_spec
   et_spec.initialize_defaults()
   ```
2. Define specifications:
   ```python
   from energy_transformer.spec import seq
   from energy_transformer.spec.library import *
   model = seq(
       PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768),
       CLSTokenSpec(),
       loop(ETBlockSpec(), times=12),
       LayerNormSpec()
   )
   ```
3. Realise into PyTorch module:
   ```python
   from energy_transformer import realise
   module = realise(model)
   ```
For more information, see the documentation."""
    print(guide)
    # DO NOT call initialize_defaults() here!


# Export pattern for specific use cases
def export_patterns() -> dict[str, Callable[..., Spec]]:
    """Return common architectural patterns ready to use."""
    from . import library

    PatchEmbedSpec = library.PatchEmbedSpec
    CLSTokenSpec = library.CLSTokenSpec
    PosEmbedSpec = library.PosEmbedSpec
    ETBlockSpec = library.ETBlockSpec
    LayerNormSpec = library.LayerNormSpec
    MHEASpec = library.MHEASpec
    HNSpec = library.HNSpec

    return {
        "vit_tiny": lambda **kwargs: seq(
            PatchEmbedSpec(
                img_size=224, patch_size=16, embed_dim=192, **kwargs
            ),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            loop(
                ETBlockSpec(
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
                ETBlockSpec(
                    attention=MHEASpec(num_heads=12, head_dim=64),
                    hopfield=HNSpec(),
                ),
                times=12,
            ),
            LayerNormSpec(),
        ),
    }


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


# Module initialization message removed. Use quickstart() for help.
