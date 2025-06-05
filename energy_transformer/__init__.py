# ruff: noqa: TRY003
"""Energy Transformer: Energy-based transformers with associative memory.

This package implements the Energy Transformer architecture, which replaces
standard transformer components with energy-based alternatives.

Quick Start
-----------
>>> from energy_transformer import realise, seq
>>> from energy_transformer.spec.library import ETBlockSpec
>>>
>>> model = realise(seq(ETBlockSpec(), ETBlockSpec()))

For visualization and optional features:
>>> from energy_transformer.utils import visualize  # Loads matplotlib
>>> from energy_transformer.models import viet_base  # Loads full models

Troubleshooting
--------------
Common issues and solutions:

1. **High memory usage**: Energy Transformer requires gradient computation
   during inference. To reduce memory:
   - Reduce et_steps
   - Use smaller batch sizes
   - Enable gradient checkpointing

2. **Slow convergence**: If energy doesn't decrease:
   - Try different et_alpha values (typically 0.01 to 10.0)
   - Check if layer norm is numerically stable (increase eps)
   - Verify input data is properly normalized

3. **NaN values**: Usually caused by:
   - Too large et_alpha
   - Numerical instability in attention (add eps to denominators)
"""
# ruff: noqa: TRY003

from typing import TYPE_CHECKING

__version__ = "0.3.1"
__author__ = "Ayan Das <bvits@riseup.net>"
__license__ = "Apache-2.0"

# Lazy import system using PEP 562
_LAZY_IMPORTS = {
    # Core - always available
    "realise": "energy_transformer.spec",
    "register": "energy_transformer.spec",
    "seq": "energy_transformer.spec",
    "loop": "energy_transformer.spec",
    "parallel": "energy_transformer.spec",
    "cond": "energy_transformer.spec",
    "Spec": "energy_transformer.spec",
    "Context": "energy_transformer.spec",
    "ValidationError": "energy_transformer.spec",
    "RealisationError": "energy_transformer.spec",
    # Heavy imports - loaded on demand
    "EnergyTransformer": "energy_transformer.models",
    "visualize": "energy_transformer.spec",
    "configure_realisation": "energy_transformer.spec",
    "testing": "energy_transformer.testing",
}

# For static type checking, import everything
if TYPE_CHECKING:  # pragma: no cover
    from .models import EnergyTransformer  # noqa: F401
    from .spec import (
        Context,
        RealisationError,
        Spec,
        ValidationError,
        cond,  # noqa: F401
        configure_realisation,  # noqa: F401
        loop,
        parallel,
        realise,
        register,  # noqa: F401
        seq,
        visualize,  # noqa: F401
    )


def __getattr__(name: str) -> object:
    """Lazy load modules and attributes.

    This implements PEP 562 for lazy loading. Attributes are only
    imported when actually accessed, dramatically improving import time.
    """
    if name in _LAZY_IMPORTS:
        module_name = _LAZY_IMPORTS[name]
        import importlib

        try:
            module = importlib.import_module(module_name)
        except ImportError as e:  # pragma: no cover - error path
            raise ImportError(
                f"Cannot import {name} from {module_name}. "
                f"This might be due to missing optional dependencies. "
                f"Try: pip install energy-transformer[all]\n"
                f"Original error: {e}",
            ) from e
        try:
            attr = getattr(module, name)
        except AttributeError as e:  # pragma: no cover - unexpected
            raise AttributeError(
                f"Module {module_name} has no attribute {name}",
            ) from e
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List available attributes for tab completion."""
    return [
        *list(_LAZY_IMPORTS.keys()),
        "__version__",
        "__author__",
        "__license__",
        "__all__",
    ]


# Only include commonly used exports in __all__
__all__ = [
    "Context",
    "RealisationError",
    "Spec",
    "ValidationError",
    "loop",
    "parallel",
    "realise",
    "seq",
]

# NO SIDE EFFECTS ON IMPORT!  Configuration should be explicit.
