"""Energy Transformer: Energy-based transformers with associative memory."""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.3.1"
__author__ = "Ayan Das <bvits@riseup.net>"
__license__ = "Apache-2.0"

_LAZY_IMPORTS = {"EnergyTransformer": "energy_transformer.models"}

if TYPE_CHECKING:  # pragma: no cover
    from .models import EnergyTransformer


def __getattr__(name: str) -> object:
    """Lazy load modules and attributes."""
    if name in _LAZY_IMPORTS:
        module_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_name)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List available attributes for tab completion."""
    return [
        *_LAZY_IMPORTS.keys(),
        "__version__",
        "__author__",
        "__license__",
    ]


__all__ = ["EnergyTransformer"]
