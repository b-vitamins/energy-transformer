"""Energy Transformer models."""

# Core models
from .models import EnergyTransformer

# Specification API - Core functionality
from .spec import (
    Context,
    RealisationError,
    # Base types for type hints
    Spec,
    # Common errors
    ValidationError,
    cond,
    configure_realisation,
    loop,
    parallel,
    # Main API functions
    realise,
    register,
    # Composition functions
    seq,
    # Advanced features
    visualize,
)

__all__ = [
    # Core model
    "EnergyTransformer",
    # Spec system - Main API
    "realise",
    "register",
    # Spec system - Composition
    "seq",
    "loop",
    "parallel",
    "cond",
    # Spec system - Base types
    "Spec",
    "Context",
    # Spec system - Errors
    "ValidationError",
    "RealisationError",
    # Spec system - Utilities
    "visualize",
    "configure_realisation",
]

__version__ = "0.1.0"
