"""Realisation of model specs into actual PyTorch modules.

This module converts specification objects into concrete PyTorch modules.
It handles dimension propagation, validation, and provides a registry
system for mapping spec types to their corresponding module constructors.

Design Principles:
- Type-safe realisation with comprehensive error handling
- Dimension and context propagation through spec pipelines
- Registry-based extensibility for custom specs
- Immutable context objects for safe parallel processing
- Clear separation between specification and implementation

Example
-------
>>> from energy_transformer.spec import seq, ETSpec, LayerNormSpec
>>> from energy_transformer.spec import PatchEmbedSpec, CLSTokenSpec
>>>
>>> # Define a model specification
>>> model_spec = seq(
...     PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768),
...     CLSTokenSpec(),
...     ETSpec(steps=4, alpha=0.125),
...     LayerNormSpec()
... )
>>>
>>> # Realise into PyTorch module
>>> model = realise(model_spec)
>>>
>>> # Use the model
>>> import torch
>>> x = torch.randn(2, 3, 224, 224)
>>> output = model(x)  # Shape: [2, 197, 768]
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn as nn

from .combinators import ParallelSpec, SequentialSpec
from .primitives import (
    CLSTokenSpec,
    EmbeddingDim,
    ETSpec,
    HNSpec,
    LayerNormSpec,
    MHEASpec,
    PatchEmbedSpec,
    PosEmbedSpec,
    Spec,
    TokenCount,
    ValidationError,
)

__all__ = [
    "realise",
    "Realise",
    "register_realiser",
    "SpecInfo",
    "RealisationError",
]

# Type alias for realiser functions
RealiserFunc = Callable[[Any, "SpecInfo"], nn.Module]

# Registry for realiser functions
_REALISERS: dict[type, RealiserFunc] = {}


class RealisationError(Exception):
    """Raised when a specification cannot be realised into a module.

    Parameters
    ----------
    message : str
        The error message.
    spec_type : str, optional
        The type of spec that failed realisation.
    suggestion : str, optional
        A helpful suggestion for fixing the error.
    """

    def __init__(
        self,
        message: str,
        spec_type: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Initialize RealisationError with enhanced messaging."""
        self.spec_type = spec_type
        self.suggestion = suggestion

        full_message = message
        if spec_type:
            full_message = f"{spec_type}: {message}"
        if suggestion:
            full_message += f"\nSuggestion: {suggestion}"

        super().__init__(full_message)


class SpecInfo:
    """Context information for specification realisation.

    Tracks embedding dimensions, token counts, and other contextual
    information as it flows through the specification pipeline.
    Provides safe copying for parallel branches.

    Parameters
    ----------
    embedding_dim : EmbeddingDim, optional
        The current embedding dimension.
    token_count : TokenCount, optional
        The current token count.

    Attributes
    ----------
    embedding_dim : EmbeddingDim | None
        The current embedding dimension.
    token_count : TokenCount | None
        The current token count.

    Examples
    --------
    >>> info = SpecInfo()
    >>> info.embedding_dim = 768
    >>> info.token_count = 196
    >>>
    >>> # Safe copying for parallel branches
    >>> branch_info = info.copy()
    >>> branch_info.embedding_dim = 512  # Doesn't affect original
    """

    def __init__(
        self,
        embedding_dim: EmbeddingDim | None = None,
        token_count: TokenCount | None = None,
    ) -> None:
        """Initialize SpecInfo with optional context.

        Parameters
        ----------
        embedding_dim : EmbeddingDim, optional
            Initial embedding dimension.
        token_count : TokenCount, optional
            Initial token count.
        """
        self.embedding_dim = embedding_dim
        self.token_count = token_count

    def update_from_spec(self, spec: Spec) -> None:
        """Update context information from a specification.

        Parameters
        ----------
        spec : Spec
            The specification to update from.

        Notes
        -----
        This method updates the context by:
        1. Getting new embedding dimension if spec defines one
        2. Getting new token count if spec defines one
        3. Adding any tokens the spec contributes
        """
        # Update embedding dimension
        dim = spec.get_embedding_dim()
        if dim is not None:
            self.embedding_dim = dim

        # Update base token count
        count = spec.get_token_count()
        if count is not None:
            self.token_count = count

        # Add any tokens this spec contributes
        if self.token_count is not None:
            self.token_count += spec.adds_tokens()

    def copy(self) -> SpecInfo:
        """Create a copy of this SpecInfo.

        Returns
        -------
        SpecInfo
            A new SpecInfo with the same state as this one.

        Examples
        --------
        >>> original = SpecInfo(embedding_dim=768, token_count=196)
        >>> copy = original.copy()
        >>> copy.embedding_dim = 512
        >>> original.embedding_dim  # Still 768
        """
        return SpecInfo(self.embedding_dim, self.token_count)

    def validate_spec(self, spec: Spec) -> None:
        """Validate a specification against this context.

        Parameters
        ----------
        spec : Spec
            The specification to validate.

        Raises
        ------
        ValidationError
            If the spec cannot be validated against this context.

        Examples
        --------
        >>> info = SpecInfo(embedding_dim=768)
        >>> layer_norm = LayerNormSpec()
        >>> info.validate_spec(layer_norm)  # OK - LayerNorm needs embed_dim
        >>>
        >>> empty_info = SpecInfo()
        >>> empty_info.validate_spec(layer_norm)  # Raises ValidationError
        """
        try:
            spec.validate(self.embedding_dim, self.token_count)
        except ValidationError as e:
            raise RealisationError(
                f"Spec validation failed: {e}",
                spec_type=spec.__class__.__name__,
                suggestion="Ensure upstream specs provide required context",
            ) from e

    def __str__(self) -> str:
        """Return string representation of SpecInfo.

        Returns
        -------
        str
            Human-readable representation of the SpecInfo state.
        """
        return (
            f"SpecInfo(embed_dim={self.embedding_dim}, "
            f"token_count={self.token_count})"
        )

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return (
            f"SpecInfo(embedding_dim={self.embedding_dim!r}, "
            f"token_count={self.token_count!r})"
        )


def register_realiser(
    spec_type: type,
) -> Callable[[RealiserFunc], RealiserFunc]:
    """Register a realiser function for a specification type.

    Parameters
    ----------
    spec_type : type
        The specification type to register a realiser for.

    Returns
    -------
    Callable
        Decorator function that registers the realiser.

    Examples
    --------
    >>> @register_realiser(MyCustomSpec)
    ... def realise_my_spec(spec: MyCustomSpec, info: SpecInfo) -> nn.Module:
    ...     return MyCustomModule(spec.param1, spec.param2)
    """

    def decorator(fn: RealiserFunc) -> RealiserFunc:
        _REALISERS[spec_type] = fn
        return fn

    return decorator


def realise(spec: Any, info: SpecInfo | None = None) -> nn.Module | None:
    """Convert a specification to a PyTorch module.

    This is the main entry point for converting specifications into
    concrete PyTorch modules. It handles validation, dimension propagation,
    and dispatches to appropriate realiser functions.

    Parameters
    ----------
    spec : Any
        A specification object to realise.
    info : SpecInfo, optional
        Context information from upstream specs. If None, creates empty
        context.

    Returns
    -------
    nn.Module | None
        A PyTorch module constructed according to the spec, or None if spec
        is None.

    Raises
    ------
    RealisationError
        If the spec cannot be realised due to missing realisers, validation
        failures, or other errors.
    TypeError
        If the spec type is not supported.

    Examples
    --------
    >>> # Simple spec realisation
    >>> layer_norm = LayerNormSpec()
    >>> context = SpecInfo(embedding_dim=768)
    >>> module = realise(layer_norm, context)
    >>> isinstance(module, nn.LayerNorm)  # True

    >>> # Complex model realisation
    >>> model_spec = seq(
    ...     PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768),
    ...     CLSTokenSpec(),
    ...     ETSpec()
    ... )
    >>> model = realise(model_spec)
    >>> isinstance(model, nn.Sequential)  # True
    """
    # Initialize context if not provided
    if info is None:
        info = SpecInfo()

    # Handle None case
    if spec is None:
        return None

    # If it's already a module, return as-is
    if isinstance(spec, nn.Module):
        return spec

    # Validate spec against current context
    if isinstance(spec, Spec):
        info.validate_spec(spec)

    # Handle sequential composition
    if isinstance(spec, SequentialSpec):
        return _realise_sequential(spec, info)

    # Handle parallel composition
    if isinstance(spec, ParallelSpec):
        return _realise_parallel(spec, info)

    # Handle leaf specs using registered realisers
    spec_type = type(spec)
    if spec_type in _REALISERS:
        try:
            module = _REALISERS[spec_type](spec, info)

            # Update context with this spec's contributions
            if isinstance(spec, Spec):
                info.update_from_spec(spec)

            return module
        except Exception as e:
            raise RealisationError(
                f"Failed to realise spec: {e}",
                spec_type=spec_type.__name__,
                suggestion="Check spec parameters and module dependencies",
            ) from e

    available_types = ", ".join(t.__name__ for t in _REALISERS.keys())
    raise TypeError(
        f"No realiser registered for {spec_type.__name__}. "
        f"Available types: {available_types}"
    )


def _realise_sequential(spec: SequentialSpec, info: SpecInfo) -> nn.Module:
    """Realise a sequential specification.

    Parameters
    ----------
    spec : SequentialSpec
        The sequential specification to realise.
    info : SpecInfo
        Context information from upstream specs.

    Returns
    -------
    nn.Module
        The realised sequential module.

    Raises
    ------
    RealisationError
        If any part of the sequence fails to realise.
    """
    modules: list[nn.Module] = []

    # Process each part in order, propagating context
    for i, part in enumerate(spec.parts):
        try:
            module = realise(part, info)
            if module is not None:
                modules.append(module)
        except (RealisationError, TypeError) as e:
            raise RealisationError(
                f"Failed to realise part {i} ({part.__class__.__name__}): {e}",
                spec_type="SequentialSpec",
            ) from e

    return nn.Sequential(*modules)


def _realise_parallel(spec: ParallelSpec, info: SpecInfo) -> nn.Module:
    """Realise a parallel specification.

    Parameters
    ----------
    spec : ParallelSpec
        The parallel specification to realise.
    info : SpecInfo
        Context information from upstream specs.

    Returns
    -------
    nn.Module
        The realised parallel module.

    Raises
    ------
    RealisationError
        If any branch fails to realise or if no valid branches exist.
    """
    branches: list[nn.Module] = []

    # Process each branch independently with copied context
    for i, branch in enumerate(spec.branches):
        try:
            branch_info = info.copy()
            module = realise(branch, branch_info)
            if module is not None:
                branches.append(module)
        except (RealisationError, TypeError) as e:
            raise RealisationError(
                f"Failed to realise branch {i} "
                f"({branch.__class__.__name__}): {e}",
                spec_type="ParallelSpec",
            ) from e

    if not branches:
        raise RealisationError(
            "No valid branches to combine",
            spec_type="ParallelSpec",
            suggestion="Ensure at least one branch can be realised",
        )

    # Create module that combines branches
    class ParallelModule(nn.Module):  # type: ignore[misc]
        """Module that combines parallel branches according to join mode."""

        def __init__(self, branches: list[nn.Module], join_mode: str) -> None:
            """Initialize parallel module.

            Parameters
            ----------
            branches : list[nn.Module]
                List of branch modules to combine.
            join_mode : str
                How to combine the branch outputs.
            """
            super().__init__()
            self.branches = nn.ModuleList(branches)
            self.join_mode = join_mode

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through all branches.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor.

            Returns
            -------
            torch.Tensor
                Combined output from all branches.
            """
            outputs = [branch(x) for branch in self.branches]

            if self.join_mode == "concat":
                return torch.cat(outputs, dim=-1)
            elif self.join_mode == "add":
                return sum(outputs)
            elif self.join_mode == "multiply":
                result = outputs[0]
                for output in outputs[1:]:
                    result = result * output
                return result
            else:
                raise ValueError(f"Unsupported join mode: {self.join_mode}")

    # Update context based on parallel module's output dimension
    output_dim = spec.get_embedding_dim()
    if output_dim is not None:
        info.embedding_dim = output_dim

    return ParallelModule(branches, spec.join_mode)


# Register realisers for primitive specs


@register_realiser(LayerNormSpec)
def _realise_layer_norm(spec: LayerNormSpec, info: SpecInfo) -> nn.Module:
    """Realise a layer normalization specification.

    Parameters
    ----------
    spec : LayerNormSpec
        The layer normalization specification.
    info : SpecInfo
        Context information containing embedding dimension.

    Returns
    -------
    nn.Module
        The layer normalization module.

    Raises
    ------
    RealisationError
        If embedding dimension is not available.
    """
    if info.embedding_dim is None:
        raise RealisationError(
            "LayerNormSpec requires embedding dimension from context",
            spec_type="LayerNormSpec",
            suggestion="Place LayerNormSpec after a component that "
            "defines embed_dim",
        )

    try:
        from energy_transformer.layers.layer_norm import LayerNorm

        return LayerNorm(in_dim=info.embedding_dim, eps=spec.eps)
    except ImportError:
        # Fallback to standard PyTorch LayerNorm
        return nn.LayerNorm(info.embedding_dim, eps=spec.eps)


@register_realiser(MHEASpec)
def _realise_mhea(spec: MHEASpec, info: SpecInfo) -> nn.Module:
    """Realise a multi-head energy attention specification.

    Parameters
    ----------
    spec : MHEASpec
        The attention specification.
    info : SpecInfo
        Context information containing embedding dimension.

    Returns
    -------
    nn.Module
        The attention module.

    Raises
    ------
    RealisationError
        If embedding dimension is not available or module cannot be imported.
    """
    if info.embedding_dim is None:
        raise RealisationError(
            "MHEASpec requires embedding dimension from context",
            spec_type="MHEASpec",
            suggestion="Place MHEASpec after a component that "
            "defines embed_dim",
        )

    try:
        from energy_transformer.layers import MultiHeadEnergyAttention

        return MultiHeadEnergyAttention(
            in_dim=info.embedding_dim,
            num_heads=spec.num_heads,
            head_dim=spec.head_dim,
            beta=spec.get_effective_beta(),
            bias=spec.bias,
        )
    except ImportError as e:
        raise RealisationError(
            f"Cannot import MultiHeadEnergyAttention: {e}",
            spec_type="MHEASpec",
            suggestion="Ensure energy_transformer.layers.attention is "
            "available",
        ) from e


@register_realiser(HNSpec)
def _realise_hn(spec: HNSpec, info: SpecInfo) -> nn.Module:
    """Realise a Hopfield network specification.

    Parameters
    ----------
    spec : HNSpec
        The Hopfield network specification.
    info : SpecInfo
        Context information containing embedding dimension.

    Returns
    -------
    nn.Module
        The Hopfield network module.

    Raises
    ------
    RealisationError
        If embedding dimension is not available or module cannot be imported.
    """
    if info.embedding_dim is None:
        raise RealisationError(
            "HNSpec requires embedding dimension from context",
            spec_type="HNSpec",
            suggestion="Place HNSpec after a component that defines embed_dim",
        )

    try:
        from energy_transformer.layers.hopfield import (
            ActivationFunction,
            HopfieldNetwork,
        )

        # Convert string to enum
        activation_map = {
            "relu": ActivationFunction.RELU,
            "softmax": ActivationFunction.SOFTMAX,
            "power": ActivationFunction.POWER,
            "tanh": ActivationFunction.TANH,
        }
        activation = activation_map[spec.activation]

        return HopfieldNetwork(
            in_dim=info.embedding_dim,
            hidden_dim=spec.hidden_dim,
            multiplier=spec.multiplier,
            bias=spec.bias,
            activation=activation,
        )
    except ImportError as e:
        raise RealisationError(
            f"Cannot import HopfieldNetwork: {e}",
            spec_type="HNSpec",
            suggestion="Ensure energy_transformer.layers.hopfield exists",
        ) from e
    except KeyError as e:
        raise RealisationError(
            f"Unknown activation function: {spec.activation}",
            spec_type="HNSpec",
            suggestion="Use 'relu', 'softmax', 'power', or 'tanh'",
        ) from e


@register_realiser(ETSpec)
def _realise_et(spec: ETSpec, info: SpecInfo) -> nn.Module:
    """Realise an Energy Transformer specification.

    Parameters
    ----------
    spec : ETSpec
        The Energy Transformer specification.
    info : SpecInfo
        Context information containing embedding dimension.

    Returns
    -------
    nn.Module
        The Energy Transformer module.

    Raises
    ------
    RealisationError
        If embedding dimension is not available or components cannot be
        created.
    """
    if info.embedding_dim is None:
        raise RealisationError(
            "ETSpec requires embedding dimension from context",
            spec_type="ETSpec",
            suggestion="Place ETSpec after a component that defines embed_dim",
        )

    try:
        from energy_transformer.models.base import EnergyTransformer

        # Create sub-components using their realisers
        layer_norm = realise(spec.layer_norm, info)
        attention = realise(spec.attention, info)
        hopfield = realise(spec.hopfield, info)

        # Ensure we have valid components
        if layer_norm is None or attention is None or hopfield is None:
            raise RealisationError(
                "Failed to create required sub-components",
                spec_type="ETSpec",
                suggestion="Check that sub-component specs are valid",
            )

        return EnergyTransformer(
            layer_norm=cast(Any, layer_norm),  # Type casting for flexibility
            attention=cast(Any, attention),
            hopfield=cast(Any, hopfield),
            steps=spec.steps,
            α=spec.alpha,
        )
    except ImportError as e:
        raise RealisationError(
            f"Cannot import EnergyTransformer: {e}",
            spec_type="ETSpec",
            suggestion="Ensure energy_transformer.models.base is available",
        ) from e


@register_realiser(CLSTokenSpec)
def _realise_cls_token(spec: CLSTokenSpec, info: SpecInfo) -> nn.Module:
    """Realise a CLS token specification.

    Parameters
    ----------
    spec : CLSTokenSpec
        The CLS token specification.
    info : SpecInfo
        Context information containing embedding dimension.

    Returns
    -------
    nn.Module
        The CLS token module.

    Raises
    ------
    RealisationError
        If embedding dimension is not available or module cannot be imported.
    """
    if info.embedding_dim is None:
        raise RealisationError(
            "CLSTokenSpec requires embedding dimension from context",
            spec_type="CLSTokenSpec",
            suggestion="Place CLSTokenSpec after a component that "
            "defines embed_dim",
        )

    try:
        from energy_transformer.layers.tokens import CLSToken

        return CLSToken(embed_dim=info.embedding_dim)
    except ImportError as e:
        raise RealisationError(
            f"Cannot import CLSToken: {e}",
            spec_type="CLSTokenSpec",
            suggestion="Ensure energy_transformer.layers.tokens is available",
        ) from e


@register_realiser(PatchEmbedSpec)
def _realise_patch_embed(spec: PatchEmbedSpec, info: SpecInfo) -> nn.Module:
    """Realise a patch embedding specification.

    Parameters
    ----------
    spec : PatchEmbedSpec
        The patch embedding specification.
    info : SpecInfo
        Context information (not used for patch embedding).

    Returns
    -------
    nn.Module
        The patch embedding module.

    Raises
    ------
    RealisationError
        If module cannot be imported.
    """
    try:
        from energy_transformer.layers.embeddings import PatchEmbedding

        return PatchEmbedding(
            img_size=spec.img_size,
            patch_size=spec.patch_size,
            in_chans=spec.in_chans,
            embed_dim=spec.embed_dim,
            bias=spec.bias,
        )
    except ImportError as e:
        raise RealisationError(
            f"Cannot import PatchEmbedding: {e}",
            spec_type="PatchEmbedSpec",
            suggestion="Ensure energy_transformer.layers.embeddings is "
            "available",
        ) from e


@register_realiser(PosEmbedSpec)
def _realise_pos_embed(spec: PosEmbedSpec, info: SpecInfo) -> nn.Module:
    """Realise a positional embedding specification.

    Parameters
    ----------
    spec : PosEmbedSpec
        The positional embedding specification.
    info : SpecInfo
        Context information containing embedding dimension and token count.

    Returns
    -------
    nn.Module
        The positional embedding module.

    Raises
    ------
    RealisationError
        If required context is not available or module cannot be imported.
    """
    if info.embedding_dim is None:
        raise RealisationError(
            "PosEmbedSpec requires embedding dimension from context",
            spec_type="PosEmbedSpec",
            suggestion="Place PosEmbedSpec after a component that "
            "defines embed_dim",
        )

    if info.token_count is None:
        raise RealisationError(
            "PosEmbedSpec requires token count from context",
            spec_type="PosEmbedSpec",
            suggestion="Place PosEmbedSpec after a component that "
            "defines token count",
        )

    try:
        from energy_transformer.layers.embeddings import PositionalEmbedding2D

        return PositionalEmbedding2D(
            num_patches=info.token_count,
            embed_dim=info.embedding_dim,
            include_cls=spec.include_cls,
            init_std=spec.init_std,
        )
    except ImportError as e:
        raise RealisationError(
            f"Cannot import PositionalEmbedding2D: {e}",
            spec_type="PosEmbedSpec",
            suggestion="Ensure energy_transformer.layers.embeddings is "
            "available",
        ) from e


Realise = realise
