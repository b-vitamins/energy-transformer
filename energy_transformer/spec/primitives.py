"""Primitive specification types for Energy Transformer models.

This module provides the foundational building blocks for describing Energy
Transformer architectures. Each primitive spec represents a single component
that can be composed into larger architectures.

Design Principles:
- Immutable data structures (frozen dataclasses)
- Early validation with helpful error messages
- Clear dependency declarations
- Composable by design
- Type-safe interfaces

Example
-------
>>> patch_embed = PatchEmbedSpec(
...     img_size=224, patch_size=16, in_chans=3, embed_dim=768
... )
>>> cls_token = CLSTokenSpec()
>>> et_block = ETSpec(steps=12, alpha=0.125)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

__all__ = [
    # Types
    "EmbeddingDim",
    "TokenCount",
    "ImageSize",
    "PatchSize",
    # Base classes
    "Spec",
    "ValidationError",
    # Primitive specs
    "LayerNormSpec",
    "MHEASpec",
    "HNSpec",
    "ETSpec",
    "CLSTokenSpec",
    "PatchEmbedSpec",
    "PosEmbedSpec",
    # Utilities
    "validate_positive",
    "validate_probability",
    "to_pair",
]

# Type aliases for semantic clarity
EmbeddingDim = int
TokenCount = int
ImageSize = int | tuple[int, int]
PatchSize = int | tuple[int, int]


class ValidationError(ValueError):
    """Raised when a spec fails validation.

    Provides enhanced error messages with context and suggestions.

    Parameters
    ----------
    message : str
        The error message.
    spec_type : str, optional
        The name of the spec type that failed validation.
    suggestion : str, optional
        A helpful suggestion for fixing the error.
    """

    def __init__(
        self,
        message: str,
        spec_type: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Initialize ValidationError with enhanced messaging."""
        self.spec_type = spec_type
        self.suggestion = suggestion

        full_message = message
        if spec_type:
            full_message = f"{spec_type}: {message}"
        if suggestion:
            full_message += f"\nSuggestion: {suggestion}"

        super().__init__(full_message)


def validate_positive(
    value: Any, name: str, spec_type: str | None = None
) -> None:
    """Validate that a value is a positive number.

    Parameters
    ----------
    value : Any
        The value to validate.
    name : str
        The name of the parameter for error messages.
    spec_type : str, optional
        The spec type for error context.

    Raises
    ------
    ValidationError
        If the value is not a positive number.
    """
    if not isinstance(value, int | float) or value <= 0:
        raise ValidationError(
            f"{name} must be a positive number, got {value!r}",
            spec_type=spec_type,
            suggestion=f"Try {name}=1 or another positive value",
        )


def validate_probability(
    value: Any, name: str, spec_type: str | None = None
) -> None:
    """Validate that a value is a valid probability [0, 1].

    Parameters
    ----------
    value : Any
        The value to validate.
    name : str
        The name of the parameter for error messages.
    spec_type : str, optional
        The spec type for error context.

    Raises
    ------
    ValidationError
        If the value is not between 0 and 1.
    """
    if not isinstance(value, int | float) or not (0 <= value <= 1):
        raise ValidationError(
            f"{name} must be a number between 0 and 1, got {value!r}",
            spec_type=spec_type,
            suggestion=f"Try {name}=0.1 for 10% or {name}=0.0 to disable",
        )


def to_pair(
    value: int | tuple[int, int], name: str = "value"
) -> tuple[int, int]:
    """Convert a value to a pair of integers.

    Parameters
    ----------
    value : int or tuple[int, int]
        Value to convert.
    name : str, default="value"
        Name of the parameter for error messages.

    Returns
    -------
    tuple[int, int]
        Pair of integers.

    Raises
    ------
    ValidationError
        If value cannot be converted to a valid pair.
    """
    if isinstance(value, int):
        if value <= 0:
            raise ValidationError(f"{name} must be positive, got {value}")
        return (value, value)
    elif isinstance(value, tuple):
        if len(value) != 2:
            raise ValidationError(
                f"{name} tuple must have exactly 2 elements, got {len(value)}",
                suggestion="Use (height, width) format",
            )
        h, w = value
        if not isinstance(h, int) or not isinstance(w, int):
            raise ValidationError(
                f"{name} tuple elements must be integers, got {value}"
            )
        if h <= 0 or w <= 0:
            raise ValidationError(
                f"{name} tuple elements must be positive, got {value}"
            )
        return (h, w)
    else:
        raise ValidationError(
            f"{name} must be int or tuple[int, int], "
            f"got {type(value).__name__}",
            suggestion="Use 224 for square or (224, 256) for rectangular",
        )


@dataclass(frozen=True)
class Spec(ABC):
    """Base class for all specifications.

    Provides the core interface that all specs must implement for composition,
    validation, and introspection.
    """

    # Class-level metadata (not dataclass fields)
    _spec_type: ClassVar[str] = "base"
    _version: ClassVar[str] = "1.0"

    def __post_init__(self) -> None:
        """Post-initialization hook for validation."""
        self._validate_parameters()

    @abstractmethod
    def _validate_parameters(self) -> None:
        """Validate the parameters of this spec.

        This abstract method must be implemented by all concrete specs
        to validate their specific parameters during initialization.

        Raises
        ------
        ValidationError
            If any parameters are invalid.
        """

    def get_embedding_dim(self) -> EmbeddingDim | None:
        """Get the embedding dimension this spec produces.

        Returns
        -------
        EmbeddingDim | None
            Embedding dimension, or None if this spec doesn't define one.
        """
        return None

    def get_token_count(self) -> TokenCount | None:
        """Get the base token count this spec produces.

        Returns
        -------
        TokenCount | None
            Number of tokens, or None if this spec doesn't define the base
            count.
        """
        return None

    def requires_embedding_dim(self) -> bool:
        """Check whether this spec requires an upstream embedding dimension.

        Returns
        -------
        bool
            True if this spec needs embedding_dim from upstream context.
        """
        return False

    def requires_token_count(self) -> bool:
        """Check whether this spec requires an upstream token count.

        Returns
        -------
        bool
            True if this spec needs token_count from upstream context.
        """
        return False

    def adds_tokens(self) -> int:
        """Get the number of tokens this spec adds to the sequence.

        Returns
        -------
        int
            Number of tokens added (0 for most specs, 1 for CLS token, etc.).
        """
        return 0

    def modifies_tokens(self) -> bool:
        """Check whether this spec modifies existing tokens in place.

        Returns
        -------
        bool
            True if this spec transforms existing tokens without
            adding/removing.
        """
        return True

    def validate(
        self,
        upstream_embedding_dim: EmbeddingDim | None = None,
        upstream_token_count: TokenCount | None = None,
    ) -> None:
        """Validate this spec against upstream context.

        Parameters
        ----------
        upstream_embedding_dim : EmbeddingDim | None, optional
            Embedding dimension from upstream specs.
        upstream_token_count : TokenCount | None, optional
            Token count from upstream specs.

        Raises
        ------
        ValidationError
            If this spec cannot work with the given upstream context.
        """
        spec_name = self.__class__.__name__

        if self.requires_embedding_dim() and upstream_embedding_dim is None:
            raise ValidationError(
                "requires an upstream component that defines embedding_dim",
                spec_type=spec_name,
                suggestion="Place this after PatchEmbedSpec or another "
                "component that sets embed_dim",
            )

        if self.requires_token_count() and upstream_token_count is None:
            raise ValidationError(
                "requires an upstream component that defines token_count",
                spec_type=spec_name,
                suggestion="Place this after PatchEmbedSpec or another "
                "component that sets token count",
            )

    def estimate_params(
        self, context_embedding_dim: EmbeddingDim | None = None
    ) -> int:
        """Estimate the number of parameters for this spec.

        Parameters
        ----------
        context_embedding_dim : EmbeddingDim | None, optional
            Embedding dimension for parameter estimation.

        Returns
        -------
        int
            Estimated parameter count.
        """
        return 0

    def get_info(self) -> dict[str, Any]:
        """Get debugging/introspection information.

        Returns
        -------
        dict[str, Any]
            Information about this spec for debugging.
        """
        return {
            "type": self.__class__.__name__,
            "produces_embedding_dim": self.get_embedding_dim(),
            "produces_token_count": self.get_token_count(),
            "requires_embedding_dim": self.requires_embedding_dim(),
            "requires_token_count": self.requires_token_count(),
            "adds_tokens": self.adds_tokens(),
            "modifies_tokens": self.modifies_tokens(),
        }

    def __str__(self) -> str:
        """Return human-readable string representation.

        Returns
        -------
        str
            String representation of this spec.
        """
        return f"{self.__class__.__name__}"


@dataclass(frozen=True)
class LayerNormSpec(Spec):
    """Specification for layer normalization.

    Layer normalization normalizes inputs across the feature dimension,
    providing stable gradients and improved training dynamics.

    Parameters
    ----------
    eps : float, default=1e-5
        Small value added to denominator for numerical stability.

    Examples
    --------
    >>> LayerNormSpec()  # Default settings
    >>> LayerNormSpec(eps=1e-6)  # Custom settings
    """

    eps: float = 1e-5

    _spec_type: ClassVar[str] = "normalization"

    def _validate_parameters(self) -> None:
        """Validate layer normalization parameters."""
        validate_positive(self.eps, "eps", self.__class__.__name__)

    def requires_embedding_dim(self) -> bool:
        """Check if embedding dimension is required from upstream.

        Returns
        -------
        bool
            Always True, as layer norm needs to know the feature dimension.
        """
        return True

    def estimate_params(
        self, context_embedding_dim: EmbeddingDim | None = None
    ) -> int:
        """Estimate parameter count for layer normalization.

        Parameters
        ----------
        context_embedding_dim : EmbeddingDim | None, optional
            The embedding dimension context.

        Returns
        -------
        int
            Estimated parameter count (2 * embed_dim for scale and bias).
        """
        if context_embedding_dim is None:
            return 0
        return 2 * context_embedding_dim  # scale and bias parameters


@dataclass(frozen=True)
class MHEASpec(Spec):
    """Specification for multi-head energy attention.

    Defines the attention mechanism used in Energy Transformer blocks.
    Uses energy-based formulation where attention weights are derived
    from energy function gradients.

    Parameters
    ----------
    num_heads : int, default=12
        Number of parallel attention heads.
    head_dim : int, default=64
        Dimension of each attention head.
    beta : float, optional
        Temperature parameter for attention. If None, uses 1/sqrt(head_dim).
    bias : bool, default=False
        Whether to use bias in key/query projections.
    dropout : float, default=0.0
        Dropout probability for attention weights.

    Examples
    --------
    >>> MHEASpec()  # 12 heads, 64 dim each
    >>> MHEASpec(num_heads=16, head_dim=48)  # Different head config
    >>> MHEASpec(beta=0.1, dropout=0.1)  # Custom temperature/dropout
    """

    num_heads: int = 12
    head_dim: int = 64
    beta: float | None = None
    bias: bool = False
    dropout: float = 0.0

    _spec_type: ClassVar[str] = "attention"

    def _validate_parameters(self) -> None:
        """Validate attention parameters."""
        spec_name = self.__class__.__name__

        validate_positive(self.num_heads, "num_heads", spec_name)
        validate_positive(self.head_dim, "head_dim", spec_name)
        validate_probability(self.dropout, "dropout", spec_name)

        if self.beta is not None:
            validate_positive(self.beta, "beta", spec_name)

        # Validate reasonable attention dimensions
        total_dim = self.num_heads * self.head_dim
        if total_dim > 4096:
            raise ValidationError(
                f"Total attention dimension ({total_dim}) is very large",
                spec_type=spec_name,
                suggestion="Consider reducing num_heads or head_dim",
            )

    def requires_embedding_dim(self) -> bool:
        """Check if embedding dimension is required from upstream.

        Returns
        -------
        bool
            Always True, as attention needs the input embedding dimension.
        """
        return True

    def get_effective_beta(self) -> float:
        """Get the effective beta value (computed default if not specified).

        Returns
        -------
        float
            The effective beta temperature parameter.
        """
        return (
            self.beta
            if self.beta is not None
            else 1.0 / math.sqrt(self.head_dim)
        )

    def estimate_params(
        self, context_embedding_dim: EmbeddingDim | None = None
    ) -> int:
        """Estimate parameter count for attention.

        Parameters
        ----------
        context_embedding_dim : EmbeddingDim | None, optional
            The embedding dimension context.

        Returns
        -------
        int
            Estimated parameter count for query/key projections.
        """
        if context_embedding_dim is None:
            return 0

        query_key_params = (
            2 * self.num_heads * self.head_dim * context_embedding_dim
        )
        bias_params = 2 * self.num_heads * self.head_dim if self.bias else 0
        return query_key_params + bias_params


@dataclass(frozen=True)
class HNSpec(Spec):
    """Specification for Hopfield network memory component.

    Defines the associative memory component that provides energy-based
    pattern completion and memory retrieval capabilities.

    Parameters
    ----------
    hidden_dim : int, optional
        Number of memory patterns. If None, computed as in_dim * multiplier.
    multiplier : float, default=4.0
        Multiplier for hidden dimension when hidden_dim is None.
    bias : bool, default=False
        Whether to include bias terms in memory patterns.
    activation : {"relu", "softmax", "power", "tanh"}, default="relu"
        Activation function for energy computation.
    dropout : float, default=0.0
        Dropout probability for memory patterns.

    Examples
    --------
    >>> HNSpec()  # Default: 4x multiplier, ReLU activation
    >>> HNSpec(hidden_dim=2048)  # Fixed hidden dimension
    >>> HNSpec(activation="softmax", dropout=0.1)  # Different config
    """

    hidden_dim: int | None = None
    multiplier: float = 4.0
    bias: bool = False
    activation: Literal["relu", "softmax", "power", "tanh"] = "relu"
    dropout: float = 0.0

    _spec_type: ClassVar[str] = "memory"

    def _validate_parameters(self) -> None:
        """Validate Hopfield network parameters."""
        spec_name = self.__class__.__name__

        if self.hidden_dim is not None:
            validate_positive(self.hidden_dim, "hidden_dim", spec_name)
        validate_positive(self.multiplier, "multiplier", spec_name)
        validate_probability(self.dropout, "dropout", spec_name)

        # Validate multiplier isn't too large
        if self.multiplier > 8.0:
            raise ValidationError(
                f"Multiplier {self.multiplier} may create very large "
                "hidden dimensions",
                spec_type=spec_name,
                suggestion="Consider multiplier <= 8.0 or set hidden_dim "
                "explicitly",
            )

    def requires_embedding_dim(self) -> bool:
        """Check if embedding dimension is required from upstream.

        Returns
        -------
        bool
            Always True, as Hopfield network needs input embedding dimension.
        """
        return True

    def get_effective_hidden_dim(self, embedding_dim: EmbeddingDim) -> int:
        """Get the effective hidden dimension given an embedding dimension.

        Parameters
        ----------
        embedding_dim : EmbeddingDim
            The input embedding dimension.

        Returns
        -------
        int
            The effective hidden dimension.
        """
        if self.hidden_dim is not None:
            return self.hidden_dim
        return int(embedding_dim * self.multiplier)

    def estimate_params(
        self, context_embedding_dim: EmbeddingDim | None = None
    ) -> int:
        """Estimate parameter count for Hopfield network.

        Parameters
        ----------
        context_embedding_dim : EmbeddingDim | None, optional
            The embedding dimension context.

        Returns
        -------
        int
            Estimated parameter count for memory patterns.
        """
        if context_embedding_dim is None:
            return 0

        hidden_dim = self.get_effective_hidden_dim(context_embedding_dim)
        memory_params = hidden_dim * context_embedding_dim
        bias_params = hidden_dim if self.bias else 0
        return memory_params + bias_params


@dataclass(frozen=True)
class ETSpec(Spec):
    """Specification for an Energy Transformer block.

    Represents a complete Energy Transformer block that combines layer
    normalization, attention, and Hopfield network components with
    energy-based optimization.

    Parameters
    ----------
    steps : int, default=12
        Number of gradient descent steps for energy optimization.
    alpha : float, default=0.125
        Step size for energy optimization.
    layer_norm : LayerNormSpec, default=LayerNormSpec()
        Specification for layer normalization component.
    attention : MHEASpec, default=MHEASpec()
        Specification for attention component.
    hopfield : HNSpec, default=HNSpec()
        Specification for Hopfield network component.

    Examples
    --------
    >>> ETSpec()  # Default configuration
    >>> ETSpec(steps=8, alpha=0.2)  # Faster optimization
    >>> ETSpec(
    ...     attention=MHEASpec(num_heads=16),
    ...     hopfield=HNSpec(activation="softmax")
    ... )  # Custom components
    """

    steps: int = 12
    alpha: float = 0.125
    layer_norm: LayerNormSpec = field(default_factory=LayerNormSpec)
    attention: MHEASpec = field(default_factory=MHEASpec)
    hopfield: HNSpec = field(default_factory=HNSpec)

    _spec_type: ClassVar[str] = "transformer_block"

    def _validate_parameters(self) -> None:
        """Validate Energy Transformer block parameters."""
        spec_name = self.__class__.__name__

        validate_positive(self.steps, "steps", spec_name)
        validate_positive(self.alpha, "alpha", spec_name)

        # Validate reasonable optimization parameters
        if self.steps > 50:
            raise ValidationError(
                f"Very high step count ({self.steps}) may be slow",
                spec_type=spec_name,
                suggestion="Consider steps <= 50 for reasonable speed",
            )

        if self.alpha > 1.0:
            raise ValidationError(
                f"Large step size ({self.alpha}) may cause instability",
                spec_type=spec_name,
                suggestion="Consider alpha <= 1.0 for stable optimization",
            )

    def requires_embedding_dim(self) -> bool:
        """Check if embedding dimension is required from upstream.

        Returns
        -------
        bool
            Always True, as all components need embedding dimension.
        """
        return True

    def estimate_params(
        self, context_embedding_dim: EmbeddingDim | None = None
    ) -> int:
        """Estimate parameter count for Energy Transformer block.

        Parameters
        ----------
        context_embedding_dim : EmbeddingDim | None, optional
            The embedding dimension context.

        Returns
        -------
        int
            Estimated parameter count for all components.
        """
        if context_embedding_dim is None:
            return 0

        return (
            self.layer_norm.estimate_params(context_embedding_dim)
            + self.attention.estimate_params(context_embedding_dim)
            + self.hopfield.estimate_params(context_embedding_dim)
        )


@dataclass(frozen=True)
class CLSTokenSpec(Spec):
    """Specification for learnable classification token.

    Adds a special classification token at the beginning of the sequence
    that can aggregate information from all other tokens through attention.

    Examples
    --------
    >>> CLSTokenSpec()
    """

    _spec_type: ClassVar[str] = "special_token"

    def _validate_parameters(self) -> None:
        """Validate CLS token parameters.

        CLS token has no parameters to validate.
        """

    def requires_embedding_dim(self) -> bool:
        """Check if embedding dimension is required from upstream.

        Returns
        -------
        bool
            Always True, as CLS token needs to know embedding dimension.
        """
        return True

    def adds_tokens(self) -> int:
        """Get the number of tokens this spec adds.

        Returns
        -------
        int
            Always 1, as CLS token adds exactly one token.
        """
        return 1

    def modifies_tokens(self) -> bool:
        """Check if this spec modifies existing tokens.

        Returns
        -------
        bool
            Always False, as CLS token adds rather than modifies.
        """
        return False  # Adds token, doesn't modify existing ones

    def estimate_params(
        self, context_embedding_dim: EmbeddingDim | None = None
    ) -> int:
        """Estimate parameter count for CLS token.

        Parameters
        ----------
        context_embedding_dim : EmbeddingDim | None, optional
            The embedding dimension context.

        Returns
        -------
        int
            Estimated parameter count (one token vector).
        """
        if context_embedding_dim is None:
            return 0
        return context_embedding_dim  # One learnable token vector


@dataclass(frozen=True)
class PatchEmbedSpec(Spec):
    """Specification for patch embedding layer.

    Converts input images into sequences of patch embeddings by dividing
    the image into non-overlapping patches and projecting each patch
    to the embedding dimension.

    Parameters
    ----------
    img_size : int or tuple[int, int]
        Input image size. If int, assumes square images.
    patch_size : int or tuple[int, int]
        Size of each patch. If int, assumes square patches.
    embed_dim : int
        Output embedding dimension.
    in_chans : int, default=3
        Number of input image channels.
    bias : bool, default=True
        Whether to include bias in the projection layer.
    flatten : bool, default=True
        Whether to flatten spatial dimensions of patches.

    Examples
    --------
    >>> PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768)
    >>> PatchEmbedSpec(
    ...     img_size=(224, 256), patch_size=(16, 16), embed_dim=512
    ... )
    >>> PatchEmbedSpec(
    ...     img_size=384, patch_size=32, in_chans=1, embed_dim=1024
    ... )
    """

    img_size: ImageSize = field()
    patch_size: PatchSize = field()
    embed_dim: EmbeddingDim = field()
    in_chans: int = 3
    bias: bool = True
    flatten: bool = True

    _spec_type: ClassVar[str] = "embedding"

    def _validate_parameters(self) -> None:
        """Validate patch embedding parameters."""
        spec_name = self.__class__.__name__

        # Validate and normalize sizes
        try:
            img_h, img_w = to_pair(self.img_size, "img_size")
            patch_h, patch_w = to_pair(self.patch_size, "patch_size")
        except ValidationError as e:
            e.spec_type = spec_name
            raise

        validate_positive(self.in_chans, "in_chans", spec_name)
        validate_positive(self.embed_dim, "embed_dim", spec_name)

        # Validate patch size divides image size evenly
        if img_h % patch_h != 0:
            raise ValidationError(
                f"Image height ({img_h}) must be divisible by patch height "
                f"({patch_h})",
                spec_type=spec_name,
                suggestion=f"Try img_size={img_h - (img_h % patch_h)} or "
                f"patch_size that divides {img_h}",
            )
        if img_w % patch_w != 0:
            raise ValidationError(
                f"Image width ({img_w}) must be divisible by patch width "
                f"({patch_w})",
                spec_type=spec_name,
                suggestion=f"Try img_size={img_w - (img_w % patch_w)} or "
                f"patch_size that divides {img_w}",
            )

        # Validate reasonable sizes
        num_patches = (img_h // patch_h) * (img_w // patch_w)
        if num_patches > 10000:
            raise ValidationError(
                f"Very large number of patches ({num_patches})",
                spec_type=spec_name,
                suggestion="Consider larger patch_size or smaller img_size",
            )

    def get_embedding_dim(self) -> EmbeddingDim:
        """Get the embedding dimension this spec produces.

        Returns
        -------
        EmbeddingDim
            The output embedding dimension.
        """
        return self.embed_dim

    def get_token_count(self) -> TokenCount:
        """Get the number of patch tokens produced.

        Returns
        -------
        TokenCount
            Number of patch tokens.
        """
        img_h, img_w = to_pair(self.img_size)
        patch_h, patch_w = to_pair(self.patch_size)
        return (img_h // patch_h) * (img_w // patch_w)

    def modifies_tokens(self) -> bool:
        """Check if this spec modifies existing tokens.

        Returns
        -------
        bool
            Always False, as patch embedding creates initial tokens.
        """
        return False  # Creates initial tokens

    def estimate_params(
        self, context_embedding_dim: EmbeddingDim | None = None
    ) -> int:
        """Estimate parameter count for patch embedding.

        Parameters
        ----------
        context_embedding_dim : EmbeddingDim | None, optional
            Not used for patch embedding.

        Returns
        -------
        int
            Estimated parameter count for convolutional projection.
        """
        patch_h, patch_w = to_pair(self.patch_size)
        conv_params = self.in_chans * (patch_h * patch_w) * self.embed_dim
        bias_params = self.embed_dim if self.bias else 0
        return conv_params + bias_params


@dataclass(frozen=True)
class PosEmbedSpec(Spec):
    """Specification for positional embeddings.

    Adds learnable positional information to patch embeddings so the model
    can understand spatial relationships between patches.

    Parameters
    ----------
    include_cls : bool, default=False
        Whether to include a position for the CLS token.
    init_std : float, default=0.02
        Standard deviation for parameter initialization.

    Examples
    --------
    >>> PosEmbedSpec()  # Basic positional embeddings
    >>> PosEmbedSpec(include_cls=True)  # Include CLS token position
    >>> PosEmbedSpec(init_std=0.01)  # Custom initialization
    """

    include_cls: bool = False
    init_std: float = 0.02

    _spec_type: ClassVar[str] = "embedding"

    def _validate_parameters(self) -> None:
        """Validate positional embedding parameters."""
        validate_positive(self.init_std, "init_std", self.__class__.__name__)

    def requires_embedding_dim(self) -> bool:
        """Check if embedding dimension is required from upstream.

        Returns
        -------
        bool
            Always True, as positional embeddings need embedding dimension.
        """
        return True

    def requires_token_count(self) -> bool:
        """Check if token count is required from upstream.

        Returns
        -------
        bool
            Always True, as positional embeddings need to know sequence
            length.
        """
        return True

    def modifies_tokens(self) -> bool:
        """Check if this spec modifies existing tokens.

        Returns
        -------
        bool
            Always True, as positional embeddings add to existing tokens.
        """
        return True  # Adds positional info to existing tokens

    def estimate_params(
        self, context_embedding_dim: EmbeddingDim | None = None
    ) -> int:
        """Estimate parameter count for positional embeddings.

        Parameters
        ----------
        context_embedding_dim : EmbeddingDim | None, optional
            The embedding dimension context.

        Returns
        -------
        int
            Cannot estimate without token count, returns 0.
        """
        # Cannot estimate without knowing token count
        return 0

    def estimate_params_with_context(
        self, embedding_dim: EmbeddingDim, token_count: TokenCount
    ) -> int:
        """Estimate parameters with full context information.

        Parameters
        ----------
        embedding_dim : EmbeddingDim
            The embedding dimension.
        token_count : TokenCount
            The number of tokens in the sequence.

        Returns
        -------
        int
            Estimated parameter count for positional embeddings.
        """
        seq_len = token_count + (1 if self.include_cls else 0)
        return seq_len * embedding_dim
