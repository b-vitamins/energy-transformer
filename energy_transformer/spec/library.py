"""Common layer specifications for Energy Transformer models.

This module provides pre-built specifications for common layers used in
transformer architectures, particularly vision transformers.
"""

from dataclasses import dataclass
from typing import Literal

from energy_transformer.layers.types import PoolType

from .primitives import (
    Context,
    Dimension,
    Spec,
    modifies,
    param,
    provides,
    requires,
)

__all__ = [
    "CLSTokenSpec",
    # Head specs
    "ClassificationHeadSpec",
    # Utility specs
    "DropoutSpec",
    "ETBlockSpec",
    "HNSpec",
    "IdentitySpec",
    "LayerNormSpec",
    "MHASpec",
    "MHEASpec",
    "MLPSpec",
    # Core layer specs
    "PatchEmbedSpec",
    "PosEmbedSpec",
    "SHNSpec",
    "TransformerBlockSpec",
    # Composite specs
    "VisionEmbeddingSpec",
    # Utility functions
    "to_pair",
    "validate_dimension",
    "validate_positive",
    "validate_probability",
]

MAX_DIM: int = 65536
MAX_MULTIPLIER: float = 8.0
MAX_COMPLEX_DIM: int = 3
MAX_STEPS: int = 50
DEFAULT_EPS: float = 1e-5
DEFAULT_INIT_STD: float = 0.02
DEFAULT_TEMPERATURE: float = 0.5


# Utility functions
def to_pair(x: int | tuple[int, int]) -> tuple[int, int]:
    """Convert single int or pair to tuple of two ints."""
    return (x, x) if isinstance(x, int) else x


def validate_positive(x: int | float | tuple[int | float, ...]) -> bool:
    """Validate that a value is positive."""
    if isinstance(x, tuple):
        return all(isinstance(v, int | float) and v > 0 for v in x)
    return isinstance(x, int | float) and x > 0


def validate_probability(x: float) -> bool:
    """Validate that a value is a valid probability [0, 1]."""
    return 0 <= x <= 1


def validate_dimension(x: int) -> bool:
    """Validate dimension is positive and reasonable."""
    return 0 < x <= MAX_DIM  # Reasonable upper bound


# Core Energy Transformer layer specifications


@dataclass(frozen=True)
@requires("embed_dim")
class LayerNormSpec(Spec):
    """Layer normalization specification.

    Parameters
    ----------
    eps : float
        Epsilon for numerical stability
    """

    eps: float = param(default=DEFAULT_EPS, validator=validate_positive)


@dataclass(frozen=True)
@provides("embed_dim", "num_patches")
class PatchEmbedSpec(Spec):
    """Patch embedding specification for vision transformers.

    Parameters
    ----------
    img_size : int | tuple[int, int]
        Input image size (height, width) or single int for square
    patch_size : int | tuple[int, int]
        Size of each patch (height, width) or single int for square
    embed_dim : int
        Output embedding dimension
    in_chans : int
        Number of input channels
    bias : bool
        Whether to include bias in projection
    """

    img_size: int | tuple[int, int] = param(validator=validate_positive)
    patch_size: int | tuple[int, int] = param(validator=validate_positive)
    embed_dim: int = param(validator=validate_dimension)
    in_chans: int = param(default=3, validator=validate_positive)
    bias: bool = param(default=True)

    def apply_context(self, context: Context) -> Context:
        """Apply patch embedding specification to context."""
        context = super().apply_context(context)

        # Calculate and provide patch count
        img_h, img_w = to_pair(self.img_size)
        patch_h, patch_w = to_pair(self.patch_size)
        num_patches = (img_h // patch_h) * (img_w // patch_w)

        context.set_dim("num_patches", num_patches)
        context.set_dim("embed_dim", self.embed_dim)

        return context

    def validate(self, context: Context) -> list[str]:
        """Validate patch embedding spec."""
        issues = super().validate(context)

        # Check patch size vs image size
        img_h, img_w = to_pair(self.img_size)
        patch_h, patch_w = to_pair(self.patch_size)

        if patch_h > img_h or patch_w > img_w:
            issues.append(
                f"Patch size ({patch_h}, {patch_w}) cannot be larger than "
                f"image size ({img_h}, {img_w})",
            )

        return issues


@dataclass(frozen=True)
@requires("embed_dim")
@modifies("num_patches")
class CLSTokenSpec(Spec):
    """Classification token specification.

    Adds a learnable classification token to the sequence.
    """

    def apply_context(self, context: Context) -> Context:
        """Apply CLS token specification to context."""
        context = super().apply_context(context)

        # Increment patch count for CLS token
        if num_patches := context.get_dim("num_patches"):
            context.set_dim("num_patches", num_patches + 1)

        return context


@dataclass(frozen=True)
@requires("embed_dim", "num_patches")
class PosEmbedSpec(Spec):
    """Positional embedding specification.

    Parameters
    ----------
    include_cls : bool
        Whether to include position for CLS token
    init_std : float
        Standard deviation for initialization
    """

    include_cls: bool = param(default=False)
    init_std: float = param(
        default=DEFAULT_INIT_STD, validator=validate_positive
    )


@dataclass(frozen=True)
@requires("embed_dim")
class MHEASpec(Spec):
    """Multi-head energy attention specification.

    Parameters
    ----------
    num_heads : int
        Number of attention heads
    head_dim : int
        Dimension of each attention head
    beta : float | None
        Temperature parameter (None for 1/sqrt(head_dim))
    bias : bool
        Whether to include bias in key/query projections
    init_std : float
        Standard deviation for weight initialization
    """

    num_heads: int = param(default=12, validator=validate_positive)
    head_dim: int = param(default=64, validator=validate_dimension)
    beta: float | None = param(default=None)
    bias: bool = param(default=False)
    init_std: float = param(default=0.002, validator=validate_positive)


@dataclass(frozen=True)
@requires("embed_dim")
class HNSpec(Spec):
    """Hopfield network specification.

    Parameters
    ----------
    hidden_dim : Dimension | None
        Hidden dimension (computed as embed_dim * multiplier by default)
    multiplier : float
        Multiplier for hidden dimension
    energy_fn : str
        Energy function type (could be extended to support custom functions)
    """

    hidden_dim: Dimension | None = param(default=None, dimension=True)
    multiplier: float = param(
        default=4.0, validator=lambda x: 0 < x <= MAX_MULTIPLIER
    )
    energy_fn: str = param(
        default="relu_squared",
        choices=["relu_squared", "softmax", "tanh"],
    )

    def apply_context(self, context: Context) -> Context:
        """Apply Hopfield network specification to context."""
        context = super().apply_context(context)

        if self.hidden_dim is None:
            if embed_dim := context.get_dim("embed_dim"):
                computed_hidden = int(embed_dim * self.multiplier)
                context.set_dim("hopfield_hidden_dim", computed_hidden)
        elif isinstance(self.hidden_dim, Dimension):
            if resolved := self.hidden_dim.resolve(context):
                context.set_dim("hopfield_hidden_dim", resolved)
        elif isinstance(self.hidden_dim, int):
            context.set_dim("hopfield_hidden_dim", self.hidden_dim)

        return context

    def validate(self, context: Context) -> list[str]:
        """Validate with proper hidden_dim handling."""
        issues = super().validate(context)

        if self.hidden_dim is None and context.get_dim("embed_dim") is None:
            issues.append(
                "Cannot compute hidden_dim: embed_dim not available in context",
            )

        return issues


@dataclass(frozen=True)
@requires("embed_dim")
class SHNSpec(Spec):
    """Simplicial Hopfield network specification.

    Parameters
    ----------
    simplices : list[list[int]] | None
        Manual specification of simplicial complex
    num_vertices : int | None
        Number of vertices (defaults to num_patches from context)
    max_dim : int
        Maximum simplex dimension (1=edges, 2=triangles, 3=tetrahedra)
    budget : float
        Fraction of full edge budget to use
    dim_weights : dict[int, float] | None
        Weight distribution across simplex dimensions
    coordinates : list[tuple[float, float]] | None
        Spatial coordinates for topology-aware generation
    hidden_dim : Dimension | None
        Hidden dimension (computed as embed_dim * multiplier by default)
    multiplier : float
        Multiplier for hidden dimension
    temperature : float
        Temperature parameter for softmax
    """

    # Simplicial complex parameters
    simplices: list[list[int]] | None = param(default=None)
    num_vertices: int | None = param(
        default=None,
        validator=lambda x: x is None or x > 0,
    )
    max_dim: int = param(
        default=1, validator=lambda x: 1 <= x <= MAX_COMPLEX_DIM
    )
    budget: float = param(default=0.1, validator=lambda x: 0 < x <= 1)
    dim_weights: dict[int, float] | None = param(default=None)
    coordinates: list[tuple[float, float]] | None = param(default=None)

    # Network parameters
    hidden_dim: Dimension | None = param(default=None, dimension=True)
    multiplier: float = param(
        default=4.0, validator=lambda x: 0 < x <= MAX_MULTIPLIER
    )
    temperature: float = param(
        default=DEFAULT_TEMPERATURE, validator=validate_positive
    )

    def apply_context(self, context: Context) -> Context:
        """Apply simplicial Hopfield specification to context."""
        context = super().apply_context(context)

        # Set hidden_dim if not explicitly provided
        if self.hidden_dim is None:
            if embed_dim := context.get_dim("embed_dim"):
                computed_hidden = int(embed_dim * self.multiplier)
                context.set_dim("simplicial_hidden_dim", computed_hidden)
        elif isinstance(self.hidden_dim, Dimension):
            if resolved := self.hidden_dim.resolve(context):
                context.set_dim("simplicial_hidden_dim", resolved)
        elif isinstance(self.hidden_dim, int):
            context.set_dim("simplicial_hidden_dim", self.hidden_dim)

        # Set num_vertices from context if not provided
        if self.num_vertices is None and (
            num_patches := context.get_dim("num_patches")
        ):
            context.set_dim("simplicial_vertices", num_patches)

        return context


@dataclass(frozen=True)
@requires("embed_dim")
class ETBlockSpec(Spec):
    """Energy Transformer block specification.

    Parameters
    ----------
    steps : int
        Number of energy minimization steps
    alpha : float
        Step size for energy minimization
    layer_norm : LayerNormSpec
        Layer normalization specification
    attention : MHEASpec
        Multi-head attention specification
    hopfield : HNSpec | SHNSpec
        Hopfield network specification (standard or simplicial)
    """

    steps: int = param(default=12, validator=lambda x: 0 < x <= MAX_STEPS)
    alpha: float = param(default=0.125, validator=validate_positive)
    layer_norm: LayerNormSpec = param(default_factory=LayerNormSpec)
    attention: MHEASpec = param(default_factory=MHEASpec)
    hopfield: HNSpec | SHNSpec = param(default_factory=HNSpec)


# Head specifications


@dataclass(frozen=True)
@requires("embed_dim")
@provides("num_classes")
class ClassificationHeadSpec(Spec):
    """Classifier head specification.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    pool_type : PoolType
        Pooling type: ``"avg"``, ``"max"``, ``"token"``, or ``"none"``.
    drop_rate : float
        Dropout rate before classification.
    use_conv : bool
        Use convolutional classifier. Only valid with ``pool_type='none'``.
    bias : bool
        Use bias in classifier layer.
    """

    num_classes: int = param(validator=validate_positive)
    pool_type: PoolType = param(default="token")
    drop_rate: float = param(default=0.0, validator=validate_probability)
    use_conv: bool = param(default=False)
    bias: bool = param(default=True)

    def apply_context(self, context: Context) -> Context:
        """Apply classification head to context."""
        context = super().apply_context(context)
        context.set_dim("num_classes", self.num_classes)
        return context


# Composite specifications


@dataclass(frozen=True)
@provides("embed_dim", "num_patches")
class VisionEmbeddingSpec(Spec):
    """Combined patch + positional + CLS token embedding.

    This is a composite spec that represents the complete
    vision embedding pipeline typically used in ViT models.

    Parameters
    ----------
    img_size : int | tuple[int, int]
        Input image size
    patch_size : int | tuple[int, int]
        Patch size
    embed_dim : int
        Embedding dimension
    in_chans : int
        Number of input channels
    use_cls_token : bool
        Whether to add CLS token
    drop_rate : float
        Dropout rate after positional embedding
    """

    img_size: int | tuple[int, int] = param(validator=validate_positive)
    patch_size: int | tuple[int, int] = param(validator=validate_positive)
    embed_dim: int = param(validator=validate_dimension)
    in_chans: int = param(default=3, validator=validate_positive)
    use_cls_token: bool = param(default=True)
    drop_rate: float = param(default=0.0, validator=validate_probability)

    def apply_context(self, context: Context) -> Context:
        """Apply vision embedding to context."""
        context = super().apply_context(context)

        # Calculate patches
        img_h, img_w = to_pair(self.img_size)
        patch_h, patch_w = to_pair(self.patch_size)
        num_patches = (img_h // patch_h) * (img_w // patch_w)

        # Add CLS token if used
        if self.use_cls_token:
            num_patches += 1

        context.set_dim("embed_dim", self.embed_dim)
        context.set_dim("num_patches", num_patches)
        return context


# Standard transformer components (for comparison/baseline)


@dataclass(frozen=True)
@requires("embed_dim")
class MHASpec(Spec):
    """Standard multi-head self-attention specification.

    Parameters
    ----------
    num_heads : int
        Number of attention heads
    qkv_bias : bool
        Whether to use bias in QKV projection
    attn_drop : float
        Attention dropout rate
    proj_drop : float
        Projection dropout rate
    """

    num_heads: int = param(default=8, validator=validate_positive)
    qkv_bias: bool = param(default=True)
    attn_drop: float = param(default=0.0, validator=validate_probability)
    proj_drop: float = param(default=0.0, validator=validate_probability)


@dataclass(frozen=True)
@requires("embed_dim")
class MLPSpec(Spec):
    """Standard MLP/FFN block specification.

    Parameters
    ----------
    hidden_features : int | None
        Hidden dimension (defaults to 4 * embed_dim)
    out_features : int | None
        Output dimension (defaults to embed_dim)
    activation : str
        Activation function name
    drop : float
        Dropout rate
    """

    Activation = Literal["gelu", "relu", "swish", "silu"]
    hidden_features: int | None = param(default=None)
    out_features: int | None = param(default=None)
    activation: Activation = param(default="gelu")
    drop: float = param(default=0.0, validator=validate_probability)

    def validate(self, context: Context) -> list[str]:
        """Validate MLP output dimension."""
        issues = super().validate(context)
        out = (
            self.out_features
            if self.out_features is not None
            else context.get_dim("embed_dim")
        )
        embed = context.get_dim("embed_dim")
        if out != embed:
            issues.append("Incompatible dimensions")
        return issues


@dataclass(frozen=True)
@requires("embed_dim")
class TransformerBlockSpec(Spec):
    """Standard transformer block specification.

    Represents a standard pre-norm or post-norm transformer block
    with attention and MLP, used for baseline comparisons.

    Parameters
    ----------
    attention : MHASpec
        Attention specification
    mlp : MLPSpec
        MLP specification
    drop_path : float
        Drop path (stochastic depth) rate
    norm_first : bool
        Whether to use pre-normalization (True) or post-normalization
    """

    attention: MHASpec = param(default_factory=MHASpec)
    mlp: MLPSpec = param(default_factory=MLPSpec)
    drop_path: float = param(default=0.0, validator=validate_probability)
    norm_first: bool = param(default=True)


# Utility specifications


@dataclass(frozen=True)
class DropoutSpec(Spec):
    """Dropout layer specification.

    Parameters
    ----------
    p : float
        Dropout probability
    inplace : bool
        Whether to perform dropout in-place
    """

    p: float = param(default=0.5, validator=validate_probability)
    inplace: bool = param(default=False)


@dataclass(frozen=True)
class IdentitySpec(Spec):
    """Identity layer specification.

    Passes input through unchanged. Useful for optional layers
    or as a no-op in conditional architectures.
    """
