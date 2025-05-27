"""Common layer specifications for Energy Transformer models.

This module provides pre-built specifications for common layers used in
transformer architectures, particularly vision transformers.
"""

from dataclasses import dataclass

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
    # Layer specs
    "LayerNormSpec",
    "PatchEmbedSpec",
    "CLSTokenSpec",
    "PosEmbedSpec",
    "MHEASpec",
    "HNSpec",
    "ETSpec",
    # Utility functions
    "to_pair",
    "validate_positive",
    "validate_probability",
]


# Utility functions
def to_pair(x: int | tuple[int, int]) -> tuple[int, int]:
    """Convert single int or pair to tuple of two ints."""
    return (x, x) if isinstance(x, int) else x


def validate_positive(x: int | float) -> bool:
    """Validate that a value is positive."""
    return x > 0


def validate_probability(x: float) -> bool:
    """Validate that a value is a valid probability [0, 1]."""
    return 0 <= x <= 1


# Layer specifications
@dataclass(frozen=True)
@requires("embed_dim")
class LayerNormSpec(Spec):
    """Layer normalization specification.

    Parameters
    ----------
    eps : float
        Epsilon for numerical stability
    """

    eps: float = param(default=1e-5, validator=validate_positive)


@dataclass(frozen=True)
@provides("embed_dim", "token_count")
class PatchEmbedSpec(Spec):
    """Patch embedding specification for vision transformers.

    Parameters
    ----------
    img_size : int
        Input image size (assumes square images)
    patch_size : int
        Size of each patch
    embed_dim : int
        Output embedding dimension
    in_chans : int
        Number of input channels
    """

    img_size: int = param(validator=validate_positive)
    patch_size: int = param(validator=validate_positive)
    embed_dim: int = param(validator=validate_positive)
    in_chans: int = param(default=3, validator=validate_positive)

    def apply_context(self, context: Context) -> Context:
        """Apply patch embedding specification to context.

        Updates the context with the embedding dimension and calculated
        token count based on image and patch sizes.

        Parameters
        ----------
        context : Context
            Context to update

        Returns
        -------
        Context
            Updated context with embed_dim and token_count dimensions
        """
        context = super().apply_context(context)

        # Calculate and provide token count
        num_patches = (self.img_size // self.patch_size) ** 2
        context.set_dim("token_count", num_patches)
        context.set_dim("embed_dim", self.embed_dim)

        return context


@dataclass(frozen=True)
@requires("embed_dim", "token_count")
@modifies("token_count")
class CLSTokenSpec(Spec):
    """Classification token specification.

    Adds a learnable classification token to the sequence.
    """

    def apply_context(self, context: Context) -> Context:
        """Apply CLS token specification to context.

        Increments the token count in the context to account for the
        additional classification token.

        Parameters
        ----------
        context : Context
            Context to update

        Returns
        -------
        Context
            Updated context with incremented token_count
        """
        context = super().apply_context(context)

        # Increment token count for CLS token
        if token_count := context.get_dim("token_count"):
            context.set_dim("token_count", token_count + 1)

        return context


@dataclass(frozen=True)
@requires("embed_dim", "token_count")
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
    init_std: float = param(default=0.02, validator=validate_positive)


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
    dropout : float
        Dropout probability
    """

    num_heads: int = param(default=12, validator=validate_positive)
    head_dim: int = param(default=64, validator=validate_positive)
    dropout: float = param(default=0.0, validator=validate_probability)


@dataclass(frozen=True)
@requires("embed_dim")
class HNSpec(Spec):
    """Hopfield network specification.

    Parameters
    ----------
    hidden_dim : Dimension
        Hidden dimension (computed as embed_dim * 4 by default)
    multiplier : float
        Multiplier for hidden dimension
    """

    hidden_dim: Dimension = param(
        default_factory=lambda: Dimension(
            "hidden_dim", formula="embed_dim * 4"
        ),
        dimension=True,
    )
    multiplier: float = param(default=4.0, validator=lambda x: 0 < x <= 8)


@dataclass(frozen=True)
@requires("embed_dim")
class ETSpec(Spec):
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
    hopfield : HNSpec
        Hopfield network specification
    """

    steps: int = param(default=12, validator=lambda x: 0 < x <= 50)
    alpha: float = param(default=0.125, validator=lambda x: 0 < x <= 1)
    layer_norm: LayerNormSpec = param(default_factory=LayerNormSpec)
    attention: MHEASpec = param(default_factory=MHEASpec)
    hopfield: HNSpec = param(default_factory=HNSpec)
