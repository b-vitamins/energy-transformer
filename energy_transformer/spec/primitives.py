"""Primitive specification types for Energy Transformer models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "Spec",
    "PatchSpec",
    "PosEncSpec",
    "CLSTokenSpec",
    "MaskTokenSpec",
    "ETBlockSpec",
    "NormSpec",
]

# Type for embedding dimensions
EmbeddingDim = int

# Type for token counts
TokenCount = int


@dataclass(frozen=True)
class Spec:
    """Base class for all specifications, providing common functionality."""

    def get_embedding_dim(self) -> EmbeddingDim | None:
        """Get the embedding dimension produced/expected by this spec.

        Returns
        -------
        Optional[EmbeddingDim]
            The embedding dimension, or None if not applicable.
        """
        return None

    def get_token_count(self) -> TokenCount | None:
        """Get the number of tokens produced/expected by this spec.

        Returns
        -------
        Optional[TokenCount]
            The number of tokens, or None if not applicable.
        """
        return None

    def adds_cls_token(self) -> bool:
        """Check whether this spec adds a CLS token.

        Returns
        -------
        bool
            True if this spec adds a CLS token, False otherwise.
        """
        return False

    def adds_mask_token(self) -> bool:
        """Check whether this spec adds a mask token.

        Returns
        -------
        bool
            True if this spec adds a mask token, False otherwise.
        """
        return False

    def validate(
        self,
        upstream_embedding_dim: EmbeddingDim | None = None,
        upstream_token_count: TokenCount | None = None,
    ) -> None:
        """Validate this spec against upstream specifications.

        Parameters
        ----------
        upstream_embedding_dim : Optional[EmbeddingDim], default=None
            Embedding dimension from upstream specs, if any.
        upstream_token_count : Optional[TokenCount], default=None
            Token count from upstream specs, if any.

        Raises
        ------
        ValueError
            If validation fails.
        """
        pass


@dataclass(frozen=True)
class PatchSpec(Spec):
    """Specification for patch embedding.

    Convert images into sequences of patches.

    Parameters
    ----------
    img_size : Union[int, tuple[int, int]]
        Size of input images. If int, assumes square images.
    patch_size : Union[int, tuple[int, int]]
        Size of patches. If int, assumes square patches.
    in_chans : int
        Number of input image channels.
    embed_dim : EmbeddingDim
        Dimension of patch embeddings.
    flatten : bool, default=True
        Whether to flatten patch embeddings.
    """

    img_size: int | tuple[int, int]
    patch_size: int | tuple[int, int]
    in_chans: int
    embed_dim: EmbeddingDim
    flatten: bool = True

    def __post_init__(self) -> None:
        """Validate arguments after initialization."""
        self._validate_img_size()
        self._validate_patch_size()
        self._validate_in_chans()
        self._validate_embed_dim()

    def _validate_img_size(self) -> None:
        """Validate image size field.

        Raises
        ------
        ValueError
            If img_size is not positive or invalid tuple.
        TypeError
            If img_size is not int or tuple of two ints.
        """
        if isinstance(self.img_size, int):
            if self.img_size <= 0:
                raise ValueError("img_size must be positive")
        elif isinstance(self.img_size, tuple):
            if len(self.img_size) != 2 or any(
                dim <= 0 for dim in self.img_size
            ):
                raise ValueError(
                    "img_size tuple must contain two positive integers"
                )
        else:
            raise TypeError("img_size must be an int or a tuple of two ints")

    def _validate_patch_size(self) -> None:
        """Validate patch size field.

        Raises
        ------
        ValueError
            If patch_size is not positive or invalid tuple.
        TypeError
            If patch_size is not int or tuple of two ints.
        """
        if isinstance(self.patch_size, int):
            if self.patch_size <= 0:
                raise ValueError("patch_size must be positive")
        elif isinstance(self.patch_size, tuple):
            if len(self.patch_size) != 2 or any(
                dim <= 0 for dim in self.patch_size
            ):
                raise ValueError(
                    "patch_size tuple must contain two positive integers"
                )
        else:
            raise TypeError("patch_size must be an int or a tuple of two ints")

    def _validate_in_chans(self) -> None:
        """Validate input channels field.

        Raises
        ------
        ValueError
            If in_chans is not a positive integer.
        """
        if not isinstance(self.in_chans, int) or self.in_chans <= 0:
            raise ValueError("in_chans must be a positive integer")

    def _validate_embed_dim(self) -> None:
        """Validate embedding dimension field.

        Raises
        ------
        ValueError
            If embed_dim is not a positive integer.
        """
        if not isinstance(self.embed_dim, int) or self.embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer")

    def get_embedding_dim(self) -> EmbeddingDim:
        """Get embedding dimension produced by patch embedding.

        Returns
        -------
        EmbeddingDim
            The embedding dimension for patch embeddings.
        """
        return self.embed_dim

    def get_token_count(self) -> TokenCount:
        """Calculate number of tokens produced by patch embedding.

        Returns
        -------
        TokenCount
            Number of patch tokens that will be produced.
        """
        # Handle both int and tuple cases
        if isinstance(self.img_size, tuple):
            h, w = self.img_size
        else:
            h = w = self.img_size

        if isinstance(self.patch_size, tuple):
            ph, pw = self.patch_size
        else:
            ph = pw = self.patch_size

        return (h // ph) * (w // pw)


@dataclass(frozen=True)
class PosEncSpec(Spec):
    """Specification for positional encoding.

    Add positional information to token embeddings.

    Parameters
    ----------
    kind : Literal["learned", "sincos"]
        Type of positional encoding to use.
    include_cls : bool, default=True
        Whether to include positional encoding for CLS token.
    """

    kind: Literal["learned", "sincos"]
    include_cls: bool = True

    def __post_init__(self) -> None:
        """Validate arguments after initialization.

        Raises
        ------
        TypeError
            If kind is not 'learned' or 'sincos'.
        """
        if self.kind not in ["learned", "sincos"]:
            # Using TypeError to match test expectations
            raise TypeError("kind must be either 'learned' or 'sincos'")

    def validate(
        self,
        upstream_embedding_dim: EmbeddingDim | None = None,
        upstream_token_count: TokenCount | None = None,
    ) -> None:
        """Validate against upstream specs.

        Parameters
        ----------
        upstream_embedding_dim : Optional[EmbeddingDim], default=None
            Embedding dimension from upstream specs.
        upstream_token_count : Optional[TokenCount], default=None
            Token count from upstream specs.

        Raises
        ------
        ValueError
            If the upstream embedding dimension or token count is not set.
        """
        if upstream_embedding_dim is None:
            raise ValueError(
                "PosEncSpec requires an upstream component that defines embed_dim"
            )

        if upstream_token_count is None:
            raise ValueError(
                "PosEncSpec requires an upstream component that defines token count"
            )


@dataclass(frozen=True)
class CLSTokenSpec(Spec):
    """Specification for CLS token.

    Add a learnable classification token to the beginning of the sequence.
    """

    def adds_cls_token(self) -> bool:
        """Check whether this spec adds a CLS token.

        Returns
        -------
        bool
            Always True, as this spec adds a CLS token.
        """
        return True

    def validate(
        self,
        upstream_embedding_dim: EmbeddingDim | None = None,
        upstream_token_count: TokenCount | None = None,
    ) -> None:
        """Validate against upstream specs.

        Parameters
        ----------
        upstream_embedding_dim : Optional[EmbeddingDim], default=None
            Embedding dimension from upstream specs.
        upstream_token_count : Optional[TokenCount], default=None
            Token count from upstream specs.

        Raises
        ------
        ValueError
            If the upstream embedding dimension is not set.
        """
        if upstream_embedding_dim is None:
            raise ValueError(
                "CLSTokenSpec requires an upstream component that defines embed_dim"
            )


@dataclass(frozen=True)
class MaskTokenSpec(Spec):
    """Specification for mask token.

    Define a learnable token used to replace masked patches.
    """

    def adds_mask_token(self) -> bool:
        """Check whether this spec adds a mask token.

        Returns
        -------
        bool
            Always True, as this spec adds a mask token.
        """
        return True

    def validate(
        self,
        upstream_embedding_dim: EmbeddingDim | None = None,
        upstream_token_count: TokenCount | None = None,
    ) -> None:
        """Validate against upstream specs.

        Parameters
        ----------
        upstream_embedding_dim : Optional[EmbeddingDim], default=None
            Embedding dimension from upstream specs.
        upstream_token_count : Optional[TokenCount], default=None
            Token count from upstream specs.

        Raises
        ------
        ValueError
            If the upstream embedding dimension is not set.
        """
        if upstream_embedding_dim is None:
            raise ValueError(
                "MaskTokenSpec requires an upstream component that defines embed_dim"
            )


@dataclass(frozen=True)
class ETBlockSpec(Spec):
    """Specification for an Energy Transformer block.

    Define a complete energy-based attention block with layer norm,
    multi-head attention, and Hopfield network.

    Parameters
    ----------
    steps : int, default=12
        Number of energy function optimization steps.
    alpha : float, default=0.125
        Step size for energy optimization.
    layer_norm_eps : float, default=1e-5
        Epsilon value for layer normalization.
    num_heads : int, default=12
        Number of attention heads.
    head_dim : int, default=64
        Dimension of each attention head.
    hidden_dim : int, default=2048
        Hidden dimension for Hopfield network.
    hopfield_type : Literal["standard", "simplicial"], default="standard"
        Type of Hopfield network to use.
    attention_beta : Optional[float], default=None
        Attention temperature parameter. If None, computed as 1/sqrt(head_dim).
    """

    steps: int = 12
    alpha: float = 0.125
    # Optional sub-components
    layer_norm_eps: float = 1e-5
    num_heads: int = 12
    head_dim: int = 64
    hidden_dim: int = 2048
    hopfield_type: Literal["standard", "simplicial"] = "standard"
    attention_beta: float | None = None  # Attention temperature beta

    def __post_init__(self) -> None:
        """Validate arguments after initialization.

        Raises
        ------
        ValueError
            If any parameter values are invalid.
        """
        if not isinstance(self.steps, int) or self.steps <= 0:
            raise ValueError("steps must be a positive integer")

        if not isinstance(self.alpha, int | float) or self.alpha <= 0:
            raise ValueError("alpha must be a positive number")

        if (
            not isinstance(self.layer_norm_eps, int | float)
            or self.layer_norm_eps <= 0
        ):
            raise ValueError("layer_norm_eps must be a positive number")

        if not isinstance(self.num_heads, int) or self.num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")

        if not isinstance(self.head_dim, int) or self.head_dim <= 0:
            raise ValueError("head_dim must be a positive integer")

        if not isinstance(self.hidden_dim, int) or self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")

        if self.hopfield_type not in ["standard", "simplicial"]:
            raise ValueError(
                "hopfield_type must be either 'standard' or 'simplicial'"
            )

        # Validate attention_beta
        if self.attention_beta is not None:
            if (
                not isinstance(self.attention_beta, int | float)
                or self.attention_beta <= 0
            ):
                raise ValueError("attention_beta must be a positive number")

    def validate(
        self,
        upstream_embedding_dim: EmbeddingDim | None = None,
        upstream_token_count: TokenCount | None = None,
    ) -> None:
        """Validate against upstream specs.

        Parameters
        ----------
        upstream_embedding_dim : Optional[EmbeddingDim], default=None
            Embedding dimension from upstream specs.
        upstream_token_count : Optional[TokenCount], default=None
            Token count from upstream specs.

        Raises
        ------
        ValueError
            If the upstream embedding dimension is not set.
        """
        if upstream_embedding_dim is None:
            raise ValueError(
                "ETBlockSpec requires an upstream component that defines embed_dim"
            )


@dataclass(frozen=True)
class NormSpec(Spec):
    """Specification for normalization layer.

    Parameters
    ----------
    eps : float, default=1e-5
        Small constant added to denominator for numerical stability.
    """

    eps: float = 1e-5

    def __post_init__(self) -> None:
        """Validate arguments after initialization.

        Raises
        ------
        ValueError
            If eps is not a positive number.
        """
        if not isinstance(self.eps, int | float) or self.eps <= 0:
            raise ValueError("eps must be a positive number")

    def validate(
        self,
        upstream_embedding_dim: EmbeddingDim | None = None,
        upstream_token_count: TokenCount | None = None,
    ) -> None:
        """Validate against upstream specs.

        Parameters
        ----------
        upstream_embedding_dim : Optional[EmbeddingDim], default=None
            Embedding dimension from upstream specs.
        upstream_token_count : Optional[TokenCount], default=None
            Token count from upstream specs.

        Raises
        ------
        ValueError
            If the upstream embedding dimension is not set.
        """
        if upstream_embedding_dim is None:
            raise ValueError(
                "NormSpec requires an upstream component that defines embed_dim"
            )
