"""Combinators for composing Energy Transformer model specifications.

This module provides composition operators for building complex model
architectures from primitive specifications. The combinators handle
dimension propagation, validation, and provide intuitive APIs for
model construction.

Design Principles:
- Compositional: Complex models built from simple primitives
- Type-safe: Comprehensive validation and error checking
- Immutable: All combinators produce new specifications
- Intuitive: Natural operators like >> and + for composition

Example
-------
>>> model = seq(
...     PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768),
...     CLSTokenSpec(),
...     repeat(ETSpec(steps=4), 12),
...     LayerNormSpec()
... )
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .primitives import EmbeddingDim, Spec, TokenCount, ValidationError

__all__ = [
    "SequentialSpec",
    "ParallelSpec",
    "seq",
    "parallel",
    "repeat",
    # Aliases
    "Seq",
    "Parallel",
    "Repeat",
]


@dataclass(frozen=True)
class SequentialSpec(Spec):
    """Sequential composition of specifications.

    Executes specifications in order, propagating dimensions and token counts
    through the pipeline. This is the primary combinator for building
    transformer architectures.

    Parameters
    ----------
    parts : tuple[Spec, ...], default=()
        The specifications to execute in sequence.

    Examples
    --------
    >>> # Basic usage
    >>> seq(PatchEmbedSpec(...), CLSTokenSpec(), ETSpec())

    >>> # Using operators
    >>> PatchEmbedSpec(...) >> CLSTokenSpec() >> ETSpec()

    >>> # Complex composition
    >>> seq(
    ...     PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768),
    ...     PosEmbedSpec(include_cls=True),
    ...     CLSTokenSpec(),
    ...     repeat(ETSpec(steps=4), 12),
    ...     LayerNormSpec()
    ... )
    """

    parts: tuple[Spec, ...] = field(default_factory=tuple)

    _spec_type: ClassVar[str] = "sequential"

    def _validate_parameters(self) -> None:
        """Validate sequential specification parameters.

        Raises
        ------
        ValidationError
            If the sequence contains invalid specifications.
        """
        # Validate that all parts are specs
        for i, part in enumerate(self.parts):
            if not isinstance(part, Spec):
                raise ValidationError(
                    f"Part {i} is not a Spec: {type(part).__name__}",
                    spec_type=self.__class__.__name__,
                    suggestion="Ensure all parts are Spec instances",
                )

    def __rshift__(self, other: Any) -> SequentialSpec:
        """Append a specification using the >> operator.

        Parameters
        ----------
        other : Any
            The specification to append.

        Returns
        -------
        SequentialSpec
            New sequential spec with the appended specification.

        Raises
        ------
        TypeError
            If other cannot be appended to SequentialSpec.

        Examples
        --------
        >>> patch_embed = PatchEmbedSpec(...)
        >>> cls_token = CLSTokenSpec()
        >>> model = patch_embed >> cls_token
        """
        if isinstance(other, SequentialSpec):
            return SequentialSpec(parts=self.parts + other.parts)
        elif isinstance(other, Spec):
            return SequentialSpec(parts=self.parts + (other,))
        else:
            raise TypeError(
                f"Cannot append {type(other).__name__} to SequentialSpec. "
                f"Expected Spec or SequentialSpec."
            )

    def __add__(self, other: Any) -> SequentialSpec:
        """Concatenate specifications using the + operator.

        Parameters
        ----------
        other : Any
            The specification to concatenate.

        Returns
        -------
        SequentialSpec
            New sequential spec with the concatenated specification.

        Examples
        --------
        >>> seq1 = seq(PatchEmbedSpec(...), CLSTokenSpec())
        >>> seq2 = seq(ETSpec(), LayerNormSpec())
        >>> combined = seq1 + seq2
        """
        return self.__rshift__(other)

    def __len__(self) -> int:
        """Get the number of parts in the sequence.

        Returns
        -------
        int
            Number of parts in the sequence.
        """
        return len(self.parts)

    def __getitem__(self, idx: int | slice) -> Spec | SequentialSpec:
        """Get a part or slice of parts from the sequence.

        Parameters
        ----------
        idx : int or slice
            Index or slice to retrieve.

        Returns
        -------
        Spec or SequentialSpec
            The requested part or a new SequentialSpec with sliced parts.

        Examples
        --------
        >>> model = seq(PatchEmbedSpec(...), CLSTokenSpec(), ETSpec())
        >>> first_part = model[0]  # PatchEmbedSpec
        >>> sub_model = model[1:]  # SequentialSpec with CLSTokenSpec & ETSpec
        """
        if isinstance(idx, slice):
            return SequentialSpec(parts=self.parts[idx])
        return self.parts[idx]

    def __iter__(self) -> Iterator[Spec]:
        """Iterate over parts in the sequence.

        Returns
        -------
        Iterator[Spec]
            Iterator over the parts in the sequence.

        Examples
        --------
        >>> model = seq(PatchEmbedSpec(...), CLSTokenSpec(), ETSpec())
        >>> for part in model:
        ...     print(part.__class__.__name__)
        """
        return iter(self.parts)

    def __contains__(self, item: Any) -> bool:
        """Check if an item is in the sequence.

        Parameters
        ----------
        item : Any
            Item to check for.

        Returns
        -------
        bool
            True if item is in the sequence, False otherwise.

        Examples
        --------
        >>> model = seq(PatchEmbedSpec(...), CLSTokenSpec())
        >>> cls_spec = CLSTokenSpec()
        >>> cls_spec in model  # False (different instance)
        """
        return item in self.parts

    def get_embedding_dim(self) -> EmbeddingDim | None:
        """Get the final embedding dimension from the sequence.

        Finds the last component that defines an embedding dimension.

        Returns
        -------
        EmbeddingDim | None
            The final embedding dimension, or None if not determinable.

        Examples
        --------
        >>> model = seq(PatchEmbedSpec(embed_dim=768), ETSpec())
        >>> model.get_embedding_dim()  # 768
        """
        if not self.parts:
            return None

        # Find the last component that defines embedding_dim
        for part in reversed(self.parts):
            dim = part.get_embedding_dim()
            if dim is not None:
                return dim

        return None

    def get_token_count(self) -> TokenCount | None:
        """Get the final token count from the sequence.

        Computes the token count by propagating through all parts,
        accounting for both base token counts and added tokens.

        Returns
        -------
        TokenCount | None
            The final token count, or None if not determinable.

        Examples
        --------
        >>> model = seq(
        ...     PatchEmbedSpec(img_size=224, patch_size=16),  # 196 tokens
        ...     CLSTokenSpec()  # +1 token
        ... )
        >>> model.get_token_count()  # 197
        """
        if not self.parts:
            return None

        token_count: TokenCount | None = None
        for part in self.parts:
            # Update base token count if this part defines one
            part_count = part.get_token_count()
            if part_count is not None:
                token_count = part_count

            # Add any tokens this part contributes
            if token_count is not None:
                token_count += part.adds_tokens()

        return token_count

    def modifies_tokens(self) -> bool:
        """Check if any part modifies tokens.

        Returns
        -------
        bool
            True if any part modifies tokens.
        """
        return any(part.modifies_tokens() for part in self.parts)

    def validate(
        self,
        upstream_embedding_dim: EmbeddingDim | None = None,
        upstream_token_count: TokenCount | None = None,
    ) -> None:
        """Validate the sequence and all its parts.

        Validates each part in order, propagating dimensions and token counts
        through the sequence to ensure compatibility.

        Parameters
        ----------
        upstream_embedding_dim : EmbeddingDim | None, optional
            Embedding dimension from upstream specs.
        upstream_token_count : TokenCount | None, optional
            Token count from upstream specs.

        Raises
        ------
        ValidationError
            If any part in the sequence fails validation or if dimensions
            are incompatible.

        Examples
        --------
        >>> model = seq(PatchEmbedSpec(...), ETSpec())
        >>> model.validate()  # Raises ValidationError if invalid
        """
        # First validate our own parameters
        self._validate_parameters()

        current_dim = upstream_embedding_dim
        current_count = upstream_token_count

        # Validate each part with propagating dimensions
        for i, part in enumerate(self.parts):
            try:
                part.validate(current_dim, current_count)
            except ValidationError as e:
                # Add context about which part failed
                raise ValidationError(
                    f"Part {i} ({part.__class__.__name__}) failed "
                    f"validation: {e}",
                    spec_type=self.__class__.__name__,
                ) from e

            # Update context for next part
            part_dim = part.get_embedding_dim()
            if part_dim is not None:
                current_dim = part_dim

            part_count = part.get_token_count()
            if part_count is not None:
                current_count = part_count

            # Account for added tokens
            if current_count is not None:
                current_count += part.adds_tokens()

    def estimate_params(
        self, context_embedding_dim: EmbeddingDim | None = None
    ) -> int:
        """Estimate total parameter count for the sequence.

        Parameters
        ----------
        context_embedding_dim : EmbeddingDim | None, optional
            Initial embedding dimension context.

        Returns
        -------
        int
            Estimated total parameter count for all parts.

        Examples
        --------
        >>> model = seq(PatchEmbedSpec(...), ETSpec())
        >>> model.estimate_params()  # Total parameters across all parts
        """
        total_params = 0
        current_dim = context_embedding_dim

        for part in self.parts:
            total_params += part.estimate_params(current_dim)
            # Update dimension context for next part
            part_dim = part.get_embedding_dim()
            if part_dim is not None:
                current_dim = part_dim

        return total_params

    def find_parts_by_type(self, spec_type: type[Spec]) -> list[Spec]:
        """Find all parts of a specific type in the sequence.

        Parameters
        ----------
        spec_type : type[Spec]
            The type of spec to find.

        Returns
        -------
        list[Spec]
            List of parts matching the specified type.

        Examples
        --------
        >>> model = seq(PatchEmbedSpec(...), ETSpec(), ETSpec())
        >>> et_blocks = model.find_parts_by_type(ETSpec)
        >>> len(et_blocks)  # 2
        """
        return [part for part in self.parts if isinstance(part, spec_type)]

    def count_parts_by_type(self, spec_type: type[Spec]) -> int:
        """Count parts of a specific type in the sequence.

        Parameters
        ----------
        spec_type : type[Spec]
            The type of spec to count.

        Returns
        -------
        int
            Number of parts of the specified type.

        Examples
        --------
        >>> model = seq(PatchEmbedSpec(...), repeat(ETSpec(), 12))
        >>> model.count_parts_by_type(ETSpec)  # 12
        """
        return len(self.find_parts_by_type(spec_type))


@dataclass(frozen=True)
class ParallelSpec(Spec):
    """Parallel composition of specifications.

    Executes specifications in parallel and combines their outputs.
    Useful for branching architectures, ensemble methods, or
    multi-scale processing.

    Parameters
    ----------
    branches : tuple[Spec, ...], default=()
        The parallel branches to execute.
    join_mode : {"concat", "add", "multiply"}, default="concat"
        How to combine branch outputs.

    Examples
    --------
    >>> # Ensemble of different ET configurations
    >>> ensemble = parallel(
    ...     ETSpec(steps=4, alpha=0.1),
    ...     ETSpec(steps=8, alpha=0.2),
    ...     join_mode="add"
    ... )

    >>> # Multi-scale processing
    >>> multi_scale = parallel(
    ...     seq(PatchEmbedSpec(patch_size=8), ETSpec()),
    ...     seq(PatchEmbedSpec(patch_size=16), ETSpec()),
    ...     join_mode="concat"
    ... )
    """

    branches: tuple[Spec, ...] = field(default_factory=tuple)
    join_mode: str = "concat"

    _spec_type: ClassVar[str] = "parallel"

    def _validate_parameters(self) -> None:
        """Validate parallel specification parameters.

        Raises
        ------
        ValidationError
            If branches are empty, join_mode is invalid, or branches
            are incompatible.
        """
        # Validate join mode
        valid_modes = {"concat", "add", "multiply"}
        if self.join_mode not in valid_modes:
            raise ValidationError(
                f"Invalid join_mode '{self.join_mode}'. "
                f"Must be one of: {', '.join(valid_modes)}",
                spec_type=self.__class__.__name__,
                suggestion="Use 'concat', 'add', or 'multiply'",
            )

        # Validate branches
        if not self.branches:
            raise ValidationError(
                "ParallelSpec must have at least one branch",
                spec_type=self.__class__.__name__,
                suggestion="Add at least one Spec to the branches",
            )

        # Validate that all branches are specs
        for i, branch in enumerate(self.branches):
            if not isinstance(branch, Spec):
                raise ValidationError(
                    f"Branch {i} is not a Spec: {type(branch).__name__}",
                    spec_type=self.__class__.__name__,
                    suggestion="Ensure all branches are Spec instances",
                )

    def __add__(self, other: Any) -> ParallelSpec:
        """Add a branch using the + operator.

        Parameters
        ----------
        other : Any
            The specification to add as a branch.

        Returns
        -------
        ParallelSpec
            New parallel spec with the added branch.

        Raises
        ------
        TypeError
            If other cannot be added to ParallelSpec.

        Examples
        --------
        >>> branch1 = ETSpec(steps=4)
        >>> branch2 = ETSpec(steps=8)
        >>> ensemble = parallel(branch1) + branch2
        """
        if (
            isinstance(other, ParallelSpec)
            and other.join_mode == self.join_mode
        ):
            return ParallelSpec(
                branches=self.branches + other.branches,
                join_mode=self.join_mode,
            )
        elif isinstance(other, Spec):
            return ParallelSpec(
                branches=self.branches + (other,), join_mode=self.join_mode
            )
        else:
            raise TypeError(
                f"Cannot add {type(other).__name__} to ParallelSpec. "
                f"Expected Spec or compatible ParallelSpec."
            )

    def __len__(self) -> int:
        """Get the number of branches.

        Returns
        -------
        int
            Number of branches in the parallel spec.
        """
        return len(self.branches)

    def __getitem__(self, idx: int | slice) -> Spec | ParallelSpec:
        """Get a branch or slice of branches.

        Parameters
        ----------
        idx : int or slice
            Index or slice to retrieve.

        Returns
        -------
        Spec or ParallelSpec
            The requested branch or a new ParallelSpec with sliced branches.

        Examples
        --------
        >>> ensemble = parallel(ETSpec(), ETSpec(), ETSpec())
        >>> first_branch = ensemble[0]
        >>> sub_ensemble = ensemble[1:]
        """
        if isinstance(idx, slice):
            return ParallelSpec(
                branches=self.branches[idx], join_mode=self.join_mode
            )
        return self.branches[idx]

    def __iter__(self) -> Iterator[Spec]:
        """Iterate over branches.

        Returns
        -------
        Iterator[Spec]
            Iterator over the branches.

        Examples
        --------
        >>> ensemble = parallel(ETSpec(), ETSpec())
        >>> for branch in ensemble:
        ...     print(branch.__class__.__name__)
        """
        return iter(self.branches)

    def get_embedding_dim(self) -> EmbeddingDim | None:
        """Get the output embedding dimension based on join mode.

        Returns
        -------
        EmbeddingDim | None
            The output embedding dimension, or None if not determinable.

        Notes
        -----
        - For "concat": Sum of all branch embedding dimensions
        - For "add"/"multiply": All branches must have same dimension

        Examples
        --------
        >>> # Concatenation case
        >>> branches = parallel(
        ...     seq(PatchEmbedSpec(embed_dim=384)),
        ...     seq(PatchEmbedSpec(embed_dim=512)),
        ...     join_mode="concat"
        ... )
        >>> branches.get_embedding_dim()  # 896 (384 + 512)
        """
        if not self.branches:
            return None

        dims = [b.get_embedding_dim() for b in self.branches]

        if self.join_mode == "concat":
            # Sum dimensions when concatenating
            if None in dims:
                return None
            valid_dims = [d for d in dims if d is not None]
            return sum(valid_dims) if valid_dims else None
        else:
            # For add and multiply, all dimensions must match
            non_none_dims = [d for d in dims if d is not None]
            if not non_none_dims:
                return None
            if len(set(non_none_dims)) != 1:
                return None  # Dimensions don't match
            return non_none_dims[0]

    def modifies_tokens(self) -> bool:
        """Check if any branch modifies tokens.

        Returns
        -------
        bool
            True if any branch modifies tokens.
        """
        return any(branch.modifies_tokens() for branch in self.branches)

    def validate(
        self,
        upstream_embedding_dim: EmbeddingDim | None = None,
        upstream_token_count: TokenCount | None = None,
    ) -> None:
        """Validate the parallel spec and all its branches.

        Parameters
        ----------
        upstream_embedding_dim : EmbeddingDim | None, optional
            Embedding dimension from upstream specs.
        upstream_token_count : TokenCount | None, optional
            Token count from upstream specs.

        Raises
        ------
        ValidationError
            If validation fails or branches are incompatible.

        Examples
        --------
        >>> ensemble = parallel(ETSpec(), ETSpec(), join_mode="add")
        >>> ensemble.validate(upstream_embedding_dim=768)
        """
        # First validate our own parameters
        self._validate_parameters()

        # Validate each branch independently
        for i, branch in enumerate(self.branches):
            try:
                branch.validate(upstream_embedding_dim, upstream_token_count)
            except ValidationError as e:
                raise ValidationError(
                    f"Branch {i} ({branch.__class__.__name__}) failed "
                    f"validation: {e}",
                    spec_type=self.__class__.__name__,
                ) from e

        # For add and multiply, verify dimensions are compatible
        if self.join_mode in ["add", "multiply"]:
            dims = [
                b.get_embedding_dim()
                for b in self.branches
                if b.get_embedding_dim() is not None
            ]
            if len(set(dims)) > 1:
                raise ValidationError(
                    f"All branches must produce same embedding dimension "
                    f"for join_mode='{self.join_mode}',"
                    f"got dimensions: {dims}",
                    spec_type=self.__class__.__name__,
                    suggestion="Use 'concat' mode or ensure all branches "
                    "have the same output dimension",
                )

    def estimate_params(
        self, context_embedding_dim: EmbeddingDim | None = None
    ) -> int:
        """Estimate total parameter count for all branches.

        Parameters
        ----------
        context_embedding_dim : EmbeddingDim | None, optional
            Embedding dimension context for all branches.

        Returns
        -------
        int
            Estimated total parameter count for all branches.

        Examples
        --------
        >>> ensemble = parallel(ETSpec(), ETSpec())
        >>> ensemble.estimate_params(context_embedding_dim=768)
        """
        return sum(
            branch.estimate_params(context_embedding_dim)
            for branch in self.branches
        )

    def find_branches_by_type(self, spec_type: type[Spec]) -> list[Spec]:
        """Find all branches of a specific type.

        Parameters
        ----------
        spec_type : type[Spec]
            The type of spec to find.

        Returns
        -------
        list[Spec]
            List of branches matching the specified type.

        Examples
        --------
        >>> ensemble = parallel(ETSpec(), LayerNormSpec(), ETSpec())
        >>> et_branches = ensemble.find_branches_by_type(ETSpec)
        >>> len(et_branches)  # 2
        """
        return [
            branch for branch in self.branches if isinstance(branch, spec_type)
        ]


def seq(*parts: Any) -> SequentialSpec:
    """Create a sequential specification from multiple parts.

    Combines specifications into a pipeline where outputs flow from
    one part to the next. This is the primary way to build transformer
    architectures.

    Parameters
    ----------
    *parts : Any
        Variable number of specifications to combine sequentially.

    Returns
    -------
    SequentialSpec
        A sequential specification combining all parts.

    Raises
    ------
    TypeError
        If any part is not a valid specification.

    Examples
    --------
    >>> # Basic vision transformer
    >>> model = seq(
    ...     PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768),
    ...     PosEmbedSpec(include_cls=True),
    ...     CLSTokenSpec(),
    ...     repeat(ETSpec(steps=4), 12),
    ...     LayerNormSpec()
    ... )

    >>> # Can also use operator chaining
    >>> model = (PatchEmbedSpec(...) >>
    ...          CLSTokenSpec() >>
    ...          ETSpec())
    """
    if not parts:
        return SequentialSpec()

    result_parts: list[Spec] = []
    for i, part in enumerate(parts):
        if isinstance(part, SequentialSpec):
            # Flatten nested sequential specs
            result_parts.extend(part.parts)
        elif isinstance(part, Spec):
            result_parts.append(part)
        else:
            raise TypeError(
                f"Part {i} is not a Spec: {type(part).__name__}. "
                f"Expected Spec instance."
            )

    return SequentialSpec(parts=tuple(result_parts))


def parallel(*branches: Any, join_mode: str = "concat") -> ParallelSpec:
    """Create a parallel specification from multiple branches.

    Combines specifications to execute in parallel, with outputs
    combined according to the specified join mode.

    Parameters
    ----------
    *branches : Any
        Variable number of specifications to combine in parallel.
    join_mode : {"concat", "add", "multiply"}, default="concat"
        How to combine branch outputs.

    Returns
    -------
    ParallelSpec
        A parallel specification combining all branches.

    Raises
    ------
    TypeError
        If any branch is not a valid specification.
    ValidationError
        If join_mode is invalid.

    Examples
    --------
    >>> # Ensemble of different configurations
    >>> ensemble = parallel(
    ...     ETSpec(steps=4, alpha=0.1),
    ...     ETSpec(steps=8, alpha=0.2),
    ...     join_mode="add"
    ... )

    >>> # Multi-head parallel processing
    >>> multi_head = parallel(
    ...     MHEASpec(num_heads=8, head_dim=64),
    ...     MHEASpec(num_heads=16, head_dim=32),
    ...     join_mode="concat"
    ... )
    """
    if not branches:
        raise ValidationError(
            "parallel() requires at least one branch",
            suggestion="Add at least one Spec to the branches",
        )

    processed_branches: list[Spec] = []
    for i, branch in enumerate(branches):
        if isinstance(branch, Spec):
            processed_branches.append(branch)
        else:
            raise TypeError(
                f"Branch {i} is not a Spec: {type(branch).__name__}. "
                f"Expected Spec instance."
            )
    branches = tuple(processed_branches)
    return ParallelSpec(branches=branches, join_mode=join_mode)


def repeat(spec: Any, times: int) -> SequentialSpec:
    """Repeat a specification multiple times in sequence.

    Creates a sequential specification where the given spec is
    repeated the specified number of times. Commonly used for
    creating stacks of transformer blocks.

    Parameters
    ----------
    spec : Any
        The specification to repeat.
    times : int
        Number of times to repeat the specification.

    Returns
    -------
    SequentialSpec
        A sequential specification with the spec repeated.

    Raises
    ------
    TypeError
        If spec is not a valid specification.
    ValueError
        If times is negative.

    Examples
    --------
    >>> # 12 transformer blocks
    >>> transformer_stack = repeat(ETSpec(steps=4), 12)

    >>> # Can be combined with other specs
    >>> model = seq(
    ...     PatchEmbedSpec(...),
    ...     repeat(ETSpec(), 6),
    ...     LayerNormSpec()
    ... )

    >>> # Repeating complex sequences
    >>> complex_block = seq(ETSpec(), LayerNormSpec())
    >>> repeated = repeat(complex_block, 3)
    """
    if times < 0:
        raise ValueError(f"times must be non-negative, got {times}")

    if times == 0:
        return SequentialSpec()

    if isinstance(spec, SequentialSpec):
        # Flatten repeated sequential specs
        repeated_parts: list[Spec] = []
        for _ in range(times):
            repeated_parts.extend(spec.parts)
        return SequentialSpec(parts=tuple(repeated_parts))
    elif isinstance(spec, Spec):
        return SequentialSpec(parts=(spec,) * times)
    else:
        raise TypeError(
            f"Cannot repeat {type(spec).__name__}. Expected Spec instance."
        )


# Aliases for more intuitive API
Seq = seq
Parallel = parallel
Repeat = repeat
