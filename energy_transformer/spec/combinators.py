"""Combinators for composing Energy Transformer model specifications."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .primitives import EmbeddingDim, Spec, TokenCount

__all__ = [
    "SequentialSpec",
    "seq",
    "repeat",
    "ParallelSpec",
    "parallel",
    # Aliases
    "Seq",
    "Repeat",
    "Parallel",
]


@dataclass(frozen=True)
class SequentialSpec(Spec):
    """Sequential composition of model specifications.

    The primary combinator for arranging specs in a pipeline.
    It propagates dimensions and token counts through the sequence.

    Parameters
    ----------
    parts : tuple[Spec, ...], default=()
        The sequence of specifications to compose.
    """

    parts: tuple[Spec, ...] = field(default_factory=tuple)

    def __rshift__(self, other: Any) -> SequentialSpec:
        """Append a new spec using the >> operator.

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
        """
        if isinstance(other, SequentialSpec):
            return SequentialSpec(self.parts + other.parts)
        elif isinstance(other, Spec):
            return SequentialSpec(self.parts + (other,))
        else:
            raise TypeError(f"Cannot append {type(other)} to SequentialSpec")

    def __add__(self, other: Any) -> SequentialSpec:
        """Concatenate specs using the + operator.

        Parameters
        ----------
        other : Any
            The specification to concatenate.

        Returns
        -------
        SequentialSpec
            New sequential spec with the concatenated specification.
        """
        return self.__rshift__(other)

    def __len__(self) -> int:
        """Return the number of parts in the sequence.

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
        idx : Union[int, slice]
            Index or slice to retrieve.

        Returns
        -------
        Union[Spec, SequentialSpec]
            The requested part or a new SequentialSpec with the sliced parts.
        """
        if isinstance(idx, slice):
            return SequentialSpec(self.parts[idx])
        return self.parts[idx]

    def __iter__(self) -> Iterator[Spec]:
        """Iterate over parts in the sequence.

        Returns
        -------
        Iterator[Spec]
            Iterator over the parts in the sequence.
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
        """
        return item in self.parts

    def get_embedding_dim(self) -> EmbeddingDim | None:
        """Get the final embedding dimension of this sequence.

        Returns
        -------
        Optional[EmbeddingDim]
            The final embedding dimension, or None if not determinable.
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
        """Get the final token count of this sequence.

        Returns
        -------
        Optional[TokenCount]
            The final token count, or None if not determinable.
        """
        if not self.parts:
            return None

        # Start with the base token count from the first component
        token_count: TokenCount | None = None
        for part in self.parts:
            # Get token count from this part
            part_count = part.get_token_count()
            if part_count is not None:
                token_count = part_count

            # Check if this part adds tokens
            if part.adds_cls_token():
                token_count = token_count + 1 if token_count else None

        return token_count

    def adds_cls_token(self) -> bool:
        """Check whether this sequence adds a CLS token.

        Returns
        -------
        bool
            True if any part in the sequence adds a CLS token.
        """
        return any(part.adds_cls_token() for part in self.parts)

    def adds_mask_token(self) -> bool:
        """Check whether this sequence adds a mask token.

        Returns
        -------
        bool
            True if any part in the sequence adds a mask token.
        """
        return any(part.adds_mask_token() for part in self.parts)

    def validate(
        self,
        upstream_embedding_dim: EmbeddingDim | None = None,
        upstream_token_count: TokenCount | None = None,
    ) -> None:
        """Validate this sequence and all its parts.

        Parameters
        ----------
        upstream_embedding_dim : Optional[EmbeddingDim], default=None
            Embedding dimension from upstream specs.
        upstream_token_count : Optional[TokenCount], default=None
            Token count from upstream specs.

        Raises
        ------
        ValueError
            If any part in the sequence fails validation.
        """
        current_dim = upstream_embedding_dim
        current_count = upstream_token_count

        # Validate each part with the propagating dimensions
        for part in self.parts:
            part.validate(current_dim, current_count)

            # Update dimensions for next part
            part_dim = part.get_embedding_dim()
            if part_dim is not None:
                current_dim = part_dim

            # Update token count for next part
            part_count = part.get_token_count()
            if part_count is not None:
                current_count = part_count

            # Account for parts that add tokens
            if part.adds_cls_token() and current_count is not None:
                current_count += 1


def seq(*parts: Any) -> SequentialSpec:
    """Combine multiple specs into a sequential specification.

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
        If any part cannot be added to SequentialSpec.

    Examples
    --------
    >>> seq(PatchSpec(...), PosEncSpec(...), CLSTokenSpec(), ETBlockSpec())

    Or using the >> operator:

    >>> PatchSpec(...) >> PosEncSpec(...) >> CLSTokenSpec() >> ETBlockSpec()
    """
    result = SequentialSpec()
    for p in parts:
        if isinstance(p, SequentialSpec):
            result = SequentialSpec(result.parts + p.parts)
        elif isinstance(p, Spec):
            result = result >> p
        else:
            raise TypeError(f"Cannot add {type(p)} to SequentialSpec")
    return result


def repeat(part: Any, times: int) -> SequentialSpec:
    """Repeat a spec multiple times.

    Parameters
    ----------
    part : Any
        The specification to repeat.
    times : int
        Number of times to repeat the specification.

    Returns
    -------
    SequentialSpec
        A sequential specification with the part repeated.

    Raises
    ------
    TypeError
        If part cannot be repeated.

    Examples
    --------
    >>> # 12 ET blocks with default settings
    >>> repeat(ETBlockSpec(), 12)

    >>> # Can be combined with seq:
    >>> seq(PatchSpec(...), PosEncSpec(...), repeat(ETBlockSpec(), 12))
    """
    if times <= 0:
        return SequentialSpec()

    if isinstance(part, SequentialSpec):
        repeats: list[Spec] = []
        for _ in range(times):
            repeats.extend(part.parts)
        return SequentialSpec(tuple(repeats))
    elif isinstance(part, Spec):
        return SequentialSpec((part,) * times)
    else:
        raise TypeError(f"Cannot repeat {type(part)}")


@dataclass(frozen=True)
class ParallelSpec(Spec):
    """Parallel composition of model specifications.

    Represent branching architectures where multiple paths process
    the same input and their outputs are combined in some way.

    Parameters
    ----------
    branches : tuple[Spec, ...], default=()
        The parallel branches to compose.
    join_mode : str, default="concat"
        How to combine branch outputs ("concat", "add", or "multiply").
    """

    branches: tuple[Spec, ...] = field(default_factory=tuple)
    join_mode: str = "concat"

    def __post_init__(self) -> None:
        """Validate join_mode after initialization.

        Raises
        ------
        ValueError
            If join_mode is not supported.
        """
        join_mode = self.join_mode.lower()
        if join_mode not in ["concat", "add", "multiply"]:
            raise ValueError(f"Unsupported join_mode: {join_mode}")
        if join_mode != self.join_mode:
            object.__setattr__(self, "join_mode", join_mode)

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
        """
        if (
            isinstance(other, ParallelSpec)
            and other.join_mode == self.join_mode
        ):
            return ParallelSpec(self.branches + other.branches, self.join_mode)
        elif isinstance(other, Spec):
            return ParallelSpec(self.branches + (other,), self.join_mode)
        else:
            raise TypeError(f"Cannot add {type(other)} to ParallelSpec")

    def __len__(self) -> int:
        """Return the number of branches.

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
        idx : Union[int, slice]
            Index or slice to retrieve.

        Returns
        -------
        Union[Spec, ParallelSpec]
            The requested branch or a new ParallelSpec with sliced branches.
        """
        if isinstance(idx, slice):
            return ParallelSpec(self.branches[idx], self.join_mode)
        return self.branches[idx]

    def __iter__(self) -> Iterator[Spec]:
        """Iterate over branches.

        Returns
        -------
        Iterator[Spec]
            Iterator over the branches.
        """
        return iter(self.branches)

    def get_embedding_dim(self) -> EmbeddingDim | None:
        """Get the embedding dimension produced by this parallel spec.

        Returns
        -------
        Optional[EmbeddingDim]
            The output embedding dimension, or None if not determinable.
        """
        if not self.branches:
            return None

        # Behavior depends on join mode
        if self.join_mode == "concat":
            # Sum dimensions when concatenating
            dims = [b.get_embedding_dim() for b in self.branches]
            if None in dims:
                return None
            # Filter out None values and sum the rest
            valid_dims = [d for d in dims if d is not None]
            return sum(valid_dims) if valid_dims else None
        else:
            # For add and multiply, all dimensions must match
            dims = [b.get_embedding_dim() for b in self.branches]
            if None in dims or len(set(dims)) != 1:
                return None
            return dims[0]

    def validate(
        self,
        upstream_embedding_dim: EmbeddingDim | None = None,
        upstream_token_count: TokenCount | None = None,
    ) -> None:
        """Validate this parallel spec and all its branches.

        Parameters
        ----------
        upstream_embedding_dim : Optional[EmbeddingDim], default=None
            Embedding dimension from upstream specs.
        upstream_token_count : Optional[TokenCount], default=None
            Token count from upstream specs.

        Raises
        ------
        ValueError
            If validation fails or branches are incompatible.
        """
        if not self.branches:
            raise ValueError("ParallelSpec must have at least one branch")

        # Validate each branch with the upstream dimensions
        for branch in self.branches:
            branch.validate(upstream_embedding_dim, upstream_token_count)

        # For add and multiply, all branches must produce same embedding dim
        if self.join_mode in ["add", "multiply"]:
            dims = [
                b.get_embedding_dim()
                for b in self.branches
                if b.get_embedding_dim() is not None
            ]
            if len(set(dims)) > 1:
                raise ValueError(
                    f"All branches must produce same embedding dimension "
                    f"when join_mode={self.join_mode}"
                )


def parallel(*branches: Any, join_mode: str = "concat") -> ParallelSpec:
    """Combine multiple specs in parallel with the specified join mode.

    Parameters
    ----------
    *branches : Any
        Variable number of specifications to combine in parallel.
    join_mode : str, default="concat"
        How to combine branch outputs ("concat", "add", or "multiply").

    Returns
    -------
    ParallelSpec
        A parallel specification combining all branches.

    Raises
    ------
    TypeError
        If any branch cannot be added to ParallelSpec.

    Examples
    --------
    >>> # Create a parallel architecture with two branches
    >>> parallel(
    ...     seq(ETBlockSpec(), ETBlockSpec()),
    ...     seq(ETBlockSpec(steps=4), ETBlockSpec(steps=4)),
    ...     join_mode="add"
    ... )
    """
    processed_branches: list[Spec] = []
    for branch in branches:
        if isinstance(branch, Spec):
            processed_branches.append(branch)
        else:
            raise TypeError(f"Cannot add {type(branch)} to ParallelSpec")

    return ParallelSpec(tuple(processed_branches), join_mode)


# Aliases for backward compatibility and user-friendly API
Seq = seq
Repeat = repeat
Parallel = parallel
