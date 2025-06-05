"""Common type aliases for Energy Transformer layers."""

# ruff: noqa: A005
from collections.abc import Callable
from typing import Literal, Protocol, TypeVar

import torch
from torch import nn

# Tensor shape aliases for documentation
BatchSize = int
SequenceLength = int
EmbedDim = int
NumHeads = int
HeadDim = int
HiddenDim = int
NumClasses = int

# Common types
Dtype = torch.dtype | None
Device = torch.device | str | None
ActivationType = Literal["relu", "softmax"]
PoolType = Literal["token", "avg", "max", "none"]

# Factory functions
ModuleFactory = Callable[..., nn.Module]
T = TypeVar("T", bound=nn.Module)


class HasDevice(Protocol):
    """Protocol for modules with device property."""

    @property
    def device(self) -> torch.device:
        """Device of the module."""
        ...


class HasDtype(Protocol):
    """Protocol for modules with dtype property."""

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the module."""
        ...
