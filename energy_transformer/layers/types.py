"""Common type aliases for Energy Transformer layers."""

from typing import Callable, Literal, Optional, Protocol, TypeVar, Union

import torch
from torch import Tensor, nn

# Tensor shape aliases for documentation
BatchSize = int
SequenceLength = int
EmbedDim = int
NumHeads = int
HeadDim = int
HiddenDim = int
NumClasses = int

# Common types
Device = Union[torch.device, str, None]
Dtype = Optional[torch.dtype]
ActivationType = Literal["relu", "softmax"]
PoolType = Literal["token", "avg", "max", "none"]

# Factory functions
ModuleFactory = Callable[..., nn.Module]
T = TypeVar("T", bound=nn.Module)


class HasDevice(Protocol):
    """Protocol for modules with device property."""

    @property
    def device(self) -> torch.device: ...


class HasDtype(Protocol):
    """Protocol for modules with dtype property."""

    @property
    def dtype(self) -> torch.dtype: ...
