"""Base classes for Energy Transformer components."""

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class BaseLayerNorm(nn.Module, ABC):  # type: ignore[misc]
    """Base class for all layer normalization implementations.

    Defines the interface required for layer normalization components
    used in BaseEnergyTransformer. Layer normalization transforms input
    tokens into a normalized representation suitable for energy computation.
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Apply layer normalization to input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [..., N, D] where N is the number
            of tokens and D is the input dimension

        Returns
        -------
        Tensor
            Normalized output tensor of shape [..., N, D]
        """
        raise NotImplementedError

    def reset_parameters(self) -> None:
        """Reset layer parameters to initial values.

        Override in subclasses to provide specific initialization.
        Default implementation does nothing.
        """
        pass


class BaseEnergyAttention(nn.Module, ABC):  # type: ignore[misc]
    """Base class for all energy-based attention implementations.

    Defines the interface required for attention components used
    in BaseEnergyTransformer. Energy attention computes a scalar
    energy value from normalized token representations.
    """

    @abstractmethod
    def forward(self, g: Tensor) -> Tensor:  # scalar
        """Compute attention energy from normalized tokens.

        Parameters
        ----------
        g : Tensor
            Normalized token tensor of shape [..., N, D] where N is
            the number of tokens and D is the feature dimension

        Returns
        -------
        Tensor
            Scalar energy value (shape == ()) representing the attention energy
        """
        raise NotImplementedError

    def reset_parameters(self) -> None:
        """Reset layer parameters to initial values.

        Override in subclasses to provide specific initialization.
        Default implementation does nothing.
        """
        pass


class BaseHopfieldNetwork(nn.Module, ABC):  # type: ignore[misc]
    """Base class for all Hopfield Network implementations.

    All Hopfield Networks must implement the forward method
    that computes energy given normalized token representations.
    """

    @abstractmethod
    def forward(self, g: Tensor) -> Tensor:  # scalar
        """Compute Hopfield Network energy.

        Parameters
        ----------
        g : Tensor
            Normalized input tensor of shape [..., N, D] where N is the number
            of tokens and D is input dimension of each token

        Returns
        -------
        Tensor
            Scalar energy value (shape == ())
        """
        raise NotImplementedError

    def reset_parameters(self) -> None:
        """Reset network parameters to initial values.

        Override in subclasses to provide specific initialization.
        Default implementation does nothing.
        """
        pass


def _validate_scalar_energy(energy: Tensor, component_name: str) -> None:
    """Validate that energy tensor is a scalar.

    Parameters
    ----------
    energy : Tensor
        Energy tensor to validate
    component_name : str
        Name of component for error messages

    Raises
    ------
    ValueError
        If energy tensor is not a scalar
    """
    if energy.ndim != 0:
        # Build error message lazily only on failure
        raise ValueError(
            f"{component_name} must return scalar energy tensor, "
            f"got shape {tuple(energy.shape)}"
        )
