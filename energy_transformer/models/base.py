"""Base Energy Transformer model definition."""

import torch
import torch.nn as nn
from torch import Tensor

from energy_transformer.layers.base import (
    BaseEnergyAttention,
    BaseHopfieldNetwork,
    BaseLayerNorm,
)

# Registry for model classes to enable lookups from realiser
REALISER_REGISTRY: dict[str, type[nn.Module]] = {}


class EnergyTransformer(nn.Module):  # type: ignore
    """Base class for all Energy Transformer models.

    Defines the core energy computation shared by all Energy Transformer variants.
    This base class is extended by domain-specific implementations such as
    ImageEnergyTransformer, GraphEnergyTransformer, etc.

    Parameters
    ----------
    layer_norm : BaseLayerNorm
        Layer normalization component that transforms input tokens into
        normalized representation
    attention : BaseEnergyAttention
        Energy-based attention component that computes attention energy
        from normalized tokens
    hopfield : BaseHopfieldNetwork
        Hopfield network component that computes memory-based energy
        from normalized tokens
    steps : int, default=12
        Number of gradient descent steps to perform
    alpha : float, default=0.125
        Step size for gradient descent

    Notes
    -----
    The energy computed is the sum of the attention energy and the Hopfield
    network energy. This combined energy defines an energy landscape over
    token configurations that can be optimized through gradient descent.

    This base class handles only the core energy computation. Domain-specific
    preprocessing, token creation, and energy optimization should be implemented
    in derived classes.
    """

    def __init__(
        self,
        layer_norm: BaseLayerNorm,
        attention: BaseEnergyAttention,
        hopfield: BaseHopfieldNetwork,
        steps: int = 12,
        alpha: float = 0.125,
    ) -> None:
        """Initialize the Energy Transformer with its components.

        Parameters
        ----------
        layer_norm : BaseLayerNorm
            Layer normalization component
        attention : BaseEnergyAttention
            Energy-based attention component
        hopfield : BaseHopfieldNetwork
            Hopfield network component
        steps : int, default=12
            Number of gradient descent steps to perform
        alpha : float, default=0.125
            Step size for gradient descent
        """
        super().__init__()
        self.layer_norm = layer_norm
        self.attention = attention
        self.hopfield = hopfield
        self.steps = steps
        self.alpha = alpha

    def energy(self, x: Tensor) -> Tensor:
        """Compute the energy of the input token state.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [..., N, D] where N is the number
            of tokens and D is input dimension of each token

        Returns
        -------
        Tensor
            Scalar energy value representing the combined energy of the input
            configuration. Lower energy corresponds to more favorable states.

        Notes
        -----
        The computation follows three steps:
        1. Normalize input tokens using layer normalization
        2. Compute attention energy from normalized tokens
        3. Compute Hopfield energy from normalized tokens
        4. Return the sum of these energies
        """
        g = self.layer_norm(x)  # shape: [..., N, K]
        return self.attention(g) + self.hopfield(g)

    def forward(
        self,
        x: Tensor,
        detach: bool = False,
        return_energy: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Optimize input tokens by performing gradient descent on the energy landscape.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape [..., N, D] where N is the number
            of tokens and D is input dimension of each token
        detach : bool, default=False
            If True, detaches gradients after optimization, useful for:
            - Frozen-backbone downstream tasks
            - Inference-only scenarios
            - Preventing gradient flow through trajectory
        return_energy : bool, default=False
            If True, returns the final energy value alongside optimized tokens

        Returns
        -------
        Union[Tensor, Tuple[Tensor, Tensor]]
            If return_energy=False: Optimized token configuration
            If return_energy=True: Tuple of (optimized tokens, final energy)
        """
        # Clone to avoid in-place side-effects that might be shared across
        # multiple ET blocks, but do not break the graph unless requested.
        x = x.clone()
        if detach:  # isolate only if the caller asks
            x = x.detach().requires_grad_(True)

        # Inner gradient-descent loop on the energy landscape
        grad_and_energy = torch.func.grad_and_value(
            self.energy
        )  # returns (∇E, E)
        final_energy = None

        for _i in range(self.steps):
            grad, energy = grad_and_energy(x)
            x = x - self.alpha * grad

        if return_energy:
            with torch.no_grad():
                final_energy = self.energy(x)
        else:
            final_energy = None

        if detach:  # frozen-backbone inference
            x = x.detach()
            if final_energy is not None:
                final_energy = final_energy.detach()

        # Return
        return (x, final_energy) if return_energy else x


# Register the EnergyTransformer class in the registry
REALISER_REGISTRY["EnergyTransformer"] = EnergyTransformer
