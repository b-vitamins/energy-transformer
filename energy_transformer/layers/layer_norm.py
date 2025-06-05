r"""Energy-based LayerNorm implementation following Energy Transformer theory."""

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from .constants import (
    DEFAULT_EPSILON,
    DEFAULT_LAYER_NORM_REGULARIZATION,
)


class EnergyLayerNorm(nn.Module):
    r"""Energy-based Layer Normalization.

    Implements layer normalization as described in Energy Transformer theory,
    where the operation emerges as the gradient of an energy function.

    Parameters
    ----------
    normalized_shape : int or tuple of ints
        Input shape from an expected input of size
        ``[* x normalized_shape[0] x normalized_shape[1] x ... x normalized_shape[-1]]``.
        If a single integer is used, it is treated as a singleton list.
    eps : float, default=1e-5
        Small constant for numerical stability in the denominator.
    regularization : float, default=0.0
        Regularization coefficient \u03bb that adds \u03bbx to preserve input information.
    enforce_positive_gamma : bool, default=True
        If True, uses log-parameterization with softplus to ensure \u03b3 > 0.
        This guarantees the energy L is bounded below, essential for
        probability distributions proportional to e^(-L) to be normalizable.
    device : torch.device, optional
        Device for parameters.
    dtype : torch.dtype, optional
        Data type for parameters.

    Attributes
    ----------
    gamma : nn.Parameter
        Scalar scaling parameter \u03b3. If enforce_positive_gamma=True,
        this is actually log(\u03b3) and \u03b3 is computed as softplus(log_gamma).
    delta : nn.Parameter
        Vector bias parameter \u03b4 \u2208 \u211d\u1d05.

    Notes
    -----
    Mathematical Foundation:
    The Energy LayerNorm operation emerges as the gradient of a Lagrangian function,
    providing a principled activation function for energy-based models.

    Given input x \in \mathbb{R}^D, the operation computes:

    .. math::
        g_i = \gamma \frac{x_i - \bar{x}}{\sqrt{\frac{1}{D}\sum_j(x_j - \bar{x})^2 + \varepsilon}} + \delta_i

    where:
    - \gamma \in \mathbb{R} is a learnable scalar scaling parameter
    - \delta \in \mathbb{R}^D is a learnable bias vector
    - \varepsilon is a small constant for numerical stability
    - :math:`\bar{x} = \frac{1}{D}\sum_{k=1}^D x_k` is the mean

    This operation is the gradient of the Lagrangian:

    .. math::
        L(x) = D \cdot \gamma \cdot \sqrt{\frac{1}{D}\sum_j(x_j - \bar{x})^2 + \varepsilon} + \sum_j \delta_j x_j

    such that :math:`g_i = \frac{\partial L}{\partial x_i}`. This property ensures that the
    Energy Transformer's dynamics minimize a well-defined energy function.

    Energy Decrease Guarantee:
    The Hessian :math:`M_{ij} = \frac{\partial^2 L}{\partial x_i \partial x_j}` is positive
    semi-definite, which guarantees that the energy decreases during the forward dynamics
    of the Energy Transformer. This is crucial for the convergence proof in equation (7)
    of the paper.

    The regularization term \u03bbx can be added to help preserve input information during
    the iterative refinement process, though it's not part of the original formulation.

    Examples
    --------
    >>> # Standard energy layer norm
    >>> layer = EnergyLayerNorm(768)
    >>> x = torch.randn(32, 100, 768)
    >>> output = layer(x)  # (32, 100, 768)

    >>> # With regularization
    >>> layer = EnergyLayerNorm(768, regularization=0.001)

    >>> # Compute the energy Lagrangian
    >>> energy = layer.compute_energy(x)  # (32, 100)

    References
    ----------
    .. [1] Hoover, B., Liang, Y., Pham, B., Panda, R., Strobelt, H., Chau, D. H.,
       Zaki, M. J., & Krotov, D. (2023). Energy Transformer.
       arXiv preprint arXiv:2302.07253. See equations (1) and (2).
    """

    def __init__(
        self,
        normalized_shape: int | Sequence[int],
        eps: float = DEFAULT_EPSILON,
        regularization: float = DEFAULT_LAYER_NORM_REGULARIZATION,
        enforce_positive_gamma: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.regularization = regularization
        self.enforce_positive_gamma = enforce_positive_gamma

        self.D = 1
        for dim in self.normalized_shape:
            self.D *= dim

        if self.enforce_positive_gamma:
            init_val = torch.log(torch.expm1(torch.tensor(1.0))).item()
            self.log_gamma = nn.Parameter(
                torch.full((), init_val, **factory_kwargs)
            )
        else:
            self.gamma = nn.Parameter(torch.ones((), **factory_kwargs))

        self.delta = nn.Parameter(
            torch.zeros(normalized_shape, **factory_kwargs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply energy-based layer normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[*, normalized_shape[0], ..., normalized_shape[-1]]``
            where * means any number of additional dimensions.

        Returns
        -------
        torch.Tensor
            Normalized tensor of same shape as input.
        """
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]

        if self.enforce_positive_gamma:
            gamma = F.softplus(self.log_gamma)
        else:
            gamma = self.gamma

        x_mean = x.mean(dim=dims, keepdim=True)  # shape: [..., 1, ..., 1]
        x_centered = x - x_mean  # shape: [..., D]
        var = x_centered.pow(2).mean(
            dim=dims, keepdim=True
        )  # shape: [..., 1, ..., 1]

        g = (
            gamma * x_centered / torch.sqrt(var + self.eps) + self.delta
        )  # shape: [..., D]

        if self.regularization != 0:
            g = g + self.regularization * x

        return g

    def compute_energy(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute the energy Lagrangian L.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[*, normalized_shape[0], ..., normalized_shape[-1]]``.

        Returns
        -------
        torch.Tensor
            Energy values of shape ``[*]`` (one per sample).

        Notes
        -----
        Computes:

        .. math::
            L = D \\cdot \\gamma \\cdot \\sqrt{\frac{1}{D}\\sum_j(x_j - \bar{x})^2 + \varepsilon} + \\sum_j \\delta_j x_j
        """
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]

        if self.enforce_positive_gamma:
            gamma = F.softplus(self.log_gamma)
        else:
            gamma = self.gamma

        x_mean = x.mean(dim=dims, keepdim=True)  # shape: [..., 1, ..., 1]
        var = (x - x_mean).pow(2).mean(dim=dims, keepdim=False)  # shape: [...]

        energy_norm = (
            self.D * gamma * torch.sqrt(var + self.eps)
        )  # shape: [...]
        energy_bias = (self.delta * x).sum(dim=dims)  # shape: [...]

        return energy_norm + energy_bias

    def extra_repr(self) -> str:
        """Extra representation string for printing."""
        s = f"{self.normalized_shape}, eps={self.eps}"
        if self.regularization != 0:
            s += f", regularization={self.regularization}"
        if not self.enforce_positive_gamma:
            s += ", enforce_positive_gamma=False"
        return s
