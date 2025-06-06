"""Energy-based Hopfield Network module implementation."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from .constants import (
    DEFAULT_HOPFIELD_BETA,
    DEFAULT_HOPFIELD_MULTIPLIER,
    DEFAULT_INIT_STD,
)
from .types import ActivationType, Device, Dtype, EmbedDim, HiddenDim
from .validation import validate_positive, validate_tensor_dim


class HopfieldNetwork(nn.Module):
    r"""Hopfield Network for associative memory.

    Mathematical Foundation
    -----------------------
    The Hopfield Network ensures token representations are consistent with
    learned memory patterns. The energy function is:

    .. math::
        E^{HN} = -\sum_{B=1}^{N} \sum_{\mu=1}^{K} G\left(\sum_{j=1}^{D} \xi_{\mu j} g_{jB}\right)

    where:
    - \(\xi_{\mu j}\) are learnable memory patterns
    - \(G(\cdot)\) is the integral of the activation function ``r`` so that
      ``G'(\cdot) = r(\cdot)``

    For different activation functions:
    - Classical (ReLU): ``r(x) = max(0, x)`` with slowly growing energy
    - Modern (softmax): ``r(x) = softmax(x)`` with sharply peaked basins

    The gradient contribution is:

    .. math::
        -\frac{\partial E^{HN}}{\partial g_{iA}} = \sum_{\mu=1}^{K} \xi_{\mu i} r\left(\sum_{j=1}^{D} \xi_{\mu j} g_{jA}\right)

    This is applied to each token individually (no inter-token mixing).

    Parameters
    ----------
    embed_dim : int
        Input dimension D of token vectors.
    hidden_dim : int, optional
        Number of memory patterns K. If None, defaults to embed_dim * hidden_ratio.
    hidden_ratio : float, default=4.0
        Multiplier for hidden dimension when hidden_dim is not specified.
    activation : {'relu', 'softmax'}, default='relu'
        Type of activation function:
        - 'relu': Classical continuous Hopfield Network with ReLU activation
        - 'softmax': Modern continuous Hopfield Network with softmax activation
    beta : float, default=0.01
        Temperature parameter for softmax activation. Only used when activation='softmax'.
        Note: This becomes a learnable parameter for softmax activation.
    bias : bool, default=False
        Whether to include bias terms in the hidden layer.
    init_std : float, default=0.02
        Standard deviation for weight initialization.
    device : torch.device, optional
        Device for parameters.
    dtype : torch.dtype, optional
        Data type for parameters.

    Attributes
    ----------
    kernel : nn.Parameter
        Memory patterns ξ in R^{D x K} stored as transposed for efficiency.
    beta : nn.Parameter or None
        Temperature parameter (learnable for softmax, None for relu).
    bias : nn.Parameter or None
        Optional bias terms for hidden layer.

    Notes
    -----
    Mathematical Foundation:
    The Hopfield Network ensures token representations align with realistic patterns
    stored in memory. The energy function is:

    .. math::
        E^{HN} = -\sum_{B=1}^{N} \sum_{\mu=1}^{K} G\left(\sum_{j=1}^{D} \xi_{\mu j} g_{jB}\right)

    where:
    - ξ in R^{K x D} are learnable memory patterns
    - G(·) is an integral of the activation function r(·), such that G'(·) = r(·)
    - B indexes tokens (N total)
    - μ indexes memories (K total)
    - j indexes feature dimensions (D total)

    For different activation functions:

    **ReLU activation** (classical continuous Hopfield):

    .. math::
        G(z) = \frac{1}{2}[\max(0, z)]^2, \quad r(z) = \max(0, z)

    This grows slowly and allows broad basins of attraction around memories.

    **Softmax activation** (modern continuous Hopfield):

    .. math::
        G(z) = \frac{1}{\beta}\log\sum\exp(\beta z), \quad r(z) = \text{softmax}(\beta z)

    This creates sharp peaks around memories with exponential capacity.

    The gradient contribution to token updates is:

    .. math::
        -\frac{\partial E^{HN}}{\partial g_{iA}} = \sum_{\mu=1}^{K} \xi_{\mu i} r\left(\sum_{j=1}^{D} \xi_{\mu j} g_{jA}\right)

    This is applied to each token individually (no mixing between tokens).

    Relationship to MLPs:
    This module is analogous to the feed-forward MLP in conventional transformers but
    with a crucial difference: the projection weights from token space to hidden space
    must be the same (transposed) as the weights from hidden space back to token space.
    This weight sharing is essential for the energy interpretation.

    Examples
    --------
    >>> # Classical Hopfield with ReLU
    >>> hn_relu = HopfieldNetwork(768, activation='relu')
    >>> tokens = torch.randn(32, 100, 768)
    >>> energy = hn_relu(tokens)  # scalar

    >>> # Modern Hopfield with softmax and learnable beta
    >>> hn_softmax = HopfieldNetwork(768, activation='softmax', beta=0.1)
    >>> energy = hn_softmax(tokens)  # scalar

    >>> # Get gradient for dynamics
    >>> grad = hn_relu.compute_grad(tokens)  # (32, 100, 768)

    References
    ----------
    .. [1] Hoover et al. (2023). Energy Transformer. See equations (5) and (9).
    .. [2] Ramsauer et al. (2020). Hopfield Networks is All You Need.
    """

    activation: ActivationType
    beta: nn.Parameter | None

    def __init__(
        self,
        embed_dim: EmbedDim,
        hidden_dim: HiddenDim | None = None,
        hidden_ratio: float = DEFAULT_HOPFIELD_MULTIPLIER,
        activation: ActivationType = "relu",
        beta: float = DEFAULT_HOPFIELD_BETA,
        bias: bool = False,
        init_std: float = DEFAULT_INIT_STD,
        device: Device = None,
        dtype: Dtype = None,
    ) -> None:
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or int(embed_dim * hidden_ratio)
        self.activation = activation
        self.use_bias = bias
        self.init_std = init_std

        if activation not in ["relu", "softmax"]:
            raise ValueError(
                f"HopfieldNetwork: activation must be 'relu' or 'softmax'. "
                f"Got: '{activation}'."
            )

        self.kernel = nn.Parameter(
            torch.empty((embed_dim, self.hidden_dim), **factory_kwargs)  # type: ignore[arg-type]
        )  # shape: [D, K]

        if self.use_bias:
            self.bias = nn.Parameter(
                torch.zeros(self.hidden_dim, **factory_kwargs)  # type: ignore[arg-type]
            )  # shape: [K]
        else:
            self.register_parameter("bias", None)

        if activation == "softmax":
            validate_positive(beta, "HopfieldNetwork", "beta")
            self.beta = nn.Parameter(
                torch.tensor(beta, device=device, dtype=dtype)
            )
        else:
            self.register_buffer("beta", None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.kernel, std=self.init_std)

        # Keep the provided beta value when using softmax activation
        # to allow custom temperature settings during initialization.

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        """Compute Hopfield Network energy.

        Parameters
        ----------
        g : torch.Tensor
            Input tensor of shape [..., N, D] where N is the number
            of tokens and D is the feature dimension.

        Returns
        -------
        torch.Tensor
            Scalar energy value.
        """
        validate_tensor_dim(g, 3, "HopfieldNetwork", "g")

        h = torch.matmul(g, self.kernel)  # [B, N, K]

        if self.use_bias:
            h.add_(self.bias)

        if self.activation == "relu":
            a = F.relu(h, inplace=True)
            energy = -0.5 * a.pow(2).sum()
        else:
            assert self.beta is not None
            h.mul_(self.beta)
            lse = torch.logsumexp(h, dim=-1)
            energy = -(1.0 / self.beta) * lse.sum()

        return energy

    def compute_grad(self, g: torch.Tensor) -> torch.Tensor:
        r"""Compute gradient of energy with respect to input.

        Parameters
        ----------
        g : torch.Tensor
            Input tensor of shape [..., N, D].

        Returns
        -------
        torch.Tensor
            Gradient tensor of same shape as input.

        Notes
        -----
        Computes:

        .. math::
            -\frac{\partial E^{HN}}{\partial g} = \xi^T r(g \xi)

        where :math:`r(\cdot)` is the activation function.
        """
        h = torch.matmul(g, self.kernel)

        if self.use_bias:
            h.add_(self.bias)

        if self.activation == "relu":
            a = F.relu(h)
        else:
            assert self.beta is not None
            h.mul_(self.beta)
            a = F.softmax(h, dim=-1)

        return -torch.matmul(a, self.kernel.t())  # shape: [..., N, D]

    @property
    def memory_dim(self) -> int:
        """Number of memory patterns stored (K in paper notation)."""
        return self.hidden_dim

    @property
    def input_dim(self) -> int:
        """Input dimension (D in paper notation)."""
        return self.embed_dim

    @property
    def activation_type(self) -> str:
        """Type of activation function used."""
        return self.activation

    @property
    def is_classical(self) -> bool:
        """Whether this is a classical (ReLU) Hopfield network."""
        return self.activation == "relu"

    @property
    def is_modern(self) -> bool:
        """Whether this is a modern (softmax) Hopfield network."""
        return self.activation == "softmax"

    @property
    def temperature(self) -> float | None:
        """Temperature parameter for softmax (None for ReLU)."""
        if self.activation == "softmax":
            assert self.beta is not None
            return (
                self.beta.item()
                if isinstance(self.beta, nn.Parameter)
                else self.beta
            )
        return None

    @property
    def total_params(self) -> int:
        """Total number of parameters."""
        param_count = self.embed_dim * self.hidden_dim
        if self.use_bias:
            param_count += self.hidden_dim
        if self.activation == "softmax" and isinstance(self.beta, nn.Parameter):
            param_count += 1
        return param_count

    @property
    def device(self) -> torch.device:
        """Device of the module parameters."""
        return self.kernel.device

    def extra_repr(self) -> str:
        """Return string representation for module printing."""
        parts = [
            f"embed_dim={self.embed_dim}",
            f"hidden_dim={self.hidden_dim}",
            f"activation='{self.activation}'",
        ]
        if self.activation == "softmax":
            assert self.beta is not None
            parts.append(f"beta={self.beta.item():.3f}")
        if self.use_bias:
            parts.append("bias=True")
        return ", ".join(parts)


class CHNReLU(HopfieldNetwork):
    """Classical Continuous Hopfield Network with ReLU activation.

    Energy function: E = -0.5 * sum((ReLU(g @ kernel))²)

    Parameters
    ----------
    embed_dim : int
        Input dimension of tokens.
    hidden_ratio : float, default=4.0
        Multiplier for hidden dimension.
    bias : bool, default=False
        Whether to include bias terms.
    init_std : float, default=0.02
        Standard deviation for weight initialization.
    device : torch.device, optional
        Device for parameters.
    dtype : torch.dtype, optional
        Data type for parameters.

    Examples
    --------
    >>> hn = CHNReLU(768)
    >>> tokens = torch.randn(32, 100, 768)
    >>> energy = hn(tokens)
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_ratio: float = DEFAULT_HOPFIELD_MULTIPLIER,
        bias: bool = False,
        init_std: float = DEFAULT_INIT_STD,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            hidden_ratio=hidden_ratio,
            activation="relu",
            bias=bias,
            init_std=init_std,
            device=device,
            dtype=dtype,
        )


class CHNSoftmax(HopfieldNetwork):
    """Modern Continuous Hopfield Network with softmax activation.

    Energy function: E = -(1/β) * sum(logsumexp(β * g @ kernel))

    Parameters
    ----------
    embed_dim : int
        Input dimension of tokens.
    hidden_ratio : float, default=4.0
        Multiplier for hidden dimension.
    beta : float, default=0.01
        Temperature parameter.
    bias : bool, default=False
        Whether to include bias terms.
    init_std : float, default=0.02
        Standard deviation for weight initialization.
    device : torch.device, optional
        Device for parameters.
    dtype : torch.dtype, optional
        Data type for parameters.

    Examples
    --------
    >>> hn = CHNSoftmax(768, beta=0.1)
    >>> tokens = torch.randn(32, 100, 768)
    >>> energy = hn(tokens)
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_ratio: float = DEFAULT_HOPFIELD_MULTIPLIER,
        beta: float = DEFAULT_HOPFIELD_BETA,
        bias: bool = False,
        init_std: float = DEFAULT_INIT_STD,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            hidden_ratio=hidden_ratio,
            activation="softmax",
            beta=beta,
            bias=bias,
            init_std=init_std,
            device=device,
            dtype=dtype,
        )
