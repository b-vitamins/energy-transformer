"""Energy-based Hopfield Network module implementation."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class HopfieldNetwork(nn.Module):
    r"""Energy-based Hopfield Network module.

    Implements the Hopfield Network component of Energy Transformer blocks,
    which ensures token representations are consistent with learned memory patterns.

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

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        hidden_ratio: float = 4.0,
        activation: str = "relu",
        beta: float = 0.01,
        bias: bool = False,
        init_std: float = 0.02,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
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
                f"activation must be 'relu' or 'softmax', got {activation}"
            )

        self.kernel = nn.Parameter(
            torch.empty(
                (embed_dim, self.hidden_dim), device=device, dtype=dtype
            )
        )  # shape: [D, K]

        if self.use_bias:
            self.bias = nn.Parameter(
                torch.zeros(self.hidden_dim, device=device, dtype=dtype)
            )  # shape: [K]
        else:
            self.register_parameter("bias", None)

        if activation == "softmax":
            self.beta = nn.Parameter(torch.tensor(beta, device=device, dtype=dtype))
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
        h = torch.matmul(g, self.kernel)  # shape: [..., N, K]

        if self.use_bias:
            h = h + self.bias  # shape: [..., N, K]

        if self.activation == "relu":
            a = F.relu(h)  # shape: [..., N, K]
            energy = -0.5 * (a**2).sum()
        else:
            h_scaled = self.beta * h  # shape: [..., N, K]
            lse = torch.logsumexp(h_scaled, dim=-1)  # shape: [..., N]
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
        h = torch.matmul(g, self.kernel)  # shape: [..., N, K]

        if self.use_bias:
            h = h + self.bias  # shape: [..., N, K]

        if self.activation == "relu":
            a = F.relu(h)  # shape: [..., N, K]
        else:
            h_scaled = self.beta * h  # shape: [..., N, K]
            a = F.softmax(h_scaled, dim=-1)  # shape: [..., N, K]

        return -torch.matmul(a, self.kernel.t())  # shape: [..., N, D]

    def extra_repr(self) -> str:
        """Return string representation for printing."""
        s = f"embed_dim={self.embed_dim}, hidden_dim={self.hidden_dim}"
        s += f", activation='{self.activation}'"
        if self.activation == "softmax":
            s += f", beta={self.beta.item():.3f}"
        if self.use_bias:
            s += ", bias=True"
        return s


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
        hidden_ratio: float = 4.0,
        bias: bool = False,
        init_std: float = 0.02,
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
        hidden_ratio: float = 4.0,
        beta: float = 0.01,
        bias: bool = False,
        init_std: float = 0.02,
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
