"""Classification head implementations for Energy Transformer models."""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from .constants import HEAD_INIT_STD, SMALL_INIT_STD
from .types import ModuleFactory, PoolType

__all__ = [
    "ClassifierHead",
    "LinearClassifierHead",
    "NormLinearClassifierHead",
    "NormMLPClassifierHead",
    "ReLUMLPClassifierHead",
]


class _TokenPool(nn.Module):
    """Extract first token (CLS token)."""

    def forward(self, x: Tensor) -> Tensor:
        return x[:, 0]


class _GlobalAvgPool(nn.Module):
    """Global average pooling over sequence dimension."""

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:  # noqa: PLR2004
            return x.mean(dim=1)
        if x.dim() == 2:  # noqa: PLR2004
            return x
        raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")


class _GlobalMaxPool(nn.Module):
    """Global max pooling over sequence dimension."""

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:  # noqa: PLR2004
            return x.max(dim=1)[0]
        if x.dim() == 2:  # noqa: PLR2004
            return x
        raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")


_POOLS: dict[PoolType, nn.Module] = {
    "avg": _GlobalAvgPool(),
    "max": _GlobalMaxPool(),
    "token": _TokenPool(),
    "none": nn.Identity(),
}


class BaseClassifierHead(nn.Module):
    """Base class for classifier heads with common functionality."""

    pool_type: PoolType

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        pool_type: PoolType = "token",
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.pool_type = pool_type
        self.drop_rate = drop_rate

        self.pool = self._create_pool(pool_type)
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    @staticmethod
    def _create_pool(pool_type: PoolType) -> nn.Module:
        """Return pooling layer based on ``pool_type``."""
        if pool_type not in _POOLS:
            raise ValueError(
                f"BaseClassifierHead: Unknown pool_type '{pool_type}'. "
                "Expected one of: 'avg', 'max', 'token', 'none'."
            )
        return _POOLS[pool_type]

    def _init_linear_zero(self, layer: nn.Linear) -> None:
        """Initialize linear layer weights to zeros.

        Parameters
        ----------
        layer : nn.Linear
            Layer to initialize.
        """
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def _init_linear_normal(self, layer: nn.Linear, std: float = 0.02) -> None:
        """Initialize linear layer with truncated normal weights.

        Parameters
        ----------
        layer : nn.Linear
            Layer to initialize.
        std : float, default=0.02
            Standard deviation of the normal distribution.
        """
        nn.init.trunc_normal_(layer.weight, std=std)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def _pool_features(self, x: Tensor) -> Tensor:
        """Apply configured pooling to ``x``.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``(B, N, C)`` or ``(B, C)``.

        Returns
        -------
        Tensor
            Pooled features.
        """
        return cast(Tensor, self.pool(x))

    def extra_repr(self) -> str:
        """Return string representation for module printing."""
        return (
            f"in_features={self.in_features}, "
            f"num_classes={self.num_classes}, "
            f"pool_type='{self.pool_type}'"
        )

    @property
    def features_in(self) -> int:
        """Input feature dimension."""
        return self.in_features

    @property
    def features_out(self) -> int:
        """Output feature dimension (number of classes)."""
        return self.num_classes

    @property
    def has_dropout(self) -> bool:
        """Whether dropout is applied."""
        return self.drop_rate > 0

    @property
    def is_pooled(self) -> bool:
        """Whether input pooling is applied."""
        return self.pool_type != "none"

    @property
    def total_params(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def _create_pool(pool_type: PoolType = "avg") -> nn.Module:
    """Create pooling layer for sequence inputs."""
    return BaseClassifierHead._create_pool(pool_type)


class ClassifierHead(nn.Module):
    """General purpose classifier head with pooling and dropout.

    Handles both spatial (CNN) and sequence (ViT) inputs with configurable
    pooling strategies.

    Parameters
    ----------
    in_features : int
        Number of input features.
    num_classes : int
        Number of output classes.
    pool_type : str, default='token'
        Type of pooling to apply: 'avg', 'max', 'token', or 'none'.
        'token' uses the first token (CLS), 'none' expects pre-pooled input.
    drop_rate : float, default=0.0
        Dropout rate before classifier.
    use_conv : bool, default=False
        Use 1x1 conv instead of linear. Only valid when pool_type='none'.
    bias : bool, default=True
        Use bias in classifier layer.

    Attributes
    ----------
    pool : nn.Module
        Pooling layer.
    drop : nn.Dropout
        Dropout layer.
    fc : nn.Module
        Final classifier (Linear or Conv1d).
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        pool_type: PoolType = "token",
        drop_rate: float = 0.0,
        use_conv: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.pool_type = pool_type
        self.drop_rate = drop_rate
        self.use_conv = use_conv

        # Pooling layer
        self.pool = BaseClassifierHead._create_pool(pool_type)

        # Dropout
        self.drop = nn.Dropout(drop_rate)

        # Classifier
        self.fc: nn.Linear | nn.Conv1d
        if use_conv:
            if pool_type != "none":
                raise ValueError("use_conv=True requires pool_type='none'")
            self.fc = nn.Conv1d(in_features, num_classes, 1, bias=bias)
        else:
            self.fc = nn.Linear(in_features, num_classes, bias=bias)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        if isinstance(self.fc, nn.Linear):
            nn.init.zeros_(self.fc.weight)
            if self.fc.bias is not None:
                nn.init.zeros_(self.fc.bias)
        elif isinstance(self.fc, nn.Conv1d):
            nn.init.normal_(self.fc.weight, std=SMALL_INIT_STD)
            if self.fc.bias is not None:
                nn.init.zeros_(self.fc.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape:
            - ``(B, N, C)`` for sequence input when ``pool_type`` is
              ``'avg'``, ``'max'`` or ``'token'``
            - ``(B, C)`` for pre-pooled input with ``pool_type='none'``
            - ``(B, C, N)`` when ``use_conv=True``

        Returns
        -------
        Tensor
            Logits of shape (B, num_classes).
        """
        if self.pool_type != "none" and x.dim() not in [2, 3]:
            msg = (
                f"{self.__class__.__name__}: Input must be 2D or 3D when "
                f"pool_type='{self.pool_type}'. "
                f"Got {x.dim()}D input with shape {list(x.shape)}."
            )
            raise ValueError(msg)

        if self.pool_type == "token":
            # Extract CLS token
            x = x[:, 0]  # (B, C)
        elif self.pool_type in ["avg", "max"]:
            # Pool sequence dimension
            x = self.pool(x)  # (B, C)

        # Apply dropout
        x = self.drop(x)

        # Apply classifier
        if self.use_conv:
            if x.ndim == 2:  # noqa: PLR2004
                x = x.unsqueeze(-1)
            x = self.fc(x)
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
        else:
            x = self.fc(x)  # (B, num_classes)

        return x


class LinearClassifierHead(BaseClassifierHead):
    """Simple linear classifier head.

    Minimal classifier head that just applies pooling, dropout, and
    a linear layer. No normalization or intermediate layers.

    Parameters
    ----------
    in_features : int
        Number of input features.
    num_classes : int
        Number of output classes.
    pool_type : str, default='token'
        Type of pooling: 'avg', 'max', 'token', or 'none'.
    drop_rate : float, default=0.0
        Dropout rate before classifier.
    bias : bool, default=True
        Use bias in classifier layer.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        pool_type: PoolType = "token",
        drop_rate: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, num_classes, pool_type, drop_rate)
        self.fc = nn.Linear(in_features, num_classes, bias=bias)
        self._init_linear_zero(self.fc)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, N, C) or (B, C).

        Returns
        -------
        Tensor
            Logits of shape (B, num_classes).
        """
        x = self._pool_features(x)
        x = self.drop(x)
        return cast(Tensor, self.fc(x))


class NormMLPClassifierHead(BaseClassifierHead):
    """Norm + MLP classifier head.

    Classifier head with layer normalization and a two-layer MLP
    with configurable activation function.

    Parameters
    ----------
    in_features : int
        Number of input features.
    num_classes : int
        Number of output classes.
    hidden_features : int or None, default=None
        Hidden dimension. If None, defaults to in_features.
    pool_type : str, default='token'
        Type of pooling: 'avg', 'max', 'token', or 'none'.
    drop_rate : float, default=0.0
        Dropout rate before final classifier.
    act_layer : Callable or None, default=nn.GELU
        Activation layer constructor.
    norm_layer : Callable or None, default=nn.LayerNorm
        Normalization layer constructor.
    bias : bool, default=True
        Use bias in linear layers.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_features: int | None = None,
        pool_type: PoolType = "token",
        drop_rate: float = 0.0,
        act_layer: ModuleFactory | None = nn.GELU,
        norm_layer: ModuleFactory | None = nn.LayerNorm,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, num_classes, pool_type, drop_rate)
        hidden_features = hidden_features or in_features

        self.norm = norm_layer(in_features) if norm_layer else nn.Identity()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer() if act_layer else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)

        self._init_linear_normal(self.fc1, std=HEAD_INIT_STD)
        self._init_linear_zero(self.fc2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, N, C) or (B, C).

        Returns
        -------
        Tensor
            Logits of shape (B, num_classes).
        """
        x = self._pool_features(x)
        x = self.norm(x)  # (B, C)
        x = self.fc1(x)  # (B, hidden)
        x = self.act(x)  # (B, hidden)
        x = self.drop(x)  # (B, hidden)
        return cast(Tensor, self.fc2(x))  # (B, num_classes)


class NormLinearClassifierHead(BaseClassifierHead):
    """Normalized linear classifier head.

    Simple classifier with layer normalization followed by linear projection.

    Parameters
    ----------
    in_features : int
        Number of input features.
    num_classes : int
        Number of output classes.
    pool_type : str, default='token'
        Type of pooling: 'avg', 'max', 'token', or 'none'.
    drop_rate : float, default=0.0
        Dropout rate before classifier.
    norm_layer : Callable or None, default=nn.LayerNorm
        Normalization layer constructor.
    bias : bool, default=True
        Use bias in classifier layer.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        pool_type: PoolType = "token",
        drop_rate: float = 0.0,
        norm_layer: ModuleFactory | None = nn.LayerNorm,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, num_classes, pool_type, drop_rate)
        self.norm = norm_layer(in_features) if norm_layer else nn.Identity()
        self.fc = nn.Linear(in_features, num_classes, bias=bias)
        self._init_linear_zero(self.fc)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, N, C) or (B, C).

        Returns
        -------
        Tensor
            Logits of shape (B, num_classes).
        """
        x = self._pool_features(x)
        x = self.norm(x)
        x = self.drop(x)
        return cast(Tensor, self.fc(x))


class ReLUMLPClassifierHead(BaseClassifierHead):
    """ReLU-based MLP classifier head.

    Two-layer MLP with ReLU activation, layer normalization,
    and configurable hidden dimension.

    Parameters
    ----------
    in_features : int
        Number of input features.
    num_classes : int
        Number of output classes.
    hidden_features : int or None, default=None
        Hidden dimension. If None, defaults to num_classes.
    pool_type : str, default='token'
        Type of pooling: 'avg', 'max', 'token', or 'none'.
    drop_rate : float, default=0.0
        Dropout rate before final classifier.
    norm_layer : Callable or None, default=nn.LayerNorm
        Normalization layer constructor.
    bias : bool, default=True
        Use bias in linear layers.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_features: int | None = None,
        pool_type: PoolType = "token",
        drop_rate: float = 0.0,
        norm_layer: ModuleFactory | None = nn.LayerNorm,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, num_classes, pool_type, drop_rate)
        hidden_features = hidden_features or num_classes

        self.norm = norm_layer(in_features) if norm_layer else nn.Identity()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        # Hidden layer with normal init
        nn.init.trunc_normal_(self.fc1.weight, std=HEAD_INIT_STD)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)

        # Output layer with zero init
        nn.init.zeros_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, N, C) or (B, C).

        Returns
        -------
        Tensor
            Logits of shape (B, num_classes).
        """
        # Handle pooling for sequence inputs
        if x.ndim == 3:  # noqa: PLR2004
            if self.pool_type == "token":
                x = x[:, 0]
            elif self.pool_type in ["avg", "max"]:
                x = self.pool(x)

        # MLP with norm
        x = self.norm(x)  # (B, C)
        x = self.fc1(x)  # (B, hidden)
        x = self.act(x)  # (B, hidden)
        x = self.drop(x)  # (B, hidden)
        return cast(Tensor, self.fc2(x))  # (B, num_classes)
