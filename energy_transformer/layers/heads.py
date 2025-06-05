"""Classification head implementations for Energy Transformer models."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from torch import Tensor, nn

__all__ = [
    "ClassifierHead",
    "LinearClassifierHead",
    "NormLinearClassifierHead",
    "NormMLPClassifierHead",
    "ReLUMLPClassifierHead",
]


def _create_pool(pool_type: str = "avg") -> nn.Module:
    """Create pooling layer for sequence inputs.

    Parameters
    ----------
    pool_type : str
        Type of pooling: 'avg', 'max', 'token', or 'none'.

    Returns
    -------
    nn.Module
        Pooling module.
    """
    if pool_type == "avg":
        return nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(1))
    if pool_type == "max":
        return nn.Sequential(nn.AdaptiveMaxPool1d(1), nn.Flatten(1))
    if pool_type == "token":
        # Use first token (CLS token)
        return nn.Identity()
    if pool_type == "none":
        return nn.Identity()
    raise ValueError(f"Unknown pool_type: {pool_type}")


class ClassifierHead(nn.Module):  # type: ignore[misc]
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
        pool_type: str = "token",
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
        self.pool = _create_pool(pool_type)

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
            nn.init.normal_(self.fc.weight, std=0.01)
            if self.fc.bias is not None:
                nn.init.zeros_(self.fc.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape:
            - (B, N, C) for sequence input with pool_type in ['avg', 'max', 'token']
            - (B, C) for pre-pooled input with pool_type='none'
            - (B, C, N) for use_conv=True

        Returns
        -------
        Tensor
            Logits of shape (B, num_classes).
        """
        if self.pool_type == "token":
            # Extract CLS token
            x = x[:, 0]  # (B, C)
        elif self.pool_type in ["avg", "max"]:
            # Pool sequence dimension
            x = x.transpose(1, 2)  # (B, C, N)
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


class LinearClassifierHead(nn.Module):  # type: ignore[misc]
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
        pool_type: str = "token",
        drop_rate: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.pool_type = pool_type
        self.pool = _create_pool(pool_type)
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(in_features, num_classes, bias=bias)

        # Zero init
        nn.init.zeros_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

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
        if x.ndim == 3:  # noqa: PLR2004
            if self.pool_type == "token":
                x = x[:, 0]
            elif self.pool_type in ["avg", "max"]:
                x = x.transpose(1, 2)
                x = self.pool(x)

        x = self.drop(x)
        return cast(Tensor, self.fc(x))


class NormMLPClassifierHead(nn.Module):  # type: ignore[misc]
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
        pool_type: str = "token",
        drop_rate: float = 0.0,
        act_layer: Callable[..., nn.Module] | None = nn.GELU,
        norm_layer: Callable[..., nn.Module] | None = nn.LayerNorm,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features

        self.pool_type = pool_type
        self.pool = _create_pool(pool_type)
        self.norm = norm_layer(in_features) if norm_layer else nn.Identity()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer() if act_layer else nn.Identity()
        self.drop = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        # Hidden layer with normal init
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
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
                x = x.transpose(1, 2)
                x = self.pool(x)

        # MLP with norm
        x = self.norm(x)  # (B, C)
        x = self.fc1(x)  # (B, hidden)
        x = self.act(x)  # (B, hidden)
        x = self.drop(x)  # (B, hidden)
        return cast(Tensor, self.fc2(x))  # (B, num_classes)


class NormLinearClassifierHead(nn.Module):  # type: ignore[misc]
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
        pool_type: str = "token",
        drop_rate: float = 0.0,
        norm_layer: Callable[..., nn.Module] | None = nn.LayerNorm,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.pool_type = pool_type
        self.pool = _create_pool(pool_type)
        self.norm = norm_layer(in_features) if norm_layer else nn.Identity()
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(in_features, num_classes, bias=bias)

        # Zero init for output
        nn.init.zeros_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

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
                x = x.transpose(1, 2)
                x = self.pool(x)

        x = self.norm(x)
        x = self.drop(x)
        return cast(Tensor, self.fc(x))


class ReLUMLPClassifierHead(nn.Module):  # type: ignore[misc]
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
        pool_type: str = "token",
        drop_rate: float = 0.0,
        norm_layer: Callable[..., nn.Module] | None = nn.LayerNorm,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or num_classes

        self.pool_type = pool_type
        self.pool = _create_pool(pool_type)
        self.norm = norm_layer(in_features) if norm_layer else nn.Identity()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        # Hidden layer with normal init
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
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
                x = x.transpose(1, 2)
                x = self.pool(x)

        # MLP with norm
        x = self.norm(x)  # (B, C)
        x = self.fc1(x)  # (B, hidden)
        x = self.act(x)  # (B, hidden)
        x = self.drop(x)  # (B, hidden)
        return cast(Tensor, self.fc2(x))  # (B, num_classes)
