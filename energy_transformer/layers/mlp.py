"""MLP module for transformer blocks."""

from torch import Tensor, nn


class MLP(nn.Module):
    """Feedforward network with configurable activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        activation: str = "gelu",
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "swish": nn.SiLU(),
            "silu": nn.SiLU(),
        }
        self.act = activations.get(activation, nn.GELU())

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x  # noqa: RET504
