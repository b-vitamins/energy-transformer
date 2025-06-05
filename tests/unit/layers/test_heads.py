import torch
from torch import nn

from energy_transformer.layers.heads import (
    ClassifierHead,
    LinearClassifierHead,
    NormLinearClassifierHead,
    ReLUMLPClassifierHead,
)


def test_classifier_head_token_pool() -> None:
    head = ClassifierHead(in_features=4, num_classes=3, pool_type="token")
    x = torch.randn(2, 5, 4)
    out = head(x)
    assert out.shape == (2, 3)


def test_classifier_head_avg_pool() -> None:
    head = ClassifierHead(in_features=4, num_classes=3, pool_type="avg")
    x = torch.randn(2, 5, 4)
    out = head(x)
    assert out.shape == (2, 3)


def test_linear_classifier_head() -> None:
    head = LinearClassifierHead(in_features=4, num_classes=2)
    x = torch.randn(2, 5, 4)
    out = head(x)
    assert out.shape == (2, 2)


def test_norm_linear_classifier_head() -> None:
    head = NormLinearClassifierHead(in_features=4, num_classes=2, drop_rate=0.1)
    x = torch.randn(2, 5, 4)
    out = head(x)
    assert out.shape == (2, 2)
    assert isinstance(head.norm, nn.LayerNorm)


def test_relu_mlp_classifier_head() -> None:
    head = ReLUMLPClassifierHead(
        in_features=4, num_classes=2, hidden_features=8, drop_rate=0.0
    )
    x = torch.randn(2, 5, 4)
    out = head(x)
    assert out.shape == (2, 2)
