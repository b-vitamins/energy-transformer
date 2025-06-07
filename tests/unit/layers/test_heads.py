import pytest
import torch
from torch import nn

import energy_transformer.layers.heads
from energy_transformer.layers.heads import (
    ClassifierHead,
    LinearClassifierHead,
    NormLinearClassifierHead,
    NormMLPClassifierHead,
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


def test_classifier_head_conv() -> None:
    head = ClassifierHead(
        in_features=4,
        num_classes=3,
        pool_type="none",
        use_conv=True,
    )
    x = torch.randn(2, 4)
    out = head(x)
    assert out.shape == (2, 3)


def test_norm_mlp_classifier_head_avg_pool() -> None:
    head = NormMLPClassifierHead(
        in_features=4,
        num_classes=2,
        hidden_features=4,
        pool_type="avg",
    )
    x = torch.randn(2, 5, 4)
    out = head(x)
    assert out.shape == (2, 2)


def test_classifier_head_invalid_conv_pool() -> None:
    """Conv classifier requires pool_type='none'."""
    with pytest.raises(ValueError, match="use_conv=True requires"):  # PT011
        ClassifierHead(
            in_features=4, num_classes=3, pool_type="avg", use_conv=True
        )


def test_create_pool_invalid() -> None:
    """Unknown pool_type should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown pool_type"):  # PT011
        energy_transformer.layers.heads._create_pool("foobar")


def test_norm_linear_classifier_head_avg_pool() -> None:
    head = NormLinearClassifierHead(
        in_features=4, num_classes=2, pool_type="avg"
    )
    x = torch.randn(2, 5, 4)
    out = head(x)
    assert out.shape == (2, 2)


def test_norm_linear_classifier_head_none_pool() -> None:
    """None pool expects pre-pooled input."""
    head = NormLinearClassifierHead(
        in_features=4, num_classes=2, pool_type="none"
    )
    x = torch.randn(2, 4)
    out = head(x)
    assert out.shape == (2, 2)


def test_relu_mlp_classifier_head_avg_pool() -> None:
    head = ReLUMLPClassifierHead(in_features=4, num_classes=2, pool_type="avg")
    x = torch.randn(2, 5, 4)
    out = head(x)
    assert out.shape == (2, 2)


def test_classifier_head_conv_sequence() -> None:
    """Conv classifier should handle sequence inputs."""
    head = ClassifierHead(
        in_features=4, num_classes=3, pool_type="none", use_conv=True
    )
    x = torch.randn(2, 4, 5)
    out = head(x)
    assert out.shape == (2, 3, 5)


def test_classifier_head_properties() -> None:
    head = NormLinearClassifierHead(
        in_features=4, num_classes=2, pool_type="avg", drop_rate=0.1
    )
    assert head.features_in == 4
    assert head.features_out == 2
    assert head.has_dropout
    assert head.is_pooled
    assert head.total_params == sum(p.numel() for p in head.parameters())


def test_pool_layers() -> None:
    token_pool = energy_transformer.layers.heads._create_pool("token")
    x = torch.randn(2, 3, 4)
    assert torch.allclose(token_pool(x), x[:, 0])

    max_pool = energy_transformer.layers.heads._create_pool("max")
    assert max_pool(torch.randn(2, 3, 4)).shape == (2, 4)
    with pytest.raises(ValueError, match="Expected 2D or 3D"):
        max_pool(torch.randn(2, 3, 4, 5))

    avg_pool = energy_transformer.layers.heads._create_pool("avg")
    y = torch.randn(2, 4)
    assert torch.allclose(avg_pool(y), y)


def test_global_max_pool_2d() -> None:
    pool = energy_transformer.layers.heads._GlobalMaxPool()
    x = torch.randn(2, 4)
    out = pool(x)
    assert torch.allclose(out, x)


def test_global_max_pool_invalid_dims() -> None:
    pool = energy_transformer.layers.heads._GlobalMaxPool()
    with pytest.raises(ValueError, match="Expected 2D or 3D"):
        pool(torch.randn(2, 3, 4, 5))


def test_global_avg_pool_2d() -> None:
    pool = energy_transformer.layers.heads._GlobalAvgPool()
    x = torch.randn(2, 4)
    out = pool(x)
    assert torch.allclose(out, x)
