import torch
import torch.nn as nn

from energy_transformer.layers.heads import ClassificationHead, FeatureHead


def test_classification_head_output_shape() -> None:
    head = ClassificationHead(embed_dim=4, num_classes=3, drop_rate=0.0)
    x = torch.ones(2, 1, 4)
    out = head(x)
    assert out.shape == (2, 3)
    assert torch.all(out == 0)


def test_classification_head_pooling() -> None:
    head = ClassificationHead(
        embed_dim=4, num_classes=2, use_cls_token=False, drop_rate=0.0
    )
    x = torch.randn(2, 3, 4)
    out = head(x)
    assert out.shape == (2, 2)


def test_feature_head_global_avg() -> None:
    head = FeatureHead(use_cls_token=False)
    x = torch.arange(6.0).view(1, 3, 2)
    out = head(x)
    expected = x.mean(dim=1)
    assert torch.allclose(out, expected)


def test_feature_head_cls_token() -> None:
    head = FeatureHead()
    x = torch.arange(6.0).view(1, 3, 2)
    out = head(x)
    expected = x[:, 0]
    assert torch.allclose(out, expected)


def test_classification_head_representation_layer() -> None:
    head = ClassificationHead(
        embed_dim=4,
        num_classes=2,
        representation_size=6,
        drop_rate=0.0,
    )
    assert isinstance(head.pre_logits, nn.Sequential)
    assert isinstance(head.pre_logits[0], nn.Linear)
    assert isinstance(head.pre_logits[1], nn.Tanh)
    assert head.pre_logits[0].in_features == 4
    assert head.pre_logits[0].out_features == 6

    x = torch.ones(1, 1, 4)
    out = head(x)
    assert out.shape == (1, 2)


def test_classification_head_weight_initialization() -> None:
    head = ClassificationHead(embed_dim=3, num_classes=4)
    assert torch.all(head.head.weight == 0)
    assert torch.all(head.head.bias == 0)
