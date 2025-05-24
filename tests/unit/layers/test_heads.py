import torch

from energy_transformer.layers.heads import ClassificationHead, FeatureHead


def test_classification_head_output_shape() -> None:
    head = ClassificationHead(embed_dim=4, num_classes=3, drop_rate=0.0)
    x = torch.ones(2, 1, 4)
    out = head(x)
    assert out.shape == (2, 3)
    assert torch.all(out == 0)


def test_classification_head_pooling() -> None:
    head = ClassificationHead(embed_dim=4, num_classes=2, use_cls_token=False, drop_rate=0.0)
    x = torch.randn(2, 3, 4)
    out = head(x)
    assert out.shape == (2, 2)


def test_feature_head_global_avg() -> None:
    head = FeatureHead(use_cls_token=False)
    x = torch.arange(6.0).view(1, 3, 2)
    out = head(x)
    expected = x.mean(dim=1)
    assert torch.allclose(out, expected)

