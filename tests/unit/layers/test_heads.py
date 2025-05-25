import torch
import torch.nn as nn
import torch.nn.functional as F

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


def test_classification_head_pre_logits_identity_and_dropout_attrs() -> None:
    head = ClassificationHead(embed_dim=5, num_classes=2, drop_rate=0.3)
    assert isinstance(head.pre_logits, nn.Identity)
    assert isinstance(head.drop, nn.Dropout)
    assert head.drop.p == 0.3


def test_classification_head_dropout_behavior() -> None:
    head = ClassificationHead(embed_dim=2, num_classes=1, drop_rate=0.5)
    head.head.weight.data.fill_(1.0)
    head.head.bias.data.zero_()
    x = torch.ones(1, 1, 2)

    torch.manual_seed(1)
    head.train()
    out_train = head(x)
    torch.manual_seed(1)
    expected_train = F.dropout(x[:, 0], p=0.5, training=True) @ torch.ones(2, 1)
    assert torch.allclose(out_train, expected_train)

    torch.manual_seed(1)
    head.eval()
    out_eval = head(x)
    expected_eval = x[:, 0] @ torch.ones(2, 1)
    assert torch.allclose(out_eval, expected_eval)


def test_classification_head_representation_weights_not_zero() -> None:
    head = ClassificationHead(embed_dim=3, num_classes=2, representation_size=4)
    w = head.pre_logits[0].weight
    assert not torch.all(w == 0)


def test_classification_head_head_input_dim_dependent_on_representation_size() -> (
    None
):
    head_no_rep = ClassificationHead(embed_dim=4, num_classes=2)
    head_rep = ClassificationHead(
        embed_dim=4, num_classes=2, representation_size=6
    )
    assert head_no_rep.head.in_features == 4
    assert head_rep.head.in_features == 6


def test_classification_head_representation_bias_zero() -> None:
    head = ClassificationHead(embed_dim=3, num_classes=1, representation_size=5)
    bias = head.pre_logits[0].bias
    assert torch.all(bias == 0)


def test_classification_head_representation_forward() -> None:
    head = ClassificationHead(
        embed_dim=2,
        num_classes=2,
        representation_size=2,
        drop_rate=0.0,
    )
    with torch.no_grad():
        head.pre_logits[0].weight.copy_(torch.eye(2))
        head.pre_logits[0].bias.zero_()
        head.head.weight.copy_(torch.eye(2))
        head.head.bias.zero_()
    x = torch.tensor([[[1.0, -1.0]], [[-2.0, 3.0]]])
    out = head(x)
    expected = torch.tanh(x[:, 0])
    assert torch.allclose(out, expected)


def test_classification_head_global_average_outputs_expected() -> None:
    head = ClassificationHead(
        embed_dim=2, num_classes=1, use_cls_token=False, drop_rate=0.0
    )
    with torch.no_grad():
        head.head.weight.fill_(1.0)
        head.head.bias.zero_()
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    out = head(x)
    expected = torch.tensor([[5.0]])
    assert torch.allclose(out, expected)
