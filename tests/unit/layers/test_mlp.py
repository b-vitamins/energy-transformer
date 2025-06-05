import pytest
import torch

from energy_transformer.layers.mlp import MLP

pytestmark = pytest.mark.unit


def test_mlp_forward_and_properties() -> None:
    mlp = MLP(in_features=4, hidden_features=8, out_features=2, drop=0.1)
    x = torch.randn(2, 4)
    out = mlp(x)
    assert out.shape == (2, 2)
    assert mlp.expansion_ratio == 2.0
    assert mlp.features_in == 4
    assert mlp.features_hidden == 8
    assert mlp.features_out == 2
    assert mlp.has_bias
    assert mlp.activation_name.lower() in {"gelu", "relu"}
