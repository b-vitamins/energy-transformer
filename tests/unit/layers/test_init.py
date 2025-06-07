import pytest

from energy_transformer import layers

pytestmark = pytest.mark.unit


def test_lazy_load_valid_attr() -> None:
    mlp_first = layers.MLP
    assert mlp_first.__name__ == "MLP"
    assert layers.MLP is mlp_first


def test_lazy_load_invalid_attr() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = layers.DoesNotExist
