import pytest

from energy_transformer.models import vision

pytestmark = pytest.mark.unit


def test_lazy_load_viet_tiny() -> None:
    func_first = vision.viet_tiny
    assert callable(func_first)
    assert vision.viet_tiny is func_first


def test_lazy_load_invalid_attr() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = vision.bar
