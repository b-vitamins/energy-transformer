import pytest

from energy_transformer import models

pytestmark = pytest.mark.unit


def test_lazy_load_energy_transformer() -> None:
    cls_first = models.EnergyTransformer
    assert cls_first.__name__ == "EnergyTransformer"
    assert models.EnergyTransformer is cls_first


def test_lazy_load_invalid_attr() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = models.Foo
