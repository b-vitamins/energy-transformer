import pytest

from energy_transformer.models.vision.utils import create_model_config


def test_create_model_config_defaults():
    cfg = create_model_config()
    assert cfg["embed_dim"] == 768
    assert cfg["img_size"] == 224
    assert cfg["patch_size"] == 16
    assert cfg["num_classes"] == 1000
    assert cfg["head_dim"] == 64
    assert cfg["et_steps"] == 4


def test_create_model_config_override_and_size():
    cfg = create_model_config("small", img_size=128, num_classes=10)
    assert cfg["embed_dim"] == 384
    assert cfg["img_size"] == 128
    assert cfg["num_classes"] == 10


def test_create_model_config_unknown_size():
    with pytest.raises(ValueError):
        create_model_config("giant")
