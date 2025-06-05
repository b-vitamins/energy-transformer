import pytest
import torch

from energy_transformer.layers.validation import (
    validate_divisibility,
    validate_positive,
    validate_shape_match,
    validate_tensor_dim,
    format_shape_error,
)

pytestmark = pytest.mark.unit


def test_validate_tensor_dim_success() -> None:
    x = torch.zeros(2, 3, 4)
    validate_tensor_dim(x, 3, "TestModule")


def test_validate_tensor_dim_failure() -> None:
    x = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="TestModule: input must be 3D tensor"):
        validate_tensor_dim(x, 3, "TestModule")


def test_validate_shape_match_success() -> None:
    x = torch.zeros(2, 5, 4)
    validate_shape_match(x, (2, -1, 4), "TestModule", dims_to_check=[0, 2])


def test_validate_shape_match_insufficient_dims() -> None:
    x = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="insufficient dimensions"):
        validate_shape_match(x, (2, 3, 4), "TestModule")


def test_validate_shape_match_value_mismatch() -> None:
    x = torch.zeros(2, 3, 4)
    with pytest.raises(ValueError, match="dimension 1 mismatch"):
        validate_shape_match(x, (2, 5, 4), "TestModule")


def test_validate_divisibility_success() -> None:
    validate_divisibility(8, 4, "Mod", "val", "div")


def test_validate_divisibility_failure() -> None:
    with pytest.raises(ValueError, match="Mod: val must be divisible by div"):
        validate_divisibility(5, 3, "Mod", "val", "div")


def test_validate_positive_success() -> None:
    validate_positive(1.0, "Mod", "param")


def test_validate_positive_failure() -> None:
    with pytest.raises(ValueError, match="Mod: param must be positive"):
        validate_positive(0.0, "Mod", "param")


def test_format_shape_error() -> None:
    msg = format_shape_error("Mod", "(1,2)", "(3,4)")
    assert "Mod: Shape mismatch" in msg
    assert "Expected: (1,2), got: (3,4)" in msg
