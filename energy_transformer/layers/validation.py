"""Input validation utilities for Energy Transformer layers."""

from collections.abc import Sequence

from torch import Tensor


def validate_tensor_dim(
    x: Tensor,
    expected_dim: int,
    module_name: str,
    param_name: str = "input",
) -> None:
    """Validate tensor dimensionality."""
    if x.dim() != expected_dim:
        msg = (
            f"{module_name}: {param_name} must be {expected_dim}D tensor. "
            f"Expected shape: {'[' + ', '.join(['...'] * expected_dim) + ']'}, "
            f"got shape: {list(x.shape)} ({x.dim()}D)."
        )
        raise ValueError(msg)


def validate_shape_match(
    x: Tensor,
    expected_shape: Sequence[int],
    module_name: str,
    param_name: str = "input",
    dims_to_check: Sequence[int] | None = None,
) -> None:
    """Validate specific dimensions of tensor shape."""
    if dims_to_check is None:
        dims_to_check = range(len(expected_shape))

    for dim_idx in dims_to_check:
        if dim_idx >= x.dim():
            msg = (
                f"{module_name}: {param_name} has insufficient dimensions. "
                f"Expected at least {dim_idx + 1} dimensions, got {x.dim()}."
            )
            raise ValueError(msg)
        if (
            expected_shape[dim_idx] != -1
            and x.shape[dim_idx] != expected_shape[dim_idx]
        ):
            msg = (
                f"{module_name}: {param_name} dimension {dim_idx} mismatch. "
                f"Expected: {expected_shape[dim_idx]}, got: {x.shape[dim_idx]}."
            )
            raise ValueError(msg)


def validate_divisibility(
    value: int,
    divisor: int,
    module_name: str,
    value_name: str,
    divisor_name: str,
) -> None:
    """Validate that value is divisible by divisor."""
    if value % divisor != 0:
        msg = (
            f"{module_name}: {value_name} must be divisible by {divisor_name}. "
            f"Got {value_name}={value}, {divisor_name}={divisor}, "
            f"remainder={value % divisor}."
        )
        raise ValueError(msg)


def validate_positive(
    value: float,
    module_name: str,
    param_name: str,
) -> None:
    """Validate that value is positive."""
    if value <= 0:
        msg = (
            f"{module_name}: {param_name} must be positive. "
            f"Got {param_name}={value}."
        )
        raise ValueError(msg)


def format_shape_error(
    module_name: str,
    expected: str,
    actual: str,
    context: str = "",
) -> str:
    """Format a standardized shape error message."""
    msg = f"{module_name}: Shape mismatch"
    if context:
        msg += f" {context}"
    msg += f". Expected: {expected}, got: {actual}."
    return msg
