"""Constants used across Energy Transformer layers."""

from __future__ import annotations

import torch

__all__ = [
    "ATTENTION_EPSILON",
    "ATTENTION_INIT_STD",
    "DEFAULT_COMPUTE_DTYPE",
    "DEFAULT_EPSILON",
    "DEFAULT_HOPFIELD_BETA",
    "DEFAULT_HOPFIELD_MULTIPLIER",
    "DEFAULT_IMAGE_CHANNELS",
    "DEFAULT_INIT_STD",
    "DEFAULT_LAYER_NORM_REGULARIZATION",
    "DEFAULT_MLP_RATIO",
    "HEAD_INIT_STD",
    "MEMORY_EFFICIENT_SEQ_THRESHOLD",
    "MIN_SEQUENCE_LENGTH",
    "MIXED_PRECISION_DTYPES",
    "SMALL_INIT_STD",
    "ZERO_INIT_STD",
    "PoolType",
]

# Numerical stability constants
DEFAULT_EPSILON: float = 1e-5
ATTENTION_EPSILON: float = 1e-6

# Initialization scales
DEFAULT_INIT_STD: float = 0.02
ATTENTION_INIT_STD: float = 0.002
HEAD_INIT_STD: float = 0.02
SMALL_INIT_STD: float = 0.01
ZERO_INIT_STD: float = 0.0

# Default hyperparameters
DEFAULT_LAYER_NORM_REGULARIZATION: float = 0.0
DEFAULT_HOPFIELD_MULTIPLIER: float = 4.0
DEFAULT_HOPFIELD_BETA: float = 0.01
DEFAULT_MLP_RATIO: float = 4.0
DEFAULT_IMAGE_CHANNELS: int = 3

# Dtype handling
MIXED_PRECISION_DTYPES: set[torch.dtype] = {torch.float16, torch.bfloat16}
DEFAULT_COMPUTE_DTYPE: torch.dtype = torch.float32

# Dimension thresholds
MEMORY_EFFICIENT_SEQ_THRESHOLD: int = 512
MIN_SEQUENCE_LENGTH: int = 1


class PoolType:
    """String constants for pooling types."""

    TOKEN = "token"  # noqa: S105
    AVG = "avg"
    MAX = "max"
    NONE = "none"
