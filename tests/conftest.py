"""Pytest configuration and shared fixtures."""

import logging
import sys
from pathlib import Path

import pytest
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@pytest.fixture(scope="session")
def device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def simple_image_batch():
    """Create a simple batch of images for testing."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test."""
    from energy_transformer.spec.realise import _config

    _config.cache.clear()
    _config.strict = True
    _config.warnings = True
    _config.auto_import = True
    _config.optimizations = True
    _config.max_recursion = 100

    yield

    _config.cache.clear()


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary directory for cache tests."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


# Markers for categorizing tests


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests",
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line(
        "markers",
        "benchmark: marks performance benchmark tests",
    )
    config.addinivalue_line("markers", "security: marks security-related tests")


# Skip GPU tests if no GPU available


def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip GPU tests when appropriate."""
    _ = config  # Unused hook argument
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
