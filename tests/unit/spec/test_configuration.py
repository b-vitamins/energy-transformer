"""Test configuration management."""

from energy_transformer.spec import configure_realisation
from energy_transformer.spec.realise import (
    RealisationConstants,
    _get_config,
    _thread_local,
)


def test_configure_constants():
    """Test configuring realisation constants."""
    if hasattr(_thread_local, "config"):
        delattr(_thread_local, "config")

    configure_realisation(
        MAX_RECURSION=200,
        UNROLL_LIMIT=20,
        DEFAULT_CACHE_SIZE=256,
    )

    config = _get_config()
    assert config.constants.MAX_RECURSION == 200
    assert config.constants.UNROLL_LIMIT == 20
    assert config.constants.DEFAULT_CACHE_SIZE == 256

    assert config.constants.MAX_STACK_PREVIEW == 5
    assert config.constants.EDGE_TUPLE_SIZE == 2


def test_configure_constants_object():
    """Test configuring with RealisationConstants object."""
    if hasattr(_thread_local, "config"):
        delattr(_thread_local, "config")

    custom_constants = RealisationConstants(
        MAX_RECURSION=50,
        MAX_STACK_PREVIEW=10,
        UNROLL_LIMIT=5,
        DEFAULT_CACHE_SIZE=64,
    )

    configure_realisation(constants=custom_constants)

    config = _get_config()
    assert config.constants.MAX_RECURSION == 50
    assert config.constants.MAX_STACK_PREVIEW == 10
    assert config.constants.UNROLL_LIMIT == 5
    assert config.constants.DEFAULT_CACHE_SIZE == 64
