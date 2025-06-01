"""Test metrics collection."""

import pytest

from energy_transformer.spec import (
    configure_realisation,
    get_realisation_metrics,
    realise,
    reset_metrics,
    seq,
)
from energy_transformer.spec.library import ETBlockSpec, LayerNormSpec


@pytest.mark.unit
def test_metrics_collection() -> None:
    """Test that metrics are collected when enabled."""
    configure_realisation(enable_metrics=True)
    reset_metrics()

    spec = seq(
        LayerNormSpec(),
        ETBlockSpec(),
        LayerNormSpec(),
    )
    realise(spec, embed_dim=768)

    metrics = get_realisation_metrics()
    assert metrics["specs_realised"] > 0
    assert metrics["total_time"] > 0
    assert "spec_stats" in metrics
    assert "Sequential" in metrics["spec_stats"]
    assert "LayerNormSpec" in metrics["spec_stats"]
    assert "ETBlockSpec" in metrics["spec_stats"]

    configure_realisation(enable_metrics=False)


@pytest.mark.unit
def test_metrics_disabled_by_default() -> None:
    """Test that metrics return empty when disabled."""
    configure_realisation(enable_metrics=False)
    reset_metrics()

    spec = ETBlockSpec()
    realise(spec, embed_dim=768)

    metrics = get_realisation_metrics()
    assert metrics == {}


@pytest.mark.unit
def test_cache_metrics() -> None:
    """Test cache hit/miss metrics."""
    configure_realisation(enable_metrics=True)
    reset_metrics()

    spec = LayerNormSpec()
    realise(spec, embed_dim=768)
    realise(spec, embed_dim=768)

    metrics = get_realisation_metrics()
    assert metrics["cache_hit_rate"] > 0
    assert metrics["specs_realised"] >= 1

    configure_realisation(enable_metrics=False)

