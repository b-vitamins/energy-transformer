"""Test debug tracing functionality."""

import pytest
from energy_transformer.spec import Context, realise, seq
from energy_transformer.spec.debug import debug_realisation
from energy_transformer.spec.library import ETBlockSpec, LayerNormSpec
from energy_transformer.spec.primitives import ValidationError
from energy_transformer.spec.realise import RealisationError


def test_debug_trace_basic() -> None:
    """Test basic debug tracing."""
    spec = seq(
        LayerNormSpec(),
        ETBlockSpec(),
    )

    with debug_realisation(trace_realisation=True) as tracer:
        realise(spec, embed_dim=768)

    events = tracer.get_trace_events()
    assert len(events) > 0

    enter_events = [e for e in events if e.event_type == "enter"]
    exit_events = [e for e in events if e.event_type == "exit"]
    assert len(enter_events) > 0
    assert len(exit_events) > 0

    spec_types = {e.spec_type for e in events}
    assert "Sequential" in spec_types
    assert "LayerNormSpec" in spec_types
    assert "ETBlockSpec" in spec_types


def test_debug_trace_cache_hits() -> None:
    """Test debug tracing of cache hits."""
    spec = LayerNormSpec()
    ctx = Context(dimensions={"embed_dim": 768})

    with debug_realisation(trace_realisation=True) as tracer:
        realise(spec, ctx)
        realise(spec, ctx)

    events = tracer.get_trace_events()
    cache_hits = [e for e in events if e.event_type == "cache_hit"]
    assert len(cache_hits) >= 1


def test_debug_trace_errors() -> None:
    """Test debug tracing of errors."""
    spec = ETBlockSpec()

    # Disable strict validation so error occurs during realisation
    from energy_transformer.spec import configure_realisation

    configure_realisation(strict=False)
    try:
        with debug_realisation(trace_realisation=True) as tracer:
            with pytest.raises(RealisationError):
                realise(spec)

        events = tracer.get_trace_events()
        error_events = [e for e in events if e.event_type == "error"]
        assert len(error_events) > 0
    finally:
        configure_realisation(strict=True)


def test_debug_trace_timing() -> None:
    """Test that timing information is recorded."""
    spec = seq(
        LayerNormSpec(),
        LayerNormSpec(),
    )

    with debug_realisation(trace_realisation=True) as tracer:
        realise(spec, embed_dim=768)

    events = tracer.get_trace_events()
    exit_events = [e for e in events if e.event_type == "exit"]
    timed_events = [e for e in exit_events if e.duration is not None]
    assert len(timed_events) > 0
    for event in timed_events:
        assert event.duration > 0
