"""Tests for :mod:`energy_transformer.spec.debug` utilities."""

from __future__ import annotations

import logging
from contextlib import redirect_stdout
from io import StringIO

import pytest
pytestmark = pytest.mark.unit
from torch import nn

from energy_transformer.spec import Context
from energy_transformer.spec.debug import (
    clear_cache,
    debug_realisation,
    inspect_cache_stats,
)
from energy_transformer.spec.realise import ModuleCache, _config
from tests.unit.spec.test_realise import SimpleSpec


def test_debug_realisation_traces_cache(caplog) -> None:
    """Cache operations are logged when tracing is enabled."""
    cache = ModuleCache()
    _config.cache = cache

    caplog.set_level(logging.DEBUG)
    with debug_realisation(trace_cache=True):
        _config.cache.get(SimpleSpec(), Context())
        _config.cache.put(SimpleSpec(), Context(), nn.Identity())

    assert any("Cache MISS" in rec.message for rec in caplog.records)
    assert any("Cache PUT" in rec.message for rec in caplog.records)


@pytest.mark.usefixtures("capsys")
def test_inspect_and_clear_cache_output() -> None:
    """``inspect_cache_stats`` prints stats and ``clear_cache`` empties cache."""
    cache = ModuleCache()
    _config.cache = cache
    cache.put(SimpleSpec(), Context(), nn.Identity())

    with redirect_stdout(StringIO()) as buf:
        inspect_cache_stats()
        output = buf.getvalue()
        assert "Cache Statistics" in output
        assert "Size: 1" in output

    with redirect_stdout(StringIO()) as buf:
        clear_cache()
        assert "Cache cleared" in buf.getvalue()
    assert len(cache._cache) == 0


def test_debug_realisation_break_on_error(monkeypatch) -> None:
    """``debug_realisation`` invokes ``pdb.post_mortem`` when requested."""
    called = {}
    monkeypatch.setattr(
        "pdb.post_mortem",
        lambda: called.setdefault("pdb", True),
    )
    with pytest.raises(RuntimeError), debug_realisation(break_on_error=True):
        raise RuntimeError("boom")
    assert called.get("pdb")
