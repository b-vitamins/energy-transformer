"""Additional tests for public specification API.

These tests cover the convenience functions and utilities exposed in
:mod:`energy_transformer.spec.__init__` that were previously
uncovered by the main test suite.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from dataclasses import dataclass
from unittest.mock import patch

import pytest

import energy_transformer.spec as et_spec
from energy_transformer.spec import Spec, requires


def test_getattr_lazy_loading() -> None:
    """``__getattr__`` lazily loads library specifications."""
    cls = getattr(et_spec, "LayerNormSpec")
    from energy_transformer.spec import library

    assert cls is library.LayerNormSpec


def test_getattr_unknown() -> None:
    """Accessing an unknown attribute raises :class:`AttributeError`."""
    with pytest.raises(AttributeError):
        getattr(et_spec, "DoesNotExist")


def test_initialize_defaults_calls_configure() -> None:
    """``initialize_defaults`` should forward defaults to ``configure_realisation``."""
    with patch("energy_transformer.spec.realise.configure_realisation") as conf:
        et_spec.initialize_defaults()
        conf.assert_called_once()
        args, kwargs = conf.call_args
        assert kwargs["cache"].max_size == 128
        assert kwargs["strict"] is True
        assert kwargs["warnings"] is True
        assert kwargs["auto_import"] is True
        assert kwargs["optimizations"] is True
        assert kwargs["max_recursion"] == 100


def test_quickstart_prints_guide() -> None:
    """``quickstart`` only prints a help message."""
    out = io.StringIO()
    with redirect_stdout(out):
        et_spec.quickstart()
    text = out.getvalue()
    assert "Energy Transformer Specification System" in text


def test_export_patterns_create_specs() -> None:
    """``export_patterns`` returns callable constructors for patterns."""
    patterns = et_spec.export_patterns()
    tiny = patterns["vit_tiny"]()
    base = patterns["vit_base"]()
    assert isinstance(tiny, Spec)
    assert isinstance(base, Spec)


@dataclass(frozen=True)
@requires("foo")
class NeedsFooSpec(Spec):
    """Spec requiring a dimension that will be missing."""

    pass


def test_validate_spec_tree_detects_issue() -> None:
    """``validate_spec_tree`` returns issues for invalid trees."""
    issues = et_spec.validate_spec_tree(NeedsFooSpec())
    assert issues and "foo" in issues[0]


def test_validate_spec_tree_verbose(capsys) -> None:
    """Verbose mode prints information and still returns issues."""
    issues = et_spec.validate_spec_tree(NeedsFooSpec(), verbose=True)
    captured = capsys.readouterr()
    assert "Validating" in captured.out
    assert issues


def test_validate_spec_tree_verbose_no_issue(capsys) -> None:
    """Validation prints success message when no issues are found."""
    spec = et_spec.seq(et_spec.IdentitySpec(), et_spec.IdentitySpec())
    issues = et_spec.validate_spec_tree(spec, verbose=True)
    out = capsys.readouterr().out
    assert "No issues found" in out
    assert issues == []


def test_benchmark_realisation_runs() -> None:
    """``benchmark_realisation`` executes realisation multiple times."""
    spec = et_spec.IdentitySpec()
    stats = et_spec.benchmark_realisation(spec, iterations=2)
    assert stats["iterations"] == 2
    assert stats["total"] > 0

