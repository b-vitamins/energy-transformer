"""Additional tests for realisation utilities and modules."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from energy_transformer.spec import Context, library
from energy_transformer.spec.realise import GraphModule, ModuleCache


def test_module_cache_etblock_key() -> None:
    """ETBlockSpec instances use their id in cache keys."""
    cache = ModuleCache()
    spec = library.ETBlockSpec(
        attention=library.MHEASpec(num_heads=1, head_dim=16),
        hopfield=library.HNSpec(),
    )
    key = cache._make_key(spec, Context())
    assert key[0] == "nocache" and key[1] == id(spec)


def test_debug_realisation_break_on_error(monkeypatch) -> None:
    """``debug_realisation`` invokes ``pdb.post_mortem`` when requested."""
    from energy_transformer.spec.debug import debug_realisation

    called = {}
    monkeypatch.setattr(
        "pdb.post_mortem",
        lambda: called.setdefault("pdb", True),
    )
    with pytest.raises(RuntimeError):
        with debug_realisation(break_on_error=True):
            raise RuntimeError("boom")
    assert called.get("pdb")


def test_validate_spec_tree_verbose_no_issues(capsys) -> None:
    """Verbose validation prints success and recurses."""
    import energy_transformer.spec as et_spec

    spec = et_spec.seq(et_spec.IdentitySpec(), et_spec.IdentitySpec())
    issues = et_spec.validate_spec_tree(spec, verbose=True)
    out = capsys.readouterr().out
    assert "No issues found" in out
    assert issues == []


def test_graph_module_errors() -> None:
    """GraphModule raises helpful errors on invalid graphs."""
    nodes = {"a": nn.Identity(), "b": nn.Identity()}
    # Cycle detection
    gm_cycle = GraphModule(
        nodes,
        [("a", "b"), ("b", "a")],
        inputs=["x"],
        outputs=["a"],
    )
    with pytest.raises(RuntimeError, match="cycles"):
        gm_cycle(torch.randn(1, 1))

    # Missing input value
    gm_missing = GraphModule(nodes, [("a", "b")], inputs=["x"], outputs=["b"])
    with pytest.raises(RuntimeError, match="no inputs"):
        gm_missing(torch.randn(1, 1))


def test_apply_edge_transform_variants() -> None:
    """``_apply_edge_transform`` handles known and unknown transforms."""
    gm = GraphModule({}, [], [], [])
    t = torch.ones(1)
    assert torch.allclose(gm._apply_edge_transform(t, "relu"), torch.relu(t))
    assert torch.allclose(
        gm._apply_edge_transform(t, "sigmoid"),
        torch.sigmoid(t),
    )
    with pytest.raises(ValueError):
        gm._apply_edge_transform(t, "unknown")
