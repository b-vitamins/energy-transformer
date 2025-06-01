"""Test security fixes in realise module."""

import builtins

import pytest
import torch

from energy_transformer.spec.realise import GraphModule


class TestGraphModuleSecurity:
    """Test that GraphModule transformations are secure."""

    def test_edge_transform_no_eval(self, monkeypatch):
        called = False

        def fake_eval(_expr):
            nonlocal called
            called = True
            raise AssertionError("eval called")

        monkeypatch.setattr(builtins, "eval", fake_eval)
        gm = GraphModule({}, [], [], [])
        out = gm._apply_edge_transform(torch.arange(5), "[0]")
        assert out.item() == 0
        assert not called

    def test_safe_indexing(self):
        gm = GraphModule({}, [], [], [])
        t = torch.arange(10)
        assert torch.equal(gm._apply_edge_transform(t, "[2]"), t[2])
        assert torch.equal(gm._apply_edge_transform(t, "[1:4]"), t[1:4])
        assert torch.equal(gm._apply_edge_transform(t, "[...]"), t[...])
        assert torch.equal(gm._apply_edge_transform(t, "detach"), t.detach())

    def test_malicious_transform_rejected(self):
        gm = GraphModule({}, [], [], [])
        dangerous_inputs = [
            "__import__('os').system('ls')",
            "exec('print(1)')",
            "eval('1+1')",
            "[__import__('os')]",
        ]
        for inp in dangerous_inputs:
            with pytest.raises(ValueError, match="Unknown|Unsupported|Invalid"):
                gm._apply_edge_transform(torch.arange(3), inp)
