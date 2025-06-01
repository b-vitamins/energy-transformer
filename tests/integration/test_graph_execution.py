"""Test graph execution and validation order fixes.

This module verifies:
1. GraphModule executes with correct topology
2. Validation order allows parent->child dimension flow
3. Parallel merge validation catches errors
4. Graph cycle detection works early
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from energy_transformer.spec import (
    Context,
    Spec,
    ValidationError,
    graph,
    param,
    realise,
)
from energy_transformer.spec.combinators import Graph
from energy_transformer.spec.primitives import provides, requires
from energy_transformer.spec.realise import GraphModule

pytestmark = pytest.mark.integration


@dataclass(frozen=True)
@provides("test_dim")
class ProviderSpec(Spec):
    """Spec that provides a dimension."""

    value: int = param(default=128)

    def apply_context(self, context: Context) -> Context:
        context = super().apply_context(context)
        context.set_dim("test_dim", self.value)
        return context


@dataclass(frozen=True)
@requires("test_dim")
class ConsumerSpec(Spec):
    """Spec that requires a dimension."""

    multiplier: int = param(default=2)

    def validate(self, context: Context) -> list[str]:
        issues = super().validate(context)
        if context.get_dim("test_dim") is None:
            issues.append("test_dim is required but not found")
        return issues


class TestGraphExecution:
    """Test that GraphModule executes with correct data flow."""

    def test_simple_linear_graph(self):
        """Test A -> B -> C linear graph execution."""

        class AddOne(nn.Module):
            def forward(self, x):
                return x + 1

        class MultiplyTwo(nn.Module):
            def forward(self, x):
                return x * 2

        class SubtractThree(nn.Module):
            def forward(self, x):
                return x - 3

        nodes = {
            "add": AddOne(),
            "mul": MultiplyTwo(),
            "sub": SubtractThree(),
        }
        edges = [
            ("input", "add", None),
            ("add", "mul", None),
            ("mul", "sub", None),
        ]
        gm = GraphModule(
            nodes=nodes,
            edges=edges,
            inputs=["input"],
            outputs=["sub"],
        )

        x = torch.tensor([5.0])
        result = gm(x)
        assert torch.allclose(result, torch.tensor([9.0]))

    def test_branching_graph(self):
        """Test graph with branching: A -> B,C; B,C -> D."""

        class Identity(nn.Module):
            def forward(self, x):
                return x

        class Add(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, x):
                return x + self.value

        class Concat(nn.Module):
            def forward(self, x):
                return x

        nodes = {
            "branch1": Add(10),
            "branch2": Add(20),
            "merge": Concat(),
        }
        edges = [
            ("input", "branch1", None),
            ("input", "branch2", None),
            ("branch1", "merge", None),
            ("branch2", "merge", None),
        ]
        gm = GraphModule(
            nodes=nodes,
            edges=edges,
            inputs=["input"],
            outputs=["merge"],
        )

        x = torch.tensor([[1.0, 2.0]])
        result = gm(x)
        expected = torch.tensor([[11.0, 12.0, 21.0, 22.0]])
        assert torch.allclose(result, expected)

    def test_graph_with_edge_transforms(self):
        """Test edge transformations in graph."""

        class Identity(nn.Module):
            def forward(self, x):
                return x

        nodes = {"node1": Identity(), "node2": Identity()}
        edges = [("input", "node1", None), ("node1", "node2", "relu")]
        gm = GraphModule(
            nodes=nodes,
            edges=edges,
            inputs=["input"],
            outputs=["node2"],
        )

        x = torch.tensor([[-1.0, 2.0, -3.0]])
        result = gm(x)
        expected = torch.tensor([[0.0, 2.0, 0.0]])
        assert torch.allclose(result, expected)

    def test_graph_missing_input_error(self):
        """Test graph fails gracefully with missing inputs."""
        nodes = {"node": nn.Identity()}
        edges = [("missing_input", "node", None)]
        gm = GraphModule(
            nodes=nodes,
            edges=edges,
            inputs=["input"],
            outputs=["node"],
        )

        with pytest.raises(RuntimeError, match="not available"):
            gm(torch.tensor([1.0]))

    def test_verify_graph_fix_script_behavior(self):
        """Test the exact computation from verify_graph_fix.py."""
        from dataclasses import dataclass

        import torch
        from torch import nn

        from energy_transformer.spec import Spec, graph, param
        from energy_transformer.spec.realise import register_typed

        class AddModule(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, x):
                return x + self.value

        @dataclass(frozen=True)
        class AddSpec(Spec):
            value: int = param()

        @register_typed
        def _realise_add(spec: AddSpec, _context: Context) -> nn.Module:
            return AddModule(spec.value)

        g = graph()
        g = g.add_node("layer1", AddSpec(1))
        g = g.add_node("layer2", AddSpec(10))
        g = g.add_node("layer3", AddSpec(100))

        g = g.add_edge("input", "layer1")
        g = g.add_edge("layer1", "layer2")
        g = g.add_edge("layer2", "layer3")

        graph_spec = Graph(
            nodes=g.nodes,
            edges=g.edges,
            inputs=["input"],
            outputs=["layer3"],
        )

        x = torch.tensor([0.0])
        module = realise(graph_spec)
        result = module(x)

        expected = 111.0
        assert torch.allclose(result, torch.tensor([expected])), (
            f"Graph computation failed: expected {expected}, got {result.item()}"
        )
        from energy_transformer.spec.primitives import SpecMeta

        SpecMeta._realisers.pop(AddSpec, None)

    def test_graph_execution_order(self, capfd):
        """Verify nodes execute in correct order with proper inputs."""
        import torch
        from torch import nn

        from energy_transformer.spec.realise import GraphModule

        class LoggingModule(nn.Module):
            def __init__(self, name):
                super().__init__()
                self.name = name

            def forward(self, x):
                print(f"{self.name} received: {x.item():.1f}")
                return x + 1

        nodes = {
            "A": LoggingModule("A"),
            "B": LoggingModule("B"),
            "C": LoggingModule("C"),
        }
        edges = [
            ("input", "A"),
            ("A", "B"),
            ("B", "C"),
        ]

        graph_module = GraphModule(nodes, edges, ["input"], ["C"])

        result = graph_module(torch.tensor([0.0]))

        captured = capfd.readouterr()
        assert "A received: 0.0" in captured.out
        assert "B received: 1.0" in captured.out
        assert "C received: 2.0" in captured.out
        assert result.item() == 3.0

    def test_graph_cycle_detection(self):
        """Test that graphs with cycles are caught."""
        g = graph()
        g = g.add_node("A", ProviderSpec())
        g = g.add_node("B", ConsumerSpec())
        g = g.add_node("C", ConsumerSpec())

        g = g.add_edge("A", "B")
        g = g.add_edge("B", "C")
        with pytest.raises(ValidationError, match="cycle"):
            g = g.add_edge("C", "B")  # Cycle detected early
