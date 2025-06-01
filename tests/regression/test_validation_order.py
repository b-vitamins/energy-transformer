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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from energy_transformer.spec import (
    Context,
    Spec,
    ValidationError,
    graph,
    parallel,
    param,
    realise,
    seq,
)
from energy_transformer.spec.combinators import Graph
from energy_transformer.spec.primitives import Dimension, provides, requires
from energy_transformer.spec.realise import GraphModule


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

        from energy_transformer.spec import Spec, graph, param, realise
        from energy_transformer.spec.combinators import Graph
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


class TestValidationOrder:
    """Test that validation order allows proper context propagation."""

    def test_parent_provides_child_requires(self):
        """Test parent spec providing dimension to child."""
        spec = seq(ProviderSpec(value=256), ConsumerSpec())
        ctx = Context()
        issues = spec.validate(ctx)
        assert len(issues) == 0

    def test_child_sees_parent_dimension(self):
        """Test child validation sees parent's context updates."""

        @dataclass(frozen=True)
        @requires("embed_dim", "test_dim")
        class ChildSpec(Spec):
            pass

        @dataclass(frozen=True)
        @provides("embed_dim", "test_dim")
        class ParentSpec(Spec):
            children_list: list[Spec] = param(default_factory=list)

            def apply_context(self, context: Context) -> Context:
                context = super().apply_context(context)
                context.set_dim("embed_dim", 768)
                context.set_dim("test_dim", 64)
                return context

            def children(self) -> list[Spec]:
                return self.children_list

        child = ChildSpec()
        parent = ParentSpec(children_list=[child])
        ctx = Context()
        issues = parent.validate(ctx)
        assert len(issues) == 0

    def test_validation_with_computed_dimensions(self):
        """Test validation with dimensions computed from formulas."""

        @dataclass(frozen=True)
        @provides("computed_dim")
        @requires("base_dim")
        class ComputeSpec(Spec):
            formula: str = param(default="base_dim * 4")

            def apply_context(self, context: Context) -> Context:
                context = super().apply_context(context)
                dim = Dimension("computed_dim", formula=self.formula)
                if value := dim.resolve(context):
                    context.set_dim("computed_dim", value)
                return context

        spec = ComputeSpec()
        ctx = Context(dimensions={"base_dim": 128})
        issues = spec.validate(ctx)
        assert len(issues) == 0

        updated_ctx = spec.apply_context(ctx)
        assert updated_ctx.get_dim("computed_dim") == 512

    def test_verify_validation_order_script_behavior(self):
        """Test the exact behavior from verify_validation_order.py."""
        from energy_transformer.spec import Context, seq
        from energy_transformer.spec.library import LayerNormSpec, MLPSpec

        spec = seq(
            LayerNormSpec(),
            MLPSpec(),
        )

        ctx = Context(dimensions={"embed_dim": 768})
        issues = spec.validate(ctx)

        assert len(issues) == 0, f"Validation failed with: {issues}"


class TestParallelMergeValidation:
    """Test Parallel spec validation with different merge strategies."""

    def test_parallel_add_requires_same_dims(self):
        """Test that add merge requires compatible dimensions."""

        @dataclass(frozen=True)
        @provides("embed_dim")
        class DimSpec(Spec):
            dim: int = param()

            def apply_context(self, context: Context) -> Context:
                context = super().apply_context(context)
                context.set_dim("embed_dim", self.dim)
                return context

        spec = parallel(DimSpec(dim=128), DimSpec(dim=256), merge="add")
        ctx = Context()
        issues = spec.validate(ctx)
        assert any("Incompatible dimensions" in i for i in issues)

    def test_parallel_concat_allows_different_dims(self):
        """Test that concat merge allows different dimensions."""

        @dataclass(frozen=True)
        @provides("embed_dim")
        class DimSpec(Spec):
            dim: int = param()

            def apply_context(self, context: Context) -> Context:
                context = super().apply_context(context)
                context.set_dim("embed_dim", self.dim)
                return context

        spec = parallel(DimSpec(dim=128), DimSpec(dim=256), merge="concat")
        ctx = Context()
        issues = spec.validate(ctx)
        assert len(issues) == 0

    def test_parallel_missing_merge_dim(self):
        """Test validation catches missing merge dimension."""

        @dataclass(frozen=True)
        class NoDimSpec(Spec):
            pass

        spec = parallel(NoDimSpec(), NoDimSpec(), merge="add")
        ctx = Context()
        issues = spec.validate(ctx)
        assert any("does not provide" in i for i in issues)

    def test_parallel_weights_validation(self):
        """Test weight validation for parallel specs."""

        spec1 = parallel(
            ProviderSpec(),
            ProviderSpec(),
            ProviderSpec(),
            merge="add",
            weights=[0.5, 0.5],
        )
        issues = spec1.validate(Context())
        assert any("Weight count" in i for i in issues)

        spec2 = parallel(
            ProviderSpec(),
            ProviderSpec(),
            merge="add",
            weights=[0.3, 0.3],
        )
        issues = spec2.validate(Context())
        assert any("Warning" in i and "sum" in i for i in issues)


class TestIntegration:
    """Integration tests combining multiple fixes."""

    def test_complex_model_graph(self):
        """Test a realistic model with multiple paths."""

        from energy_transformer.spec.library import (
            IdentitySpec,
            LayerNormSpec,
        )
        from energy_transformer.spec.realise import register_typed

        @register_typed
        def _realise_layer_norm(
            spec: LayerNormSpec,
            context: Context,
        ) -> nn.Module:
            return nn.LayerNorm(context.get_dim("embed_dim"), eps=spec.eps)

        @register_typed
        def _realise_identity(
            _spec: IdentitySpec,
            _context: Context,
        ) -> nn.Module:
            return nn.Identity()

        g = graph()
        g = g.add_node("norm1", LayerNormSpec())
        g = g.add_node("norm2", LayerNormSpec())
        g = g.add_node("attn", IdentitySpec())
        g = g.add_node("mlp", IdentitySpec())
        g = g.add_node("add", IdentitySpec())

        g = g.add_edge("input", "norm1")
        g = g.add_edge("input", "norm2")
        g = g.add_edge("norm1", "attn")
        g = g.add_edge("norm2", "mlp")
        g = g.add_edge("attn", "add")
        g = g.add_edge("mlp", "add")

        graph_spec = Graph(
            nodes=g.nodes,
            edges=g.edges,
            inputs=["input"],
            outputs=["add"],
        )

        ctx = Context(dimensions={"embed_dim": 768})
        issues = graph_spec.validate(ctx)
        assert len(issues) == 0

        model = realise(graph_spec, embed_dim=768)
        x = torch.randn(2, 10, 768)
        output = model(x)
        assert output.shape == (2, 10, 768 * 2)

    def test_nested_validation_context_flow(self):
        """Test deeply nested specs with context propagation."""

        @dataclass(frozen=True)
        @provides("test_dim", "embed_dim")
        class BothProvider(Spec):
            value: int = param(default=64)

            def apply_context(self, context: Context) -> Context:
                context = super().apply_context(context)
                context.set_dim("test_dim", self.value)
                context.set_dim("embed_dim", self.value)
                return context

        spec = seq(
            BothProvider(value=64),
            seq(
                ConsumerSpec(),
                parallel(ConsumerSpec(), ConsumerSpec(), merge="add"),
            ),
        )
        ctx = Context()
        issues = spec.validate(ctx)
        assert len(issues) == 0


class TestErrorMessages:
    """Test that error messages are helpful and accurate."""

    def test_cycle_error_shows_path(self):
        """Test cycle detection shows the cycle path."""
        g = graph()
        g = g.add_node("start", ProviderSpec())
        g = g.add_node("middle", ConsumerSpec())
        g = g.add_node("end", ConsumerSpec())

        g = g.add_edge("start", "middle")
        g = g.add_edge("middle", "end")
        with pytest.raises(ValidationError) as exc:
            g = g.add_edge("end", "middle")
        err = str(exc.value)
        assert "cycle" in err.lower()
        assert "middle" in err
        assert "end" in err

    def test_missing_dimension_error_context(self):
        """Test missing dimension errors show context."""
        spec = ConsumerSpec()
        ctx = Context(dimensions={"other_dim": 123})
        issues = spec.validate(ctx)
        assert issues
        assert any("test_dim" in i for i in issues)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
