"""Tests for specification combinators.

This module tests the combinator functionality without relying on
specific library specs, using minimal mock specs instead.
"""

from dataclasses import dataclass

import pytest

from energy_transformer.spec.combinators import (
    Conditional,
    Graph,
    Identity,
    Lambda,
    Loop,
    Parallel,
    Residual,
    Sequential,
    Switch,
    cond,
    graph,
    loop,
    mixture_of_experts,
    multi_scale,
    parallel,
    residual,
    seq,
    switch,
    transformer_block,
)
from energy_transformer.spec.primitives import (
    Context,
    Spec,
    ValidationError,
    param,
    provides,
    requires,
)

pytestmark = pytest.mark.unit


# Mock specs for testing combinators
@dataclass(frozen=True)
class MockSpec(Spec):
    """Simple mock spec for testing."""

    value: int = param(default=1)  # noqa: RUF009


@dataclass(frozen=True)
@provides("output_dim", "input_dim")  # Also provide input_dim
class MockProviderSpec(Spec):
    """Mock spec that provides dimensions."""

    output_dim: int = param(default=100)  # noqa: RUF009

    def apply_context(self, context: Context) -> Context:
        context = super().apply_context(context)
        context.set_dim("output_dim", self.output_dim)
        context.set_dim("input_dim", self.output_dim)  # Add this
        return context


@dataclass(frozen=True)
@requires("input_dim")
class MockRequirerSpec(Spec):
    """Mock spec that requires dimensions."""

    multiplier: int = param(default=2)  # noqa: RUF009


@dataclass(frozen=True)
class MockTransformSpec(Spec):
    """Mock spec for transformations."""

    transform: str = param(default="identity")  # noqa: RUF009


class TestSequential:
    """Test Sequential combinator."""

    def test_empty_sequential(self):
        """Test empty sequential creation."""
        s = seq()
        assert isinstance(s, Sequential)
        assert len(s) == 0
        assert list(s) == []

    def test_single_spec_passthrough(self):
        """Test that single spec is returned directly."""
        spec = MockSpec(value=42)
        result = seq(spec)
        assert result is spec
        assert not isinstance(result, Sequential)

    def test_basic_composition(self):
        """Test basic sequential composition."""
        s = seq(MockSpec(1), MockSpec(2), MockSpec(3))

        assert isinstance(s, Sequential)
        assert len(s) == 3
        assert s[0].value == 1
        assert s[1].value == 2
        assert s[2].value == 3

    def test_operator_chaining(self):
        """Test >> and << operators."""
        # Right shift
        s1 = MockSpec(1) >> MockSpec(2) >> MockSpec(3)
        assert isinstance(s1, Sequential)
        assert len(s1) == 3
        assert [p.value for p in s1] == [1, 2, 3]

        # Left shift
        s2 = MockSpec(3) << MockSpec(2) << MockSpec(1)
        assert isinstance(s2, Sequential)
        assert len(s2) == 3
        assert [p.value for p in s2] == [1, 2, 3]

        # Mixed operators
        s3 = MockSpec(1) >> (MockSpec(3) << MockSpec(2))
        assert len(s3) == 2  # Outer Sequential has 2 parts
        assert isinstance(s3.parts[1], Sequential)  # Second part is Sequential

    def test_nested_flattening(self):
        """Test that nested Sequential specs are flattened."""
        inner = seq(MockSpec(2), MockSpec(3))
        outer = seq(MockSpec(1), inner, MockSpec(4))

        assert len(outer) == 4
        assert [p.value for p in outer] == [1, 2, 3, 4]

        # Triple nesting
        s = seq(seq(MockSpec(1)), seq(seq(MockSpec(2), MockSpec(3))))
        assert len(s) == 3
        assert [p.value for p in s] == [1, 2, 3]

    def test_indexing_and_slicing(self):
        """Test sequence indexing and slicing."""
        s = seq(MockSpec(1), MockSpec(2), MockSpec(3), MockSpec(4))

        # Indexing
        assert s[0].value == 1
        assert s[-1].value == 4
        assert s[2].value == 3

        # Slicing
        sub = s[1:3]
        assert isinstance(sub, Sequential)
        assert len(sub) == 2
        assert [p.value for p in sub] == [2, 3]

        # Step slicing
        sub2 = s[::2]
        assert len(sub2) == 2
        assert [p.value for p in sub2] == [1, 3]

    def test_iteration(self):
        """Test iteration over sequential."""
        s = seq(MockSpec(1), MockSpec(2), MockSpec(3))

        values = [spec.value for spec in s]
        assert values == [1, 2, 3]

        # Test that it's truly iterable
        assert list(s) == list(s)  # Can iterate multiple times

    def test_context_propagation(self):
        """Test context flows through sequence."""
        s = seq(
            MockProviderSpec(output_dim=256),
            MockRequirerSpec(),
            MockTransformSpec(),
        )

        ctx = Context()

        # Validate - should pass since provider comes before requirer
        issues = s.validate(ctx)
        assert len(issues) == 0

        # Apply context
        final_ctx = s.apply_context(ctx)
        assert final_ctx.get_dim("output_dim") == 256

    def test_validation_with_missing_requirements(self):
        """Test validation fails when requirements not met."""
        # Requirer comes before provider - should fail
        s = seq(MockRequirerSpec(), MockProviderSpec())

        issues = s.validate(Context())
        assert len(issues) > 0
        assert any("input_dim" in issue for issue in issues)

    def test_children_method(self):
        """Test children returns all parts."""
        specs = [MockSpec(i) for i in range(5)]
        s = seq(*specs)

        children = s.children()
        assert children == specs

    def test_parallel_operator(self):
        """Test | operator creates Parallel."""
        s = MockSpec(1) >> MockSpec(2)
        p = s | MockSpec(3)

        assert isinstance(p, Parallel)
        assert len(p.branches) == 2
        assert isinstance(p.branches[0], Sequential)
        assert isinstance(p.branches[1], MockSpec)


class TestParallel:
    """Test Parallel combinator."""

    def test_basic_parallel(self):
        """Test basic parallel creation."""
        p = parallel(MockSpec(1), MockSpec(2), MockSpec(3))

        assert isinstance(p, Parallel)
        assert len(p.branches) == 3
        assert p.merge == "concat"
        assert [b.value for b in p.branches] == [1, 2, 3]

    def test_empty_parallel_error(self):
        """Test that empty parallel raises error."""
        with pytest.raises(ValueError, match="at least one branch"):
            parallel()

    def test_merge_strategies(self):
        """Test different merge strategies."""
        for merge in ["concat", "add", "multiply", "mean", "max"]:
            p = parallel(MockSpec(1), MockSpec(2), merge=merge)
            assert p.merge == merge

    def test_merge_dimension_validation(self):
        """Test dimension validation for merges requiring same dims."""
        # These merges require matching dimensions
        for merge in ["add", "multiply", "mean", "max"]:
            p = parallel(
                MockProviderSpec(100),
                MockProviderSpec(200),
                merge=merge,
                merge_dim="output_dim",
            )

            issues = p.validate(Context())
            assert len(issues) > 0
            assert any("Incompatible dimensions" in issue for issue in issues)

        # Same dimensions should validate
        p2 = parallel(
            MockProviderSpec(100),
            MockProviderSpec(100),
            merge="add",
            merge_dim="output_dim",
        )
        issues = p2.validate(Context())
        # Should only have issues about missing dimensions, not incompatibility
        assert not any("Incompatible dimensions" in issue for issue in issues)

    def test_weighted_merge(self):
        """Test weighted merge."""
        p = parallel(
            MockSpec(1),
            MockSpec(2),
            MockSpec(3),
            merge="add",
            weights=[0.5, 0.3, 0.2],
        )

        assert p.weights == (0.5, 0.3, 0.2)

        # Validate weight count
        issues = p.validate(Context())
        assert not any("Weight count" in issue for issue in issues)

        # Wrong weight count
        p2 = parallel(
            MockSpec(1),
            MockSpec(2),
            merge="add",
            weights=[0.5, 0.3, 0.2],  # 3 weights but 2 branches
        )

        issues = p2.validate(Context())
        assert any("Weight count" in issue for issue in issues)

    def test_operator_chaining(self):
        """Test | operator for building parallel."""
        p1 = MockSpec(1) | MockSpec(2)
        assert isinstance(p1, Parallel)
        assert len(p1.branches) == 2

        # Chain more
        p2 = p1 | MockSpec(3) | MockSpec(4)
        assert len(p2.branches) == 4

        # Flatten nested parallels with same merge
        p3 = (MockSpec(1) | MockSpec(2)) | (MockSpec(3) | MockSpec(4))
        assert len(p3.branches) == 4

    def test_nested_parallel_no_flatten_different_merge(self):
        """Test that parallels with different merges don't flatten."""
        p1 = parallel(MockSpec(1), MockSpec(2), merge="add")
        p2 = parallel(p1, MockSpec(3), merge="concat")

        assert len(p2.branches) == 2
        assert isinstance(p2.branches[0], Parallel)

    def test_children_method(self):
        """Test children returns all branches."""
        branches = [MockSpec(i) for i in range(4)]
        p = parallel(*branches)

        assert p.children() == branches

    def test_validation_propagates_to_branches(self):
        """Test validation checks all branches."""
        p = parallel(
            MockRequirerSpec(),  # Requires input_dim
            MockProviderSpec(),  # Provides output_dim
            MockSpec(),  # No requirements
        )

        issues = p.validate(Context())
        assert len(issues) > 0
        assert any(
            "Branch 0" in issue and "input_dim" in issue for issue in issues
        )


class TestConditional:
    """Test Conditional combinator."""

    def test_basic_conditional(self):
        """Test basic conditional creation."""

        def condition(ctx):
            return ctx.get_dim("test") == 42

        c = cond(condition, MockSpec(1), MockSpec(2))

        assert isinstance(c, Conditional)
        assert c.if_true.value == 1
        assert c.if_false.value == 2

    def test_conditional_without_else(self):
        """Test conditional without else branch."""
        c = cond(lambda _ctx: False, MockSpec(1))

        assert c.if_false is None
        assert len(c.children()) == 1

    def test_string_condition(self):
        """Test string conditions check dimension existence."""
        c = cond("some_dim", MockSpec(1), MockSpec(2))

        # With dimension
        ctx_with = Context(dimensions={"some_dim": 100})
        assert c.condition(ctx_with)

        # Without dimension
        ctx_without = Context()
        assert not c.condition(ctx_without)

    def test_validation_follows_branches(self):
        """Test validation follows the active branch."""
        c = cond(
            lambda ctx: ctx.get_dim("branch") == "true",
            MockRequirerSpec(),  # Requires input_dim
            MockSpec(),  # No requirements
        )

        # True branch - should have validation issues
        ctx_true = Context(dimensions={"branch": "true"})
        issues = c.validate(ctx_true)
        assert any("input_dim" in issue for issue in issues)

        # False branch - should have no issues
        ctx_false = Context(dimensions={"branch": "false"})
        issues = c.validate(ctx_false)
        assert len(issues) == 0

    def test_context_application(self):
        """Test context application follows active branch."""
        c = cond(
            lambda ctx: ctx.get_dim("use_provider"),
            MockProviderSpec(output_dim=500),
            MockSpec(),
        )

        # True branch
        ctx_true = Context(dimensions={"use_provider": True})
        result = c.apply_context(ctx_true)
        assert result.get_dim("output_dim") == 500

        # False branch
        ctx_false = Context(dimensions={"use_provider": False})
        result = c.apply_context(ctx_false)
        assert result.get_dim("output_dim") is None

    def test_children_includes_both_branches(self):
        """Test children includes both branches when present."""
        c = cond(lambda _ctx: True, MockSpec(1), MockSpec(2))
        children = c.children()
        assert len(children) == 2
        assert children[0].value == 1
        assert children[1].value == 2


class TestResidual:
    """Test Residual combinator."""

    def test_basic_residual(self):
        """Test basic residual creation."""
        r = residual(MockSpec(42))

        assert isinstance(r, Residual)
        assert r.inner.value == 42
        assert r.merge == "add"
        assert r.scale == 1.0

    def test_residual_with_scale(self):
        """Test scaled residual."""
        r = residual(MockSpec(), scale=0.5)

        assert r.scale == 0.5

        # Negative scale should fail validation
        r2 = residual(MockSpec(), scale=-0.5)
        issues = r2.validate(Context())
        assert any("Scale must be positive" in issue for issue in issues)

    def test_residual_merge_strategies(self):
        """Test different merge strategies."""
        # Add merge (default)
        r1 = residual(MockSpec(), merge="add")
        assert r1.merge == "add"

        # Concat merge
        r2 = residual(MockSpec(), merge="concat")
        assert r2.merge == "concat"

        # Gate merge requires gate_dim
        r3 = residual(MockSpec(), merge="gate", gate_dim="hidden")
        assert r3.merge == "gate"
        assert r3.gate_dim == "hidden"

    def test_gate_merge_validation(self):
        """Test gate merge requires gate_dim."""
        r = residual(MockSpec(), merge="gate")

        issues = r.validate(Context())
        assert any("gate_dim" in issue for issue in issues)

    def test_children_method(self):
        """Test children returns inner spec."""
        inner = MockSpec(123)
        r = residual(inner)

        assert r.children() == [inner]

    def test_validation_propagates(self):
        """Test validation checks inner spec."""
        r = residual(MockRequirerSpec())

        issues = r.validate(Context())
        assert any("input_dim" in issue for issue in issues)


class TestGraph:
    """Test Graph combinator."""

    def test_empty_graph(self):
        """Test empty graph creation."""
        g = graph()

        assert isinstance(g, Graph)
        assert len(g.nodes) == 0
        assert len(g.edges) == 0
        assert len(g.inputs) == 0
        assert len(g.outputs) == 0

    def test_add_nodes(self):
        """Test adding nodes to graph."""
        g = graph()
        g = g.add_node("a", MockSpec(1))
        g = g.add_node("b", MockSpec(2))

        assert len(g.nodes) == 2
        assert g.nodes["a"].value == 1
        assert g.nodes["b"].value == 2

    def test_add_edges(self):
        """Test adding edges to graph."""
        g = graph()
        g = g.add_node("a", MockSpec())
        g = g.add_node("b", MockSpec())
        g = g.add_edge("a", "b")
        g = g.add_edge("a", "b", "transform")

        assert len(g.edges) == 2
        assert ("a", "b", None) in g.edges
        assert ("a", "b", "transform") in g.edges

    def test_graph_construction(self):
        """Test building a complete graph."""
        g = (
            graph()
            .add_node("input", MockProviderSpec())
            .add_node("process", MockTransformSpec())
            .add_node("output", MockSpec())
            .add_edge("input", "process")
            .add_edge("process", "output")
        )

        g = Graph(
            nodes=g.nodes,
            edges=g.edges,
            inputs=["input"],
            outputs=["output"],
        )

        assert len(g.nodes) == 3
        assert len(g.edges) == 2
        assert g.inputs == ["input"]
        assert g.outputs == ["output"]

    def test_cycle_detection(self):
        """Test that cycles are detected."""
        g = (
            graph()
            .add_node("a", MockSpec())
            .add_node("b", MockSpec())
            .add_node("c", MockSpec())
            .add_edge("a", "b")
            .add_edge("b", "c")
        )

        with pytest.raises(ValidationError):
            g.add_edge("c", "a")  # Creates cycle

    def test_unknown_node_validation(self):
        """Test validation catches unknown nodes."""
        g = graph().add_edge("unknown1", "unknown2")

        issues = g.validate(Context())
        assert any("Unknown source node: unknown1" in issue for issue in issues)
        assert any("Unknown target node: unknown2" in issue for issue in issues)

    def test_node_validation_propagates(self):
        """Test that node validation is checked."""
        g = (
            graph()
            .add_node("requirer", MockRequirerSpec())
            .add_node("provider", MockProviderSpec())
        )

        issues = g.validate(Context())
        assert any(
            "Node requirer" in issue and "input_dim" in issue
            for issue in issues
        )

    def test_children_method(self):
        """Test children returns all node specs."""
        specs = [MockSpec(i) for i in range(3)]
        g = (
            graph()
            .add_node("a", specs[0])
            .add_node("b", specs[1])
            .add_node("c", specs[2])
        )

        children = g.children()
        assert set(children) == set(specs)


class TestLoop:
    """Test Loop combinator."""

    def test_static_loop(self):
        """Test loop with static count."""
        ell = loop(MockSpec(42), times=5)

        assert isinstance(ell, Loop)
        assert ell.body.value == 42
        assert ell.times == 5
        assert not ell.unroll
        assert ell.share_weights

    def test_dynamic_loop(self):
        """Test loop with dynamic count from context."""
        ell = loop(MockSpec(), times="num_iterations")

        # With dimension
        ctx = Context(dimensions={"num_iterations": 10})
        issues = ell.validate(ctx)
        assert len(issues) == 0

        # Without dimension
        ctx2 = Context()
        issues2 = ell.validate(ctx2)
        assert any("num_iterations" in issue for issue in issues2)

    def test_loop_with_options(self):
        """Test loop with unroll and share_weights options."""
        l1 = loop(MockSpec(), times=3, unroll=True, share_weights=False)

        assert l1.unroll
        assert not l1.share_weights

    def test_unrolled_children(self):
        """Test that unrolled loop repeats body in children."""
        ell = loop(MockSpec(123), times=4, unroll=True)

        children = ell.children()
        assert len(children) == 4
        assert all(c.value == 123 for c in children)

    def test_non_unrolled_children(self):
        """Test that non-unrolled loop has single child."""
        ell = loop(MockSpec(123), times=4, unroll=False)

        children = ell.children()
        assert len(children) == 1
        assert children[0].value == 123

    def test_validation_with_invalid_times(self):
        """Test validation with invalid loop count."""
        ell = loop(MockSpec(), times=0)

        issues = ell.validate(Context())
        assert any("must be positive" in issue for issue in issues)

    def test_body_validation_propagates(self):
        """Test that body validation is checked."""
        ell = loop(MockRequirerSpec(), times=3)

        issues = ell.validate(Context())
        assert any("input_dim" in issue for issue in issues)


class TestSwitch:
    """Test Switch combinator."""

    def test_basic_switch(self):
        """Test basic switch creation."""
        s = switch(
            "mode",
            {"fast": MockSpec(1), "slow": MockSpec(2), "medium": MockSpec(3)},
            default=MockSpec(0),
        )

        assert isinstance(s, Switch)
        assert len(s.cases) == 3
        assert s.default.value == 0

    def test_switch_without_default(self):
        """Test switch without default case."""
        s = switch("mode", {"a": MockSpec(1), "b": MockSpec(2)})

        assert s.default is None

    def test_callable_key(self):
        """Test switch with callable key function."""

        def compute_key(ctx):
            size = ctx.get_dim("size") or 0
            if size < 10:
                return "small"
            if size < 100:
                return "medium"
            return "large"

        s = switch(
            compute_key,
            {"small": MockSpec(1), "medium": MockSpec(2), "large": MockSpec(3)},
        )

        assert callable(s.key)

        # Test the key function
        assert s.key(Context(dimensions={"size": 5})) == "small"
        assert s.key(Context(dimensions={"size": 50})) == "medium"
        assert s.key(Context(dimensions={"size": 500})) == "large"

    def test_validation_checks_all_cases(self):
        """Test that validation checks all cases."""
        s = switch(
            "mode",
            {"a": MockRequirerSpec(), "b": MockProviderSpec(), "c": MockSpec()},
            default=MockTransformSpec(),
        )

        issues = s.validate(Context())

        # Should have issues for case 'a' requiring input_dim
        assert any(
            "Case a" in issue and "input_dim" in issue for issue in issues
        )

        # Should not have issues for other cases
        assert not any("Case b" in issue for issue in issues)
        assert not any("Case c" in issue for issue in issues)

    def test_children_includes_all_cases(self):
        """Test children includes all cases and default."""
        cases = {"a": MockSpec(1), "b": MockSpec(2), "c": MockSpec(3)}
        default = MockSpec(0)

        s = switch("key", cases, default)

        children = s.children()
        assert len(children) == 4
        assert set(children) == {*list(cases.values()), default}


class TestIdentityAndLambda:
    """Test Identity and Lambda specs."""

    def test_identity(self):
        """Test Identity spec."""
        i = Identity()

        assert i.children() == []

        # Context passes through unchanged
        ctx = Context(dimensions={"test": 42})
        result = i.apply_context(ctx)
        assert result.dimensions == ctx.dimensions

    def test_lambda(self):
        """Test Lambda spec."""

        def my_transform(x, ctx):
            return x * ctx.get_dim("scale", 1)

        ell = Lambda(fn=my_transform, name="scaler")

        assert ell.fn is my_transform
        assert ell.name == "scaler"
        assert ell.children() == []


class TestHighLevelFactories:
    """Test high-level factory functions."""

    def test_transformer_block(self):
        """Test transformer block pattern."""
        # Create with mock specs
        block = transformer_block(
            norm_first=True,
            attention=MockSpec(1),
            mlp=MockSpec(2),
            drop_path=0.1,
        )

        assert isinstance(block, Sequential)
        # Should contain residual blocks

        # Post-norm version
        block2 = transformer_block(
            norm_first=False,
            attention=MockSpec(1),
            mlp=MockSpec(2),
        )

        assert isinstance(block2, Sequential)

    def test_multi_scale(self):
        """Test multi-scale pattern."""

        def make_scale_spec(scale):
            return MockProviderSpec(output_dim=scale)

        ms = multi_scale(make_scale_spec, scales=[64, 128, 256], merge="concat")

        assert isinstance(ms, Parallel)
        assert len(ms.branches) == 3
        assert ms.merge == "concat"
        assert ms.branches[0].output_dim == 64
        assert ms.branches[1].output_dim == 128
        assert ms.branches[2].output_dim == 256

    def test_mixture_of_experts(self):
        """Test mixture of experts pattern."""
        experts = [MockSpec(i) for i in range(4)]
        router = MockTransformSpec(transform="route")

        moe = mixture_of_experts(experts, router, top_k=2)

        assert isinstance(moe, Graph)
        assert "router" in moe.nodes
        assert all(f"expert_{i}" in moe.nodes for i in range(4))
        assert "combine" in moe.nodes
        assert moe.inputs == ["input"]
        assert moe.outputs == ["combine"]


class TestComplexCompositions:
    """Test complex compositions of combinators."""

    def test_nested_composition(self):
        """Test deeply nested compositions."""
        # Sequential of parallels
        model = seq(
            MockProviderSpec(output_dim=128),
            parallel(
                seq(MockTransformSpec("a"), MockTransformSpec("b")),
                seq(MockTransformSpec("c"), MockTransformSpec("d")),
                merge="add",
            ),
            MockSpec(),
        )

        assert isinstance(model, Sequential)
        assert len(model) == 3
        assert isinstance(model[1], Parallel)
        assert len(model[1].branches) == 2

    def test_conditional_in_sequence(self):
        """Test conditional embedded in sequence."""
        model = seq(
            MockProviderSpec(),
            cond(
                "use_transform",
                MockTransformSpec("complex"),
                MockTransformSpec("simple"),
            ),
            MockSpec(),
        )

        # Validate with condition true
        ctx = Context(dimensions={"use_transform": True})
        issues = model.validate(ctx)
        assert len(issues) == 0

    def test_loop_of_residuals(self):
        """Test loop containing residual blocks."""
        model = loop(
            residual(
                seq(MockTransformSpec("attention"), MockTransformSpec("mlp")),
            ),
            times=6,
        )

        assert isinstance(model, Loop)
        assert isinstance(model.body, Residual)
        assert isinstance(model.body.inner, Sequential)

    def test_graph_with_conditionals(self):
        """Test graph containing conditional nodes."""
        g = (
            graph()
            .add_node("input", MockProviderSpec())
            .add_node(
                "conditional",
                cond("fast_path", MockSpec(1), MockSpec(2)),
            )
            .add_node("output", MockSpec())
            .add_edge("input", "conditional")
            .add_edge("conditional", "output")
        )

        g = Graph(
            nodes=g.nodes,
            edges=g.edges,
            inputs=["input"],
            outputs=["output"],
        )

        assert "conditional" in g.nodes
        assert isinstance(g.nodes["conditional"], Conditional)

    def test_switch_with_complex_cases(self):
        """Test switch with complex case specifications."""
        s = switch(
            "architecture",
            {
                "simple": MockSpec(),
                "residual": residual(MockTransformSpec()),
                "parallel": parallel(MockSpec(1), MockSpec(2)),
                "sequential": seq(MockSpec(1), MockSpec(2)),
            },
        )

        assert isinstance(s.cases["residual"], Residual)
        assert isinstance(s.cases["parallel"], Parallel)
        assert isinstance(s.cases["sequential"], Sequential)


class TestTreeOperations:
    """Test tree traversal operations on specs."""

    def test_map_operation(self):
        """Test map over spec tree."""
        model = seq(
            MockSpec(1),
            parallel(MockSpec(2), MockSpec(3)),
            residual(MockSpec(4)),
        )

        # Extract all values
        values = model.map(
            lambda s: s.value if isinstance(s, MockSpec) else None,
        )
        mock_values = [v for v in values if v is not None]

        assert sorted(mock_values) == [1, 2, 3, 4]

    def test_find_operation(self):
        """Test find specs matching predicate."""
        model = seq(
            MockSpec(10),
            parallel(MockSpec(20), MockSpec(30)),
            residual(MockSpec(40)),
        )

        # Find all MockSpecs with value > 25
        found = model.find(lambda s: isinstance(s, MockSpec) and s.value > 25)

        assert len(found) == 2
        assert all(isinstance(s, MockSpec) for s in found)
        assert sorted(s.value for s in found) == [30, 40]

    def test_to_dict_and_from_dict(self):
        """Test serialization to dict."""
        model = seq(
            MockSpec(42),
            parallel(MockSpec(1), MockSpec(2)),
            cond("test", MockSpec(3)),
        )

        # To dict
        d = model.to_dict()
        assert d["_type"] == "Sequential"
        assert len(d["parts"]) == 3
        assert d["parts"][0]["value"] == 42
        assert d["parts"][1]["_type"] == "Parallel"

        # From dict
        model2 = Sequential.from_dict(d)
        assert isinstance(model2, Sequential)
        assert len(model2) == 3
        assert model2[0].value == 42


class TestValidationIntegration:
    """Test validation across complex specs."""

    def test_dimension_flow(self):
        """Test dimensions flow correctly through combinators."""
        model = seq(
            MockProviderSpec(output_dim=256),
            parallel(
                MockRequirerSpec(),  # Will see output_dim as input_dim? No
                MockSpec(),
                merge="concat",
            ),
            cond(
                "output_dim",
                MockTransformSpec("has_dim"),
                MockTransformSpec("no_dim"),
            ),
        )

        ctx = Context()
        ctx.set_dim("input_dim", 128)  # For requirer

        issues = model.validate(ctx)
        assert len(issues) == 0

        # Check final context
        final_ctx = model.apply_context(ctx)
        assert final_ctx.get_dim("output_dim") == 256

    def test_validation_error_messages(self):
        """Test validation produces helpful error messages."""
        model = seq(
            parallel(MockRequirerSpec(), MockRequirerSpec(), merge="add"),
            loop(MockRequirerSpec(), times="unknown_dim"),
        )

        issues = model.validate(Context())

        # Should have issues for:
        # - Missing input_dim in parallel branches
        # - Unknown loop dimension
        assert len(issues) >= 3
        assert any(
            "Part 0" in issue for issue in issues
        )  # Sequential part indexing
        assert any(
            "Branch 0" in issue for issue in issues
        )  # Parallel branch indexing
        assert any("unknown_dim" in issue for issue in issues)  # Loop dimension
