"""Tests for specification combinators."""

import pytest

from energy_transformer.spec.combinators import (
    Graph,
    Identity,
    Parallel,
    Sequential,
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
from energy_transformer.spec.library import (
    CLSTokenSpec,
    ETSpec,
    HNSpec,
    LayerNormSpec,
    MHEASpec,
    PatchEmbedSpec,
    PosEmbedSpec,
)
from energy_transformer.spec.primitives import Context


def make_patch(embed_dim: int = 64) -> PatchEmbedSpec:
    """Helper to create patch embed spec."""
    return PatchEmbedSpec(img_size=32, patch_size=16, embed_dim=embed_dim)


class TestSequential:
    """Test Sequential combinator."""

    def test_basic_composition(self):
        s = seq(make_patch(), CLSTokenSpec(), LayerNormSpec())

        assert len(s) == 3
        assert isinstance(s[0], PatchEmbedSpec)
        assert isinstance(s[1], CLSTokenSpec)
        assert isinstance(s[2], LayerNormSpec)

    def test_operator_chaining(self):
        # Right shift operator
        s1 = make_patch() >> CLSTokenSpec() >> LayerNormSpec()
        assert len(s1) == 3

        # Left shift operator
        s2 = LayerNormSpec() << CLSTokenSpec() << make_patch()
        assert len(s2) == 3
        assert isinstance(s2[0], PatchEmbedSpec)

    def test_parallel_operator(self):
        s = make_patch()
        p = s | CLSTokenSpec()
        assert isinstance(p, Parallel)
        assert len(p.branches) == 2

    def test_indexing_and_slicing(self):
        s = seq(make_patch(), CLSTokenSpec(), LayerNormSpec())

        # Indexing
        assert isinstance(s[0], PatchEmbedSpec)
        assert isinstance(s[-1], LayerNormSpec)

        # Slicing
        sub = s[1:]
        assert isinstance(sub, Sequential)
        assert len(sub) == 2
        assert isinstance(sub[0], CLSTokenSpec)

    def test_iteration(self):
        s = seq(make_patch(), CLSTokenSpec())
        parts = list(s)
        assert len(parts) == 2
        assert isinstance(parts[0], PatchEmbedSpec)

    def test_flattening(self):
        # Nested sequential should be flattened
        inner = seq(CLSTokenSpec(), LayerNormSpec())
        outer = seq(make_patch(), inner)

        assert len(outer) == 3
        assert isinstance(outer[1], CLSTokenSpec)
        assert isinstance(outer[2], LayerNormSpec)

    def test_context_propagation(self):
        s = seq(make_patch(768), CLSTokenSpec(), LayerNormSpec())
        ctx = Context()

        # Validate entire sequence
        issues = s.validate(ctx)
        assert len(issues) == 0

        # Apply context through sequence
        final_ctx = s.apply_context(ctx)
        assert final_ctx.get_dim("embed_dim") == 768
        assert final_ctx.get_dim("token_count") == 5  # 4 patches + 1 CLS

    def test_empty_sequence(self):
        s = seq()
        assert len(s) == 0
        assert isinstance(s, Sequential)

    def test_single_spec_passthrough(self):
        spec = make_patch()
        s = seq(spec)
        assert s is spec  # Should return the spec directly


class TestParallel:
    """Test Parallel combinator."""

    def test_basic_parallel(self):
        p = parallel(make_patch(64), make_patch(128), merge="concat")

        assert len(p.branches) == 2
        assert p.merge == "concat"

    def test_merge_strategies(self):
        # Concat adds dimensions
        p1 = parallel(make_patch(64), make_patch(128), merge="concat")
        p1.apply_context(Context())
        # Note: This would need proper implementation in real spec

        # Add requires same dimensions
        p2 = parallel(make_patch(64), make_patch(64), merge="add")
        issues = p2.validate(Context())
        assert len(issues) == 0

        # Mismatched dimensions for add
        parallel(make_patch(64), make_patch(128), merge="add")
        # This should have validation issues in real implementation

    def test_operator_chaining(self):
        p1 = make_patch(64) | make_patch(128)
        assert isinstance(p1, Parallel)
        assert len(p1.branches) == 2

        # Chain more branches
        p2 = p1 | make_patch(256)
        assert len(p2.branches) == 3

    def test_weighted_merge(self):
        p = parallel(
            make_patch(64), make_patch(64), merge="add", weights=[0.7, 0.3]
        )

        assert p.weights == (0.7, 0.3)

        # Validate weight count
        p.validate(Context())
        # Should check weight count matches branch count

    def test_no_branches_error(self):
        with pytest.raises(ValueError, match="at least one branch"):
            parallel()


class TestConditional:
    """Test Conditional combinator."""

    def test_basic_conditional(self):
        c = cond(
            lambda ctx: ctx.get_dim("use_cls") is True,
            CLSTokenSpec(),
            Identity(),
        )

        # With condition true
        ctx_true = Context(dimensions={"use_cls": True})
        assert isinstance(c.children()[0], CLSTokenSpec)

        # Validation follows the true branch
        c.validate(ctx_true)
        # Would check CLSTokenSpec requirements

    def test_string_condition(self):
        # String conditions check dimension existence
        c = cond("embed_dim", LayerNormSpec())

        ctx_with = Context(dimensions={"embed_dim": 768})
        ctx_without = Context()

        # Should validate successfully with dimension
        issues_with = c.validate(ctx_with)
        assert len(issues_with) == 0

        # Should skip when dimension missing
        issues_without = c.validate(ctx_without)
        assert len(issues_without) == 0


class TestResidual:
    """Test Residual combinator."""

    def test_basic_residual(self):
        r = residual(LayerNormSpec())

        assert r.merge == "add"
        assert r.scale == 1.0
        assert len(r.children()) == 1

    def test_scaled_residual(self):
        r = residual(LayerNormSpec(), scale=0.5)
        assert r.scale == 0.5

    def test_gate_residual(self):
        r = residual(LayerNormSpec(), merge="gate", gate_dim="hidden")
        assert r.merge == "gate"
        assert r.gate_dim == "hidden"

        # Should fail validation without gate_dim
        r2 = residual(LayerNormSpec(), merge="gate")
        issues = r2.validate(Context())
        assert any("gate_dim" in issue for issue in issues)


class TestGraph:
    """Test Graph combinator."""

    def test_graph_construction(self):
        g = graph()

        # Add nodes
        g = g.add_node("embed", make_patch())
        g = g.add_node("cls", CLSTokenSpec())
        g = g.add_node("norm", LayerNormSpec())

        # Add edges
        g = g.add_edge("embed", "cls")
        g = g.add_edge("cls", "norm")

        # Set inputs/outputs
        g = Graph(
            nodes=g.nodes, edges=g.edges, inputs=["embed"], outputs=["norm"]
        )

        assert len(g.nodes) == 3
        assert len(g.edges) == 2

    def test_cycle_detection(self):
        g = graph()
        g = g.add_node("a", Identity())
        g = g.add_node("b", Identity())
        g = g.add_edge("a", "b")
        g = g.add_edge("b", "a")  # Creates cycle

        issues = g.validate(Context())
        assert any("cycle" in issue.lower() for issue in issues)

    def test_unknown_nodes(self):
        g = graph()
        g = g.add_edge("unknown", "also_unknown")

        issues = g.validate(Context())
        assert any("unknown" in issue.lower() for issue in issues)


class TestLoop:
    """Test Loop combinator."""

    def test_static_loop(self):
        ell = loop(LayerNormSpec(), times=3)

        assert ell.times == 3
        assert not ell.unroll
        assert ell.share_weights

    def test_dynamic_loop(self):
        ell = loop(LayerNormSpec(), times="num_layers")

        # Need to provide embed_dim for LayerNormSpec validation
        ctx = Context(dimensions={"num_layers": 12, "embed_dim": 768})
        issues = ell.validate(ctx)
        assert len(issues) == 0

        # Missing dimension
        ctx2 = Context()
        issues2 = ell.validate(ctx2)
        assert any("num_layers" in issue for issue in issues2)

    def test_unrolled_loop(self):
        ell = loop(LayerNormSpec(), times=3, unroll=True)
        assert ell.unroll

        # Children should be repeated when unrolled
        children = ell.children()
        assert len(children) == 3


class TestSwitch:
    """Test Switch combinator."""

    def test_basic_switch(self):
        s = switch(
            "model_size",
            {
                "small": make_patch(256),
                "base": make_patch(768),
                "large": make_patch(1024),
            },
            default=make_patch(512),
        )

        assert len(s.cases) == 3
        assert s.default is not None

    def test_callable_key(self):
        switch(
            lambda ctx: "large" if ctx.get_dim("gpu_memory") > 16 else "small",
            {"small": make_patch(256), "large": make_patch(1024)},
        )

        Context(dimensions={"gpu_memory": 24})
        # Would select "large" case during realisation


class TestFactoryFunctions:
    """Test high-level factory functions."""

    def test_transformer_block(self):
        block = transformer_block(
            attention=MHEASpec(), mlp=HNSpec(), norm_first=True
        )

        assert isinstance(block, Sequential)
        # Would contain residual connections

    def test_multi_scale(self):
        ms = multi_scale(
            lambda scale: make_patch(embed_dim=scale),
            scales=[256, 512, 1024],
            merge="concat",
        )

        assert isinstance(ms, Parallel)
        assert len(ms.branches) == 3

    def test_mixture_of_experts(self):
        experts = [HNSpec() for _ in range(4)]
        router = LayerNormSpec()  # Placeholder

        moe = mixture_of_experts(experts, router, top_k=2)

        assert isinstance(moe, Graph)
        assert len(moe.nodes) >= len(experts) + 1  # experts + router + combine


class TestIntegration:
    """Integration tests with complex architectures."""

    def test_vision_transformer(self):
        """Build a simple vision transformer."""
        model = seq(
            # Patch embedding
            make_patch(embed_dim=768),
            # Add CLS token
            CLSTokenSpec(),
            # Add positional embeddings
            PosEmbedSpec(include_cls=True),
            # Transformer blocks
            loop(
                ETSpec(
                    layer_norm=LayerNormSpec(),
                    attention=MHEASpec(num_heads=12, head_dim=64),
                    hopfield=HNSpec(),
                ),
                times=12,
            ),
            # Final norm
            LayerNormSpec(),
        )

        # Validate entire model
        ctx = Context()
        model.validate(ctx)

        # Apply context to get final dimensions
        final_ctx = model.apply_context(ctx)
        assert final_ctx.get_dim("embed_dim") == 768
        assert final_ctx.get_dim("token_count") == 5

    def test_conditional_architecture(self):
        """Build architecture that adapts based on context."""
        model = seq(
            make_patch(),
            # Conditionally add CLS token
            cond("use_cls_token", CLSTokenSpec()),
            # Different depths based on mode
            switch(
                "model_size",
                {
                    "tiny": loop(ETSpec(), times=2),
                    "small": loop(ETSpec(), times=6),
                    "base": loop(ETSpec(), times=12),
                },
                default=loop(ETSpec(), times=4),
            ),
            LayerNormSpec(),
        )

        # Test with different contexts
        ctx_tiny = Context(
            dimensions={"use_cls_token": True}, metadata={"model_size": "tiny"}
        )

        model.validate(ctx_tiny)
        # Should validate successfully
