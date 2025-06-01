"""Test various spec composition patterns."""

import pytest
import torch

from energy_transformer.spec import (
    Context,
    graph,
    loop,
    realise,
    residual,
    seq,
    switch,
)
from energy_transformer.spec.library import (
    CLSTokenSpec,
    ETBlockSpec,
    HNSpec,
    LayerNormSpec,
    MHEASpec,
    PatchEmbedSpec,
)

pytestmark = pytest.mark.integration


class TestCompositionPatterns:
    """Test complex composition patterns."""

    def test_mixed_depth_composition(self):
        """Test mixing different depth blocks."""
        spec = seq(
            loop(ETBlockSpec(steps=2, alpha=1.0), times=2),
            loop(ETBlockSpec(steps=10, alpha=0.1), times=4),
            ETBlockSpec(steps=3, alpha=0.5),
        )
        ctx = Context(dimensions={"embed_dim": 256})
        module = realise(spec, ctx)
        et_blocks = [
            m
            for m in module.modules()
            if m.__class__.__name__ == "EnergyTransformer"
        ]
        assert len(et_blocks) == 7

    def test_residual_patterns(self):
        """Test residual connection patterns."""
        spec = seq(
            residual(
                seq(LayerNormSpec(), MHEASpec(num_heads=8, head_dim=64)),
                scale=0.5,
            ),
            residual(HNSpec(multiplier=4.0), scale=1.0),
        )
        ctx = Context(dimensions={"embed_dim": 512})
        module = realise(spec, ctx)
        x = torch.randn(2, 10, 512)
        out = module(x)
        assert out.shape == x.shape

    def test_switch_pattern(self):
        """Test switch-based architecture selection."""
        spec = switch(
            key="model_size",
            cases={
                "tiny": ETBlockSpec(steps=2, alpha=1.0),
                "small": ETBlockSpec(steps=4, alpha=0.5),
                "base": ETBlockSpec(steps=6, alpha=0.125),
            },
            default=ETBlockSpec(steps=4, alpha=0.25),
        )
        for size, expected_steps in [
            ("tiny", 2),
            ("small", 4),
            ("base", 6),
            ("large", 4),
        ]:
            ctx = Context(
                dimensions={"embed_dim": 256}, metadata={"model_size": size}
            )
            module = realise(spec, ctx)
            assert module.steps == expected_steps

    def test_graph_pattern(self):
        """Test graph-based composition."""
        g = graph()
        g = g.add_node(
            "embed", PatchEmbedSpec(img_size=32, patch_size=4, embed_dim=192)
        )
        g = g.add_node("cls", CLSTokenSpec())
        g = g.add_node("et1", ETBlockSpec(steps=4))
        g = g.add_node("et2", ETBlockSpec(steps=4))
        g = g.add_node("norm", LayerNormSpec())
        g = g.add_edge("input", "embed")
        g = g.add_edge("embed", "cls")
        g = g.add_edge("cls", "et1")
        g = g.add_edge("et1", "et2")
        g = g.add_edge("et2", "norm")
        g = g.add_edge("norm", "output")
        g.inputs = ["input"]
        g.outputs = ["output"]
        ctx = Context()
        issues = g.validate(ctx)
        assert len(issues) > 0

    def test_dynamic_architecture(self):
        """Test dynamic architecture based on input size."""

        def build_model(img_size: int):
            if img_size <= 32:
                depth = 2
                patch_size = 4
            elif img_size <= 224:
                depth = 12
                patch_size = 16
            else:
                depth = 24
                patch_size = 32
            return seq(
                PatchEmbedSpec(
                    img_size=img_size, patch_size=patch_size, embed_dim=768
                ),
                CLSTokenSpec(),
                loop(ETBlockSpec(), times=depth),
                LayerNormSpec(),
            )

        for img_size in [32, 224, 384]:
            spec = build_model(img_size)
            model = realise(spec)
            x = torch.randn(1, 3, img_size, img_size)
            out = model(x)
            assert out.dim() == 3
