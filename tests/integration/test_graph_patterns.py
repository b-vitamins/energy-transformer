"""Test graph-based model construction."""

import pytest
import torch

from energy_transformer.spec import Context, graph, realise
from energy_transformer.spec.library import (
    CLSTokenSpec,
    ETBlockSpec,
    IdentitySpec,
    LayerNormSpec,
    PatchEmbedSpec,
    PosEmbedSpec,
)

pytestmark = pytest.mark.integration


class TestGraphPatterns:
    """Test graph-based architectural patterns."""

    def test_simple_vision_graph(self):
        """Test building a simple vision model with graph."""
        g = graph()

        g = g.add_node(
            "patch", PatchEmbedSpec(img_size=32, patch_size=4, embed_dim=192)
        )
        g = g.add_node("cls", CLSTokenSpec())
        g = g.add_node("pos", PosEmbedSpec(include_cls=True))
        g = g.add_node("et", ETBlockSpec(steps=4, alpha=0.1))
        g = g.add_node("norm", LayerNormSpec())

        g = g.add_edge("input", "patch")
        g = g.add_edge("patch", "cls")
        g = g.add_edge("cls", "pos")
        g = g.add_edge("pos", "et")
        g = g.add_edge("et", "norm")

        g.inputs = ["input"]
        g.outputs = ["norm"]

        ctx = Context(dimensions={"embed_dim": 192, "num_patches": 65})
        model = realise(g, ctx)

        x = torch.randn(2, 3, 32, 32)
        out = model(x)

        assert out.shape == (2, 65, 192)

    def test_multi_path_graph(self):
        """Test graph with multiple paths."""
        g = graph()

        g = g.add_node("split", IdentitySpec())
        g = g.add_node("attn1", IdentitySpec())
        g = g.add_node("attn2", IdentitySpec())
        g = g.add_node("merge", IdentitySpec())

        g = g.add_edge("input", "split")
        g = g.add_edge("split", "attn1")
        g = g.add_edge("split", "attn2")
        g = g.add_edge("attn1", "merge")
        g = g.add_edge("attn2", "merge")

        g.inputs = ["input"]
        g.outputs = ["merge"]

        ctx = Context(dimensions={"embed_dim": 128})
        model = realise(g, ctx)

        x = torch.randn(2, 10, 128)
        out = model(x)
        assert out.shape == (2, 10, 256)
