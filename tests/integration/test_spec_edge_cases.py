"""Edge case tests for spec system."""

import pytest
import torch

from energy_transformer.spec import Context, parallel, realise, seq
from energy_transformer.spec.library import (
    ETBlockSpec,
    HNSpec,
    LayerNormSpec,
    MHEASpec,
    PatchEmbedSpec,
)
from energy_transformer.spec.primitives import ValidationError

pytest.skip("Edge case spec tests not implemented", allow_module_level=True)

pytestmark = pytest.mark.integration


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_sequences(self):
        """Test empty sequential specs."""
        empty_seq = seq()
        module = realise(empty_seq)
        assert isinstance(module, torch.nn.Sequential)
        assert len(module) == 0

    def test_single_element_sequences(self):
        """Test single element sequences."""
        single = seq(LayerNormSpec())
        ctx = Context(dimensions={"embed_dim": 768})
        module = realise(single, ctx)
        assert not isinstance(module, torch.nn.Sequential)

    def test_deeply_nested_specs(self):
        """Test deeply nested specifications."""
        deep = seq(seq(seq(seq(LayerNormSpec()))))
        ctx = Context(dimensions={"embed_dim": 768})
        module = realise(deep, ctx)
        assert not isinstance(module, torch.nn.Sequential)

    def test_zero_dimension_handling(self):
        """Test handling of zero dimensions."""
        with pytest.raises(ValidationError):
            realise(PatchEmbedSpec(img_size=16, patch_size=32, embed_dim=768))

    def test_extreme_dimensions(self):
        """Test extremely large/small dimensions."""
        spec = ETBlockSpec(
            steps=1,
            alpha=0.01,
            attention=MHEASpec(num_heads=1, head_dim=1),
            hopfield=HNSpec(hidden_dim=1),
        )
        ctx = Context(dimensions={"embed_dim": 1})
        module = realise(spec, ctx)
        assert module.steps == 1

    def test_dimension_overrides(self):
        """Test context dimension overrides."""
        spec = MHEASpec(num_heads=8, head_dim=64)
        ctx1 = Context(dimensions={"embed_dim": 512})
        ctx2 = ctx1.child(embed_dim=768)
        ctx3 = ctx2.child(embed_dim=1024)
        module1 = realise(spec, ctx1)
        module2 = realise(spec, ctx2)
        module3 = realise(spec, ctx3)
        assert module1.in_dim == 512
        assert module2.in_dim == 768
        assert module3.in_dim == 1024

    def test_conditional_specs(self):
        """Test conditional specification behavior."""
        from energy_transformer.spec import cond

        spec = cond(
            lambda ctx: ctx.get_dim("embed_dim", 0) > 512,
            if_true=MHEASpec(num_heads=16, head_dim=64),
            if_false=MHEASpec(num_heads=8, head_dim=64),
        )
        ctx_small = Context(dimensions={"embed_dim": 384})
        ctx_large = Context(dimensions={"embed_dim": 768})
        module_small = realise(spec, ctx_small)
        module_large = realise(spec, ctx_large)
        assert module_small.num_heads == 8
        assert module_large.num_heads == 16

    def test_parallel_composition(self):
        """Test parallel spec composition."""
        spec = parallel(
            MHEASpec(num_heads=8, head_dim=64),
            HNSpec(multiplier=4.0),
            merge="add",
        )
        ctx = Context(dimensions={"embed_dim": 512})
        module = realise(spec, ctx)
        x = torch.randn(2, 10, 512)
        out = module(x)
        assert out.shape == x.shape
