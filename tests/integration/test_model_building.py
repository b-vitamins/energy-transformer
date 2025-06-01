"""Integration tests for complete workflows."""

import pytest
import torch
from torch import nn

from energy_transformer import realise, seq
from energy_transformer.spec import Context

pytestmark = pytest.mark.integration


class TestCompleteWorkflows:
    """Test complete model building workflows."""

    def test_vision_transformer_workflow(self, simple_image_batch):
        """Test building a complete vision transformer."""
        from energy_transformer.spec.library import (
            ClassificationHeadSpec,
            CLSTokenSpec,
            ETBlockSpec,
            LayerNormSpec,
            PatchEmbedSpec,
            PosEmbedSpec,
        )

        vit_spec = seq(
            PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            ETBlockSpec(),
            LayerNormSpec(),
            ClassificationHeadSpec(num_classes=1000),
        )

        ctx = Context()
        issues = vit_spec.validate(ctx)
        assert len(issues) == 0, f"Validation failed: {issues}"

        model = realise(vit_spec)
        assert isinstance(model, nn.Module)

        output = model(simple_image_batch)
        assert output.shape == (4, 1000)

    def test_custom_model_workflow(self):
        """Test building a custom model with mixed components."""
        from energy_transformer.spec import loop, parallel
        from energy_transformer.spec.library import (
            HNSpec,
            LayerNormSpec,
            MHEASpec,
        )

        custom_block = seq(
            LayerNormSpec(),
            parallel(
                MHEASpec(num_heads=8, head_dim=64),
                HNSpec(multiplier=2),
                merge="add",
            ),
        )

        model_spec = loop(custom_block, times=3)

        model = realise(model_spec, embed_dim=512)

        x = torch.randn(2, 10, 512)
        output = model(x)
        assert output.shape == x.shape

    @pytest.mark.slow
    def test_deep_model_workflow(self):
        """Test building very deep models."""
        from energy_transformer.spec.library import ETBlockSpec

        deep_spec = seq(*[ETBlockSpec() for _ in range(24)])

        model = realise(deep_spec, embed_dim=768)

        num_blocks = len(
            [m for m in model.modules() if "ETBlock" in str(type(m))],
        )
        assert num_blocks >= 24
