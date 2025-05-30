"""Integration tests for complete workflows."""

import pytest
import torch
import torch.nn as nn

from energy_transformer import seq, realise, ValidationError
from energy_transformer.spec import Context


class TestCompleteWorkflows:
    """Test complete model building workflows."""

    def test_vision_transformer_workflow(self, simple_image_batch):
        """Test building a complete vision transformer."""
        from energy_transformer.spec.library import (
            PatchEmbedSpec,
            CLSTokenSpec,
            PosEmbedSpec,
            ETBlockSpec,
            LayerNormSpec,
            ClassificationHeadSpec,
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
            LayerNormSpec,
            MHEASpec,
            HNSpec,
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

        num_blocks = len([m for m in model.modules() if "ETBlock" in str(type(m))])
        assert num_blocks >= 24


class TestErrorHandling:
    """Test error handling in complete workflows."""

    def test_missing_dimension_error(self):
        """Test helpful error when required dimension is missing."""
        from energy_transformer.spec.library import LayerNormSpec

        spec = LayerNormSpec()

        ctx = Context()
        issues = spec.validate(ctx)
        assert len(issues) > 0
        assert any("embed_dim" in issue for issue in issues)

    def test_incompatible_dimensions_error(self):
        """Test error when dimensions don't match."""
        from energy_transformer.spec import parallel
        from energy_transformer.spec.library import MLPSpec

        spec = parallel(
            MLPSpec(out_features=256),
            MLPSpec(out_features=512),
            merge="add",
        )

        ctx = Context(dimensions={"embed_dim": 128})
        issues = spec.validate(ctx)
        assert len(issues) > 0
        assert any("Incompatible dimensions" in issue for issue in issues)


class TestImportPerformance:
    """Test that imports are fast."""

    def test_core_import_time(self):
        """Test that core imports are fast."""
        import time
        import subprocess
        import sys

        code = """import time\nstart = time.perf_counter()\nimport energy_transformer\nelapsed = time.perf_counter() - start\nprint(f'{elapsed:.3f}')"""

        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )

        import_time = float(result.stdout.strip())
        print(f"Core import time: {import_time:.3f}s")

        assert import_time < 0.5, f"Import too slow: {import_time:.3f}s"

    def test_lazy_import_behavior(self):
        """Test that heavy imports are actually lazy."""
        import subprocess
        import sys

        code = """import sys\nimport energy_transformer\nassert 'scipy' not in sys.modules, 'scipy was loaded on import!'\nassert 'matplotlib' not in sys.modules, 'matplotlib was loaded on import!'\nassert 'energy_transformer.models' not in sys.modules, 'models loaded early!'\nfrom energy_transformer import EnergyTransformer\nassert 'energy_transformer.models' in sys.modules, 'models not loaded when needed!'\nprint('SUCCESS')"""

        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )

        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "SUCCESS" in result.stdout
