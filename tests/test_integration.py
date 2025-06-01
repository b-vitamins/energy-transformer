"""Integration tests for complete workflows."""

import pytest
import torch
from torch import nn

from energy_transformer import realise, seq
from energy_transformer.spec import Context


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
            [m for m in model.modules() if "ETBlock" in str(type(m))]
        )
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
        import subprocess
        import sys

        code = """import time\nstart = time.perf_counter()\nimport energy_transformer\nelapsed = time.perf_counter() - start\nprint(f'{elapsed:.3f}')"""

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
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
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "SUCCESS" in result.stdout


class TestImportBehavior:
    """Test import performance and behavior from verify_imports.py."""

    def test_verify_imports_script_core_import_time(self):
        """Test core import time from verify_imports.py."""
        import subprocess
        import sys

        code = """
import time
start = time.perf_counter()
import energy_transformer
elapsed = time.perf_counter() - start
print(elapsed)
"""

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        assert result.returncode == 0, f"Import failed: {result.stderr}"

        import_time = float(result.stdout.strip())
        assert import_time < 0.5, f"Core import too slow: {import_time:.3f}s"

    def test_verify_imports_script_lazy_loading(self):
        """Test lazy loading behavior from verify_imports.py."""
        import subprocess
        import sys

        code = """
import sys

# Import core
import energy_transformer

# Check what's NOT loaded
heavy_modules = ['scipy', 'matplotlib', 'seaborn', 'energy_transformer.models']
not_loaded = [m for m in heavy_modules if m not in sys.modules]

# These should NOT be loaded yet
assert 'scipy' not in sys.modules, "scipy should not be loaded on import"
assert 'matplotlib' not in sys.modules, "matplotlib should not be loaded on import"
assert 'energy_transformer.models' not in sys.modules, "models should not be loaded on import"

# Now trigger a heavy import
from energy_transformer import EnergyTransformer

# Models should now be loaded
assert 'energy_transformer.models' in sys.modules, "models should load when accessed"

print("SUCCESS")
"""

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        assert result.returncode == 0, (
            f"Lazy loading test failed: {result.stderr}"
        )
        assert "SUCCESS" in result.stdout

    def test_verify_imports_script_no_side_effects(self):
        """Test no side effects on import from verify_imports.py."""
        import subprocess
        import sys

        code = """
# Import and check for side effects
import energy_transformer.spec
from energy_transformer.spec.realise import _config

# Get initial state
initial_cache_enabled = _config.cache.enabled
initial_cache_size = _config.cache.max_size
initial_strict = _config.strict

# Import everything
from energy_transformer import *
from energy_transformer.spec import *

# Check if state changed
assert _config.cache.enabled == initial_cache_enabled, "cache.enabled changed on import!"
assert _config.cache.max_size == initial_cache_size, "cache.max_size changed on import!"
assert _config.strict == initial_strict, "strict mode changed on import!"

print("SUCCESS: No side effects detected")
"""

        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        assert result.returncode == 0, (
            f"Side effects test failed: {result.stderr}"
        )
        assert "SUCCESS" in result.stdout
