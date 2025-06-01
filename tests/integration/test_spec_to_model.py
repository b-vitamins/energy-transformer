import pytest

from energy_transformer.spec import Context

pytestmark = pytest.mark.integration


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
