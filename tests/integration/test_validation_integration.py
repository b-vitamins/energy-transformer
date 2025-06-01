import pytest

pytestmark = pytest.mark.integration

class TestIntegration:
    """Integration tests combining multiple fixes."""

    def test_complex_formula_with_validation(self):
        from energy_transformer.spec.primitives import Context, Dimension

        ctx = Context(
            dimensions={"embed_dim": 768, "num_heads": 12, "mlp_ratio": 4},
        )

        dim1 = Dimension("head_dim", formula="embed_dim / num_heads")
        assert dim1.resolve(ctx) == 64

        dim2 = Dimension("mlp_hidden", formula="embed_dim * mlp_ratio")
        assert dim2.resolve(ctx) == 3072

        dim3 = Dimension("exploit", formula="exec('x=1') or embed_dim")
        assert dim3.resolve(ctx) is None

