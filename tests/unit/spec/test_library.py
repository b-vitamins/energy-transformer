"""Tests for library specifications.

This module tests all pre-built specifications in the library module,
including validation, context handling, and composition.
"""

from dataclasses import fields

import pytest

from energy_transformer.spec.library import (
    ClassificationHeadSpec,
    CLSTokenSpec,
    DropoutSpec,
    ETBlockSpec,
    FeatureHeadSpec,
    HNSpec,
    IdentitySpec,
    LayerNormSpec,
    MHASpec,
    MHEASpec,
    MLPSpec,
    PatchEmbedSpec,
    PosEmbedSpec,
    SHNSpec,
    TransformerBlockSpec,
    VisionEmbeddingSpec,
    to_pair,
    validate_dimension,
    validate_positive,
    validate_probability,
)
from energy_transformer.spec.primitives import Context, ValidationError


class TestUtilityFunctions:
    """Test utility functions."""

    def test_to_pair(self):
        """Test to_pair conversion."""
        assert to_pair(5) == (5, 5)
        assert to_pair((3, 4)) == (3, 4)
        assert to_pair((10, 20)) == (10, 20)

    def test_validate_positive(self):
        """Test positive validation."""
        assert validate_positive(1)
        assert validate_positive(0.5)
        assert validate_positive(1000)
        assert not validate_positive(0)
        assert not validate_positive(-1)
        assert not validate_positive(-0.5)

    def test_validate_probability(self):
        """Test probability validation."""
        assert validate_probability(0)
        assert validate_probability(0.5)
        assert validate_probability(1)
        assert not validate_probability(-0.1)
        assert not validate_probability(1.1)
        assert not validate_probability(2)

    def test_validate_dimension(self):
        """Test dimension validation."""
        assert validate_dimension(1)
        assert validate_dimension(768)
        assert validate_dimension(65536)
        assert not validate_dimension(0)
        assert not validate_dimension(-1)
        assert not validate_dimension(65537)


class TestLayerNormSpec:
    """Test LayerNormSpec."""

    def test_creation(self):
        """Test LayerNormSpec creation."""
        ln = LayerNormSpec()
        assert ln.eps == 1e-5

        ln2 = LayerNormSpec(eps=1e-6)
        assert ln2.eps == 1e-6

    def test_validation(self):
        """Test validation."""
        # Valid eps
        ln = LayerNormSpec(eps=1e-8)
        ctx = Context(dimensions={"embed_dim": 768})
        issues = ln.validate(ctx)
        assert len(issues) == 0

        # Invalid eps
        with pytest.raises(ValidationError):
            LayerNormSpec(eps=0)

        with pytest.raises(ValidationError):
            LayerNormSpec(eps=-1e-5)

    def test_requires_embed_dim(self):
        """Test embed_dim requirement."""
        ln = LayerNormSpec()

        # Without embed_dim
        ctx = Context()
        issues = ln.validate(ctx)
        assert any("embed_dim" in issue for issue in issues)

        # With embed_dim
        ctx = Context(dimensions={"embed_dim": 512})
        issues = ln.validate(ctx)
        assert len(issues) == 0

    def test_frozen(self):
        """Test spec is frozen."""
        ln = LayerNormSpec()
        with pytest.raises(AttributeError):
            ln.eps = 1e-4


class TestPatchEmbedSpec:
    """Test PatchEmbedSpec."""

    def test_creation(self):
        """Test basic creation."""
        pe = PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768)
        assert pe.img_size == 224
        assert pe.patch_size == 16
        assert pe.embed_dim == 768
        assert pe.in_chans == 3
        assert pe.bias is True

    def test_tuple_sizes(self):
        """Test non-square image and patch sizes."""
        pe = PatchEmbedSpec(
            img_size=(224, 336), patch_size=(16, 24), embed_dim=768
        )
        assert pe.img_size == (224, 336)
        assert pe.patch_size == (16, 24)

    def test_patch_calculation(self):
        """Test num_patches calculation."""
        # Square
        pe1 = PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768)
        ctx1 = pe1.apply_context(Context())
        assert ctx1.get_dim("num_patches") == 196  # (224/16)^2
        assert ctx1.get_dim("embed_dim") == 768

        # Non-square
        pe2 = PatchEmbedSpec(
            img_size=(224, 112), patch_size=(16, 16), embed_dim=512
        )
        ctx2 = pe2.apply_context(Context())
        assert ctx2.get_dim("num_patches") == 98  # (224/16) * (112/16)
        assert ctx2.get_dim("embed_dim") == 512

    def test_validation(self):
        """Test parameter validation."""
        # Valid
        pe = PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768)
        issues = pe.validate(Context())
        assert len(issues) == 0

        # Invalid sizes
        with pytest.raises(ValidationError):
            PatchEmbedSpec(img_size=0, patch_size=16, embed_dim=768)

        with pytest.raises(ValidationError):
            PatchEmbedSpec(img_size=224, patch_size=-16, embed_dim=768)

        with pytest.raises(ValidationError):
            PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=0)

    def test_provides_dimensions(self):
        """Test that it provides required dimensions."""
        pe = PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768)
        assert "embed_dim" in pe._provides
        assert "num_patches" in pe._provides


class TestCLSTokenSpec:
    """Test CLSTokenSpec."""

    def test_creation(self):
        """Test basic creation."""
        CLSTokenSpec()
        # No parameters to check

    def test_increments_patches(self):
        """Test that it increments num_patches."""
        cls = CLSTokenSpec()

        # With existing patches
        ctx = Context(dimensions={"embed_dim": 768, "num_patches": 196})
        new_ctx = cls.apply_context(ctx)
        assert new_ctx.get_dim("num_patches") == 197

        # Without patches - shouldn't fail
        ctx2 = Context(dimensions={"embed_dim": 768})
        new_ctx2 = cls.apply_context(ctx2)
        assert new_ctx2.get_dim("num_patches") is None

    def test_requires_embed_dim(self):
        """Test embed_dim requirement."""
        cls = CLSTokenSpec()

        ctx = Context()
        issues = cls.validate(ctx)
        assert any("embed_dim" in issue for issue in issues)

        ctx = Context(dimensions={"embed_dim": 768})
        issues = cls.validate(ctx)
        assert len(issues) == 0

    def test_modifies_num_patches(self):
        """Test that it declares modifying num_patches."""
        cls = CLSTokenSpec()
        assert "num_patches" in cls._modifies


class TestPosEmbedSpec:
    """Test PosEmbedSpec."""

    def test_creation(self):
        """Test basic creation."""
        pe = PosEmbedSpec()
        assert pe.include_cls is False
        assert pe.init_std == 0.02

        pe2 = PosEmbedSpec(include_cls=True, init_std=0.01)
        assert pe2.include_cls is True
        assert pe2.init_std == 0.01

    def test_validation(self):
        """Test validation."""
        pe = PosEmbedSpec(init_std=0.02)

        # Valid with requirements
        ctx = Context(dimensions={"embed_dim": 768, "num_patches": 196})
        issues = pe.validate(ctx)
        assert len(issues) == 0

        # Invalid init_std
        with pytest.raises(ValidationError):
            PosEmbedSpec(init_std=0)

        with pytest.raises(ValidationError):
            PosEmbedSpec(init_std=-0.02)

    def test_requires_dimensions(self):
        """Test dimension requirements."""
        pe = PosEmbedSpec()

        # Missing embed_dim
        ctx = Context(dimensions={"num_patches": 196})
        issues = pe.validate(ctx)
        assert any("embed_dim" in issue for issue in issues)

        # Missing num_patches
        ctx = Context(dimensions={"embed_dim": 768})
        issues = pe.validate(ctx)
        assert any("num_patches" in issue for issue in issues)


class TestMHEASpec:
    """Test Multi-Head Energy Attention spec."""

    def test_creation(self):
        """Test basic creation."""
        mhea = MHEASpec()
        assert mhea.num_heads == 12
        assert mhea.head_dim == 64
        assert mhea.beta is None
        assert mhea.bias is False
        assert mhea.init_std == 0.002

        mhea2 = MHEASpec(num_heads=8, head_dim=96, beta=0.5, bias=True)
        assert mhea2.num_heads == 8
        assert mhea2.head_dim == 96
        assert mhea2.beta == 0.5
        assert mhea2.bias is True

    def test_validation(self):
        """Test parameter validation."""
        # Valid
        mhea = MHEASpec(num_heads=16, head_dim=64)
        ctx = Context(dimensions={"embed_dim": 1024})
        issues = mhea.validate(ctx)
        assert len(issues) == 0

        # Invalid num_heads
        with pytest.raises(ValidationError):
            MHEASpec(num_heads=0)

        with pytest.raises(ValidationError):
            MHEASpec(num_heads=-1)

        # Invalid head_dim
        with pytest.raises(ValidationError):
            MHEASpec(head_dim=0)

        # Invalid init_std
        with pytest.raises(ValidationError):
            MHEASpec(init_std=0)

    def test_requires_embed_dim(self):
        """Test embed_dim requirement."""
        mhea = MHEASpec()

        ctx = Context()
        issues = mhea.validate(ctx)
        assert any("embed_dim" in issue for issue in issues)


class TestHNSpec:
    """Test Hopfield Network spec."""

    def test_creation(self):
        """Test basic creation."""
        hn = HNSpec()
        assert hn.hidden_dim is None
        assert hn.multiplier == 4.0
        assert hn.energy_fn == "relu_squared"

        hn2 = HNSpec(multiplier=3.0, energy_fn="softmax")
        assert hn2.multiplier == 3.0
        assert hn2.energy_fn == "softmax"

    def test_hidden_dim_calculation(self):
        """Test automatic hidden_dim calculation."""
        hn = HNSpec(multiplier=4.0)

        ctx = Context(dimensions={"embed_dim": 768})
        new_ctx = hn.apply_context(ctx)
        assert new_ctx.get_dim("hopfield_hidden_dim") == 3072  # 768 * 4

        # With explicit hidden_dim
        from energy_transformer.spec.primitives import Dimension

        HNSpec(hidden_dim=Dimension("custom", value=2048))
        # Would need more complex setup to test Dimension resolution

    def test_validation(self):
        """Test parameter validation."""
        # Valid
        hn = HNSpec(multiplier=2.0, energy_fn="relu_squared")
        ctx = Context(dimensions={"embed_dim": 768})
        issues = hn.validate(ctx)
        assert len(issues) == 0

        # Invalid multiplier
        with pytest.raises(ValidationError):
            HNSpec(multiplier=0)

        with pytest.raises(ValidationError):
            HNSpec(multiplier=10)  # > 8

        # Invalid energy_fn
        with pytest.raises(ValidationError):
            HNSpec(energy_fn="invalid")

    def test_energy_fn_choices(self):
        """Test energy function choices."""
        valid_fns = ["relu_squared", "softmax", "tanh"]

        for fn in valid_fns:
            hn = HNSpec(energy_fn=fn)
            assert hn.energy_fn == fn

        with pytest.raises(ValidationError):
            HNSpec(energy_fn="custom")


class TestSHNSpec:
    """Test Simplicial Hopfield Network spec."""

    def test_creation(self):
        """Test basic creation."""
        sh = SHNSpec()
        assert sh.simplices is None
        assert sh.num_vertices is None
        assert sh.max_dim == 1
        assert sh.budget == 0.1
        assert sh.dim_weights is None
        assert sh.coordinates is None
        assert sh.hidden_dim is None
        assert sh.multiplier == 4.0
        assert sh.temperature == 0.5

    def test_with_simplices(self):
        """Test with manual simplices."""
        simplices = [[0, 1], [1, 2], [0, 1, 2]]  # Edges and triangle
        sh = SHNSpec(simplices=simplices, max_dim=2)
        assert sh.simplices == simplices
        assert sh.max_dim == 2

    def test_with_coordinates(self):
        """Test with spatial coordinates."""
        coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        sh = SHNSpec(coordinates=coords, num_vertices=4, budget=0.2)
        assert sh.coordinates == coords
        assert sh.num_vertices == 4

    def test_validation(self):
        """Test parameter validation."""
        # Valid
        sh = SHNSpec(max_dim=2, budget=0.5, temperature=1.0)
        ctx = Context(dimensions={"embed_dim": 768, "num_patches": 196})
        issues = sh.validate(ctx)
        assert len(issues) == 0

        # Invalid max_dim
        with pytest.raises(ValidationError):
            SHNSpec(max_dim=0)

        with pytest.raises(ValidationError):
            SHNSpec(max_dim=4)

        # Invalid budget
        with pytest.raises(ValidationError):
            SHNSpec(budget=0)

        with pytest.raises(ValidationError):
            SHNSpec(budget=1.5)

        # Invalid temperature
        with pytest.raises(ValidationError):
            SHNSpec(temperature=0)

    def test_context_application(self):
        """Test context dimension setting."""
        sh = SHNSpec(multiplier=3.0)

        ctx = Context(dimensions={"embed_dim": 512, "num_patches": 100})
        new_ctx = sh.apply_context(ctx)

        assert new_ctx.get_dim("simplicial_hidden_dim") == 1536  # 512 * 3
        assert new_ctx.get_dim("simplicial_vertices") == 100

        # With explicit num_vertices
        sh2 = SHNSpec(num_vertices=50)
        ctx2 = Context(dimensions={"embed_dim": 512, "num_patches": 100})
        new_ctx2 = sh2.apply_context(ctx2)
        assert new_ctx2.get_dim("simplicial_vertices") != 50  # Doesn't override


class TestETBlockSpec:
    """Test Energy Transformer spec."""

    def test_creation(self):
        """Test basic creation."""
        et = ETBlockSpec()
        assert et.steps == 12
        assert et.alpha == 0.125
        assert isinstance(et.layer_norm, LayerNormSpec)
        assert isinstance(et.attention, MHEASpec)
        assert isinstance(et.hopfield, HNSpec)

    def test_custom_components(self):
        """Test with custom components."""
        et = ETBlockSpec(
            steps=6,
            alpha=0.25,
            layer_norm=LayerNormSpec(eps=1e-6),
            attention=MHEASpec(num_heads=8),
            hopfield=SHNSpec(max_dim=2),
        )

        assert et.steps == 6
        assert et.alpha == 0.25
        assert et.layer_norm.eps == 1e-6
        assert et.attention.num_heads == 8
        assert isinstance(et.hopfield, SHNSpec)
        assert et.hopfield.max_dim == 2

    def test_validation(self):
        """Test parameter validation."""
        # Valid
        et = ETBlockSpec(steps=20, alpha=0.1)
        ctx = Context(dimensions={"embed_dim": 768})
        issues = et.validate(ctx)
        assert len(issues) == 0

        # Invalid steps
        with pytest.raises(ValidationError):
            ETBlockSpec(steps=0)

        with pytest.raises(ValidationError):
            ETBlockSpec(steps=100)  # > 50

        # Invalid alpha
        with pytest.raises(ValidationError):
            ETBlockSpec(alpha=0)

        with pytest.raises(ValidationError):
            ETBlockSpec(alpha=-0.1)

    def test_requires_embed_dim(self):
        """Test embed_dim requirement."""
        et = ETBlockSpec()

        ctx = Context()
        issues = et.validate(ctx)
        assert any("embed_dim" in issue for issue in issues)

    def test_nested_validation(self):
        """Test that nested specs are validated."""
        et = ETBlockSpec(
            layer_norm=LayerNormSpec(), attention=MHEASpec(), hopfield=HNSpec()
        )

        # All nested specs require embed_dim
        ctx = Context()
        issues = et.validate(ctx)
        # Should have multiple issues about embed_dim
        embed_issues = [i for i in issues if "embed_dim" in i]
        assert len(embed_issues) > 1


class TestClassificationHeadSpec:
    """Test ClassificationHeadSpec."""

    def test_creation(self):
        """Test basic creation."""
        head = ClassificationHeadSpec(num_classes=1000)
        assert head.num_classes == 1000
        assert head.representation_size is None
        assert head.drop_rate == 0.0
        assert head.use_cls_token is True

        head2 = ClassificationHeadSpec(
            num_classes=10,
            representation_size=512,
            drop_rate=0.1,
            use_cls_token=False,
        )
        assert head2.num_classes == 10
        assert head2.representation_size == 512
        assert head2.drop_rate == 0.1
        assert head2.use_cls_token is False

    def test_validation(self):
        """Test parameter validation."""
        # Valid
        head = ClassificationHeadSpec(num_classes=100, drop_rate=0.5)
        ctx = Context(dimensions={"embed_dim": 768})
        issues = head.validate(ctx)
        assert len(issues) == 0

        # Invalid num_classes
        with pytest.raises(ValidationError):
            ClassificationHeadSpec(num_classes=0)

        with pytest.raises(ValidationError):
            ClassificationHeadSpec(num_classes=-1)

        # Invalid drop_rate
        with pytest.raises(ValidationError):
            ClassificationHeadSpec(num_classes=10, drop_rate=-0.1)

        with pytest.raises(ValidationError):
            ClassificationHeadSpec(num_classes=10, drop_rate=1.1)

    def test_provides_num_classes(self):
        """Test that it provides num_classes."""
        head = ClassificationHeadSpec(num_classes=1000)
        assert "num_classes" in head._provides

        ctx = Context(dimensions={"embed_dim": 768})
        new_ctx = head.apply_context(ctx)
        assert new_ctx.get_dim("num_classes") == 1000

    def test_requires_embed_dim(self):
        """Test embed_dim requirement."""
        head = ClassificationHeadSpec(num_classes=10)

        ctx = Context()
        issues = head.validate(ctx)
        assert any("embed_dim" in issue for issue in issues)


class TestFeatureHeadSpec:
    """Test FeatureHeadSpec."""

    def test_creation(self):
        """Test basic creation."""
        head = FeatureHeadSpec()
        assert head.use_cls_token is True

        head2 = FeatureHeadSpec(use_cls_token=False)
        assert head2.use_cls_token is False

    def test_requires_embed_dim(self):
        """Test embed_dim requirement."""
        head = FeatureHeadSpec()

        ctx = Context()
        issues = head.validate(ctx)
        assert any("embed_dim" in issue for issue in issues)

        ctx = Context(dimensions={"embed_dim": 768})
        issues = head.validate(ctx)
        assert len(issues) == 0


class TestVisionEmbeddingSpec:
    """Test VisionEmbeddingSpec composite."""

    def test_creation(self):
        """Test basic creation."""
        ve = VisionEmbeddingSpec(img_size=224, patch_size=16, embed_dim=768)
        assert ve.img_size == 224
        assert ve.patch_size == 16
        assert ve.embed_dim == 768
        assert ve.in_chans == 3
        assert ve.use_cls_token is True
        assert ve.drop_rate == 0.0

    def test_patch_calculation_with_cls(self):
        """Test patch calculation includes CLS token."""
        ve = VisionEmbeddingSpec(
            img_size=224, patch_size=16, embed_dim=768, use_cls_token=True
        )

        ctx = ve.apply_context(Context())
        assert ctx.get_dim("embed_dim") == 768
        assert ctx.get_dim("num_patches") == 197  # 196 patches + 1 CLS

    def test_patch_calculation_without_cls(self):
        """Test patch calculation without CLS token."""
        ve = VisionEmbeddingSpec(
            img_size=224, patch_size=16, embed_dim=768, use_cls_token=False
        )

        ctx = ve.apply_context(Context())
        assert ctx.get_dim("num_patches") == 196  # Just patches

    def test_validation(self):
        """Test parameter validation."""
        # Valid
        ve = VisionEmbeddingSpec(
            img_size=224, patch_size=16, embed_dim=768, drop_rate=0.1
        )
        issues = ve.validate(Context())
        assert len(issues) == 0

        # Invalid parameters
        with pytest.raises(ValidationError):
            VisionEmbeddingSpec(img_size=0, patch_size=16, embed_dim=768)

        with pytest.raises(ValidationError):
            VisionEmbeddingSpec(img_size=224, patch_size=16, embed_dim=0)

        with pytest.raises(ValidationError):
            VisionEmbeddingSpec(
                img_size=224, patch_size=16, embed_dim=768, drop_rate=1.5
            )

    def test_provides_dimensions(self):
        """Test that it provides required dimensions."""
        ve = VisionEmbeddingSpec(img_size=224, patch_size=16, embed_dim=768)
        assert "embed_dim" in ve._provides
        assert "num_patches" in ve._provides


class TestMHASpec:
    """Test MHASpec."""

    def test_creation(self):
        """Test basic creation."""
        attn = MHASpec()
        assert attn.num_heads == 8
        assert attn.qkv_bias is True
        assert attn.attn_drop == 0.0
        assert attn.proj_drop == 0.0

        attn2 = MHASpec(
            num_heads=16, qkv_bias=False, attn_drop=0.1, proj_drop=0.2
        )
        assert attn2.num_heads == 16
        assert attn2.qkv_bias is False
        assert attn2.attn_drop == 0.1
        assert attn2.proj_drop == 0.2

    def test_validation(self):
        """Test parameter validation."""
        # Valid
        attn = MHASpec(num_heads=12, attn_drop=0.1)
        ctx = Context(dimensions={"embed_dim": 768})
        issues = attn.validate(ctx)
        assert len(issues) == 0

        # Invalid num_heads
        with pytest.raises(ValidationError):
            MHASpec(num_heads=0)

        # Invalid dropout rates
        with pytest.raises(ValidationError):
            MHASpec(attn_drop=-0.1)

        with pytest.raises(ValidationError):
            MHASpec(proj_drop=1.1)

    def test_requires_embed_dim(self):
        """Test embed_dim requirement."""
        attn = MHASpec()

        ctx = Context()
        issues = attn.validate(ctx)
        assert any("embed_dim" in issue for issue in issues)


class TestMLPSpec:
    """Test MLPSpec."""

    def test_creation(self):
        """Test basic creation."""
        mlp = MLPSpec()
        assert mlp.hidden_features is None
        assert mlp.out_features is None
        assert mlp.activation == "gelu"
        assert mlp.drop == 0.0

        mlp2 = MLPSpec(
            hidden_features=3072, out_features=768, activation="relu", drop=0.1
        )
        assert mlp2.hidden_features == 3072
        assert mlp2.out_features == 768
        assert mlp2.activation == "relu"
        assert mlp2.drop == 0.1

    def test_activation_choices(self):
        """Test activation function choices."""
        valid_activations = ["gelu", "relu", "swish", "silu"]

        for act in valid_activations:
            mlp = MLPSpec(activation=act)
            assert mlp.activation == act

        # Invalid activation should fail at type checking level
        # but we use string literal type

    def test_validation(self):
        """Test parameter validation."""
        # Valid
        mlp = MLPSpec(hidden_features=3072, drop=0.1)
        ctx = Context(dimensions={"embed_dim": 768})
        issues = mlp.validate(ctx)
        assert len(issues) == 0

        # Invalid drop rate
        with pytest.raises(ValidationError):
            MLPSpec(drop=-0.1)

        with pytest.raises(ValidationError):
            MLPSpec(drop=1.1)

    def test_requires_embed_dim(self):
        """Test embed_dim requirement."""
        mlp = MLPSpec()

        ctx = Context()
        issues = mlp.validate(ctx)
        assert any("embed_dim" in issue for issue in issues)


class TestTransformerBlockSpec:
    """Test TransformerBlockSpec."""

    def test_creation(self):
        """Test basic creation."""
        block = TransformerBlockSpec()
        assert isinstance(block.attention, MHASpec)
        assert isinstance(block.mlp, MLPSpec)
        assert block.drop_path == 0.0
        assert block.norm_first is True

        # Custom components
        block2 = TransformerBlockSpec(
            attention=MHASpec(num_heads=16),
            mlp=MLPSpec(activation="swish"),
            drop_path=0.1,
            norm_first=False,
        )
        assert block2.attention.num_heads == 16
        assert block2.mlp.activation == "swish"
        assert block2.drop_path == 0.1
        assert block2.norm_first is False

    def test_validation(self):
        """Test parameter validation."""
        # Valid
        block = TransformerBlockSpec(drop_path=0.2)
        ctx = Context(dimensions={"embed_dim": 768})
        issues = block.validate(ctx)
        assert len(issues) == 0

        # Invalid drop_path
        with pytest.raises(ValidationError):
            TransformerBlockSpec(drop_path=-0.1)

        with pytest.raises(ValidationError):
            TransformerBlockSpec(drop_path=1.1)

    def test_nested_requirements(self):
        """Test nested components' requirements."""
        block = TransformerBlockSpec()

        # Without embed_dim, nested components should fail
        ctx = Context()
        issues = block.validate(ctx)
        # Should have issues from attention and mlp
        assert len(issues) >= 2
        assert any("embed_dim" in issue for issue in issues)


class TestDropoutSpec:
    """Test DropoutSpec."""

    def test_creation(self):
        """Test basic creation."""
        drop = DropoutSpec()
        assert drop.p == 0.5
        assert drop.inplace is False

        drop2 = DropoutSpec(p=0.1, inplace=True)
        assert drop2.p == 0.1
        assert drop2.inplace is True

    def test_validation(self):
        """Test parameter validation."""
        # Valid
        drop = DropoutSpec(p=0.3)
        issues = drop.validate(Context())
        assert len(issues) == 0

        # Invalid probability
        with pytest.raises(ValidationError):
            DropoutSpec(p=-0.1)

        with pytest.raises(ValidationError):
            DropoutSpec(p=1.1)

    def test_edge_cases(self):
        """Test edge case probabilities."""
        drop0 = DropoutSpec(p=0.0)  # No dropout
        drop1 = DropoutSpec(p=1.0)  # Always drop

        assert drop0.p == 0.0
        assert drop1.p == 1.0


class TestIdentitySpec:
    """Test IdentitySpec."""

    def test_creation(self):
        """Test basic creation."""
        IdentitySpec()
        # No parameters to check

    def test_no_requirements(self):
        """Test that it has no requirements."""
        identity = IdentitySpec()

        # Should validate in any context
        ctx = Context()
        issues = identity.validate(ctx)
        assert len(issues) == 0

        # Even with dimensions
        ctx = Context(dimensions={"anything": 42})
        issues = identity.validate(ctx)
        assert len(issues) == 0


class TestComplexCompositions:
    """Test complex compositions of library specs."""

    def test_vision_transformer_embedding(self):
        """Test typical ViT embedding pipeline."""
        from energy_transformer.spec.combinators import seq

        # Traditional approach
        embed_pipeline = seq(
            PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
        )

        ctx = Context()
        final_ctx = embed_pipeline.apply_context(ctx)

        assert final_ctx.get_dim("embed_dim") == 768
        assert final_ctx.get_dim("num_patches") == 197  # 196 + 1 CLS

    def test_vision_embedding_equivalent(self):
        """Test VisionEmbeddingSpec produces similar results."""
        ve = VisionEmbeddingSpec(
            img_size=224, patch_size=16, embed_dim=768, use_cls_token=True
        )

        ctx = ve.apply_context(Context())

        assert ctx.get_dim("embed_dim") == 768
        assert ctx.get_dim("num_patches") == 197

    def test_energy_transformer_block(self):
        """Test ET block with different Hopfield variants."""
        # Standard Hopfield
        et1 = ETBlockSpec(steps=6, hopfield=HNSpec(multiplier=4.0))

        # Simplicial Hopfield
        et2 = ETBlockSpec(
            steps=6,
            hopfield=SHNSpec(max_dim=2, budget=0.2, temperature=0.7),
        )

        ctx = Context(dimensions={"embed_dim": 768})

        issues1 = et1.validate(ctx)
        issues2 = et2.validate(ctx)

        assert len(issues1) == 0
        assert len(issues2) == 0

    def test_complete_model_spec(self):
        """Test complete model specification."""
        from energy_transformer.spec.combinators import loop, seq

        model = seq(
            # Embedding
            VisionEmbeddingSpec(
                img_size=224,
                patch_size=16,
                embed_dim=768,
                use_cls_token=True,
                drop_rate=0.1,
            ),
            # Transformer blocks
            loop(
                ETBlockSpec(
                    steps=12,
                    alpha=0.125,
                    attention=MHEASpec(num_heads=12, head_dim=64),
                    hopfield=HNSpec(multiplier=4.0),
                ),
                times=12,
            ),
            # Output head
            LayerNormSpec(),
            ClassificationHeadSpec(
                num_classes=1000, drop_rate=0.1, use_cls_token=True
            ),
        )

        # Validate entire model
        ctx = Context()
        issues = model.validate(ctx)
        assert len(issues) == 0

        # Check final context
        final_ctx = model.apply_context(ctx)
        assert final_ctx.get_dim("embed_dim") == 768
        assert final_ctx.get_dim("num_patches") == 197
        assert final_ctx.get_dim("num_classes") == 1000


class TestSpecMetadata:
    """Test spec metadata and introspection."""

    def test_all_specs_have_version(self):
        """Test all specs have version info."""
        specs = [
            LayerNormSpec,
            PatchEmbedSpec,
            CLSTokenSpec,
            PosEmbedSpec,
            MHEASpec,
            HNSpec,
            SHNSpec,
            ETBlockSpec,
            ClassificationHeadSpec,
            FeatureHeadSpec,
            VisionEmbeddingSpec,
            MHASpec,
            MLPSpec,
            TransformerBlockSpec,
            DropoutSpec,
            IdentitySpec,
        ]

        for spec_cls in specs:
            assert hasattr(spec_cls, "_version")
            assert spec_cls._version == "1.0"
            assert hasattr(spec_cls, "_compatible_versions")
            assert "1.0" in spec_cls._compatible_versions

    def test_dimension_declarations(self):
        """Test dimension requirements are properly declared."""
        # Specs that require embed_dim
        require_embed = [
            LayerNormSpec,
            CLSTokenSpec,
            PosEmbedSpec,
            MHEASpec,
            HNSpec,
            SHNSpec,
            ETBlockSpec,
            ClassificationHeadSpec,
            FeatureHeadSpec,
            MHASpec,
            MLPSpec,
            TransformerBlockSpec,
        ]

        for spec_cls in require_embed:
            assert "embed_dim" in spec_cls._requires

        # Specs that provide dimensions
        assert "embed_dim" in PatchEmbedSpec._provides
        assert "num_patches" in PatchEmbedSpec._provides
        assert "num_classes" in ClassificationHeadSpec._provides
        assert "embed_dim" in VisionEmbeddingSpec._provides
        assert "num_patches" in VisionEmbeddingSpec._provides

        # Specs that modify dimensions
        assert "num_patches" in CLSTokenSpec._modifies

    def test_parameter_metadata(self):
        """Test parameter metadata is properly set."""
        # Check a few key parameters
        pe_fields = {f.name: f for f in fields(PatchEmbedSpec)}

        assert pe_fields["img_size"].metadata.get("validator") is not None
        assert pe_fields["embed_dim"].metadata.get("validator") is not None

        # Check choices
        hn_fields = {f.name: f for f in fields(HNSpec)}
        assert hn_fields["energy_fn"].metadata.get("choices") == [
            "relu_squared",
            "softmax",
            "tanh",
        ]

    def test_spec_serialization(self):
        """Test specs can be serialized and reconstructed."""
        specs = [
            LayerNormSpec(eps=1e-6),
            PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768),
            MHEASpec(num_heads=8, head_dim=96),
            ClassificationHeadSpec(num_classes=100, drop_rate=0.1),
        ]

        for spec in specs:
            # To dict
            d = spec.to_dict()
            assert d["_type"] == spec.__class__.__name__
            assert d["_version"] == "1.0"

            # From dict
            spec2 = type(spec).from_dict(d)
            assert type(spec2) is type(spec)

            # Check key attributes match
            for field in fields(spec):
                assert getattr(spec2, field.name) == getattr(spec, field.name)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_patch_sizes(self):
        """Test invalid patch size configurations."""
        # Patch size larger than image - should create but fail validation
        spec = PatchEmbedSpec(img_size=16, patch_size=32, embed_dim=768)

        # Should be created successfully
        assert spec.img_size == 16
        assert spec.patch_size == 32

        # But should fail validation
        ctx = Context()
        issues = spec.validate(ctx)
        assert len(issues) > 0
        assert any("cannot be larger than" in issue for issue in issues)

        # Should produce 0 patches
        new_ctx = spec.apply_context(ctx)
        assert new_ctx.get_dim("num_patches") == 0  # 16 // 32 = 0

    def test_dimension_limits(self):
        """Test dimension size limits."""
        # Very large embed_dim
        pe = PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=65536)
        assert pe.embed_dim == 65536  # At limit

        # Too large
        with pytest.raises(ValidationError):
            PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=65537)

    def test_probability_edge_cases(self):
        """Test probability parameters at boundaries."""
        # 0 and 1 are valid probabilities
        drop1 = DropoutSpec(p=0.0)
        drop2 = DropoutSpec(p=1.0)

        head1 = ClassificationHeadSpec(num_classes=10, drop_rate=0.0)
        head2 = ClassificationHeadSpec(num_classes=10, drop_rate=1.0)

        # All should validate
        for spec in [drop1, drop2, head1, head2]:
            issues = spec.validate(Context(dimensions={"embed_dim": 768}))
            assert len(issues) == 0

    def test_empty_simplicial_complex(self):
        """Test simplicial complex with edge cases."""
        # Empty simplices
        sh1 = SHNSpec(simplices=[])
        assert sh1.simplices == []

        # Single vertex
        sh2 = SHNSpec(num_vertices=1)
        assert sh2.num_vertices == 1

        # Zero budget (should fail)
        with pytest.raises(ValidationError):
            SHNSpec(budget=0)
