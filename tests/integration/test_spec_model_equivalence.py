"""Integration tests verifying spec system produces identical modules to direct construction."""

import pytest
import torch

from energy_transformer.layers import (
    ClassifierHead,
    ConvPatchEmbed,
    EnergyLayerNorm,
    HopfieldNetwork,
    MultiheadEnergyAttention,
    PosEmbed2D,
)
from energy_transformer.models import EnergyTransformer
from energy_transformer.models.vision import (
    viet_2l_cifar,
    viet_base,
    viset_2l_e50_t50_cifar,
)
from energy_transformer.spec import Context, loop, realise, seq
from energy_transformer.spec.library import (
    ClassificationHeadSpec,
    CLSTokenSpec,
    ETBlockSpec,
    HNSpec,
    LayerNormSpec,
    MHEASpec,
    PatchEmbedSpec,
    PosEmbedSpec,
    SHNSpec,
)
from energy_transformer.spec.primitives import ValidationError
from energy_transformer.utils.optimizers import SGD

pytestmark = pytest.mark.integration


class TestComponentEquivalence:
    """Test individual components match between spec and direct construction."""

    def test_patch_embedding_equivalence(self):
        """PatchEmbedSpec should produce identical ConvPatchEmbed."""
        direct = ConvPatchEmbed(
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            bias=True,
        )

        spec = PatchEmbedSpec(
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            bias=True,
        )
        from_spec = realise(spec)

        assert isinstance(from_spec, type(direct))
        assert direct.img_size == from_spec.img_size
        assert direct.patch_size == from_spec.patch_size
        assert direct.num_patches == from_spec.num_patches
        assert direct.proj.in_channels == from_spec.proj.in_channels
        assert direct.proj.out_channels == from_spec.proj.out_channels
        assert direct.proj.kernel_size == from_spec.proj.kernel_size
        assert direct.proj.stride == from_spec.proj.stride

    def test_cls_token_equivalence(self):
        """CLSTokenSpec should produce correct CLS token parameter."""
        embed_dim = 768
        spec = CLSTokenSpec()
        ctx = Context(dimensions={"embed_dim": embed_dim})
        from_spec = realise(spec, ctx)
        assert hasattr(from_spec, "cls_token")
        assert from_spec.cls_token.shape == (1, 1, embed_dim)

    def test_positional_embedding_equivalence(self):
        """PosEmbedSpec should produce identical PosEmbed2D."""
        num_patches = 196
        embed_dim = 768
        direct = PosEmbed2D(
            num_patches=num_patches,
            embed_dim=embed_dim,
            cls_token=True,
            dropout=0.0,
        )
        spec = PosEmbedSpec(include_cls=True, init_std=0.02)
        ctx = Context(
            dimensions={"num_patches": num_patches + 1, "embed_dim": embed_dim}
        )
        from_spec = realise(spec, ctx)
        assert isinstance(from_spec, type(direct))
        assert direct.pos_embed.shape == from_spec.pos_embed.shape

    def test_mhea_equivalence(self):
        """MHEASpec should produce identical MultiheadEnergyAttention."""
        direct = MultiheadEnergyAttention(
            embed_dim=768,
            num_heads=12,
            beta=None,
            init_std=0.002,
        )
        spec = MHEASpec(
            num_heads=12,
            beta=None,
            init_std=0.002,
        )
        ctx = Context(dimensions={"embed_dim": 768})
        from_spec = realise(spec, ctx)
        assert isinstance(from_spec, type(direct))
        assert direct.num_heads == from_spec.num_heads
        assert direct.embed_dim == from_spec.embed_dim
        assert direct.k_proj_weight.shape == from_spec.k_proj_weight.shape
        assert direct.q_proj_weight.shape == from_spec.q_proj_weight.shape

    def test_hopfield_network_equivalence(self):
        """HNSpec should produce identical HopfieldNetwork."""
        direct = HopfieldNetwork(768, hidden_dim=3072)
        spec = HNSpec(hidden_dim=3072)
        ctx = Context(dimensions={"embed_dim": 768})
        from_spec = realise(spec, ctx)
        assert isinstance(from_spec, type(direct))
        assert direct.embed_dim == from_spec.embed_dim
        assert direct.hidden_dim == from_spec.hidden_dim
        assert direct.kernel.shape == from_spec.kernel.shape
        spec2 = HNSpec(multiplier=4.0)
        from_spec2 = realise(spec2, ctx)
        assert from_spec2.hidden_dim == 3072

    def test_layer_norm_equivalence(self):
        """LayerNormSpec should produce identical EnergyLayerNorm."""
        spec = LayerNormSpec(eps=1e-5)
        ctx = Context(dimensions={"embed_dim": 768})
        from_spec = realise(spec, ctx)
        assert isinstance(from_spec, EnergyLayerNorm)

    def test_classification_head_equivalence(self):
        """ClassificationHeadSpec should produce identical ClassifierHead."""
        direct = ClassifierHead(
            in_features=768,
            num_classes=1000,
            pool_type="token",
            drop_rate=0.0,
        )
        spec = ClassificationHeadSpec(
            num_classes=1000,
            pool_type="token",
            drop_rate=0.0,
        )
        ctx = Context(dimensions={"embed_dim": 768})
        from_spec = realise(spec, ctx)
        assert isinstance(from_spec, type(direct))
        assert from_spec.pool_type == direct.pool_type
        assert from_spec.fc.out_features == direct.fc.out_features


class TestETBlockEquivalence:
    """Test Energy Transformer block construction."""

    def test_et_block_construction(self):
        """ETBlockSpec should produce identical EnergyTransformer."""
        embed_dim = 768
        direct = EnergyTransformer(
            layer_norm=EnergyLayerNorm(embed_dim),
            attention=MultiheadEnergyAttention(
                embed_dim=embed_dim,
                num_heads=12,
            ),
            hopfield=HopfieldNetwork(embed_dim, hidden_dim=3072),
            steps=4,
            optimizer=SGD(alpha=0.125),
        )
        spec = ETBlockSpec(
            steps=4,
            alpha=0.125,
            layer_norm=LayerNormSpec(),
            attention=MHEASpec(num_heads=12, head_dim=64),
            hopfield=HNSpec(hidden_dim=3072),
        )
        ctx = Context(dimensions={"embed_dim": embed_dim})
        from_spec = realise(spec, ctx)
        assert isinstance(from_spec, type(direct))
        assert direct.steps == from_spec.steps
        assert direct.alpha == from_spec.alpha
        assert isinstance(from_spec.attention, type(direct.attention))
        assert isinstance(from_spec.hopfield, type(direct.hopfield))


class TestFullModelEquivalence:
    """Test complete model construction matches factory functions."""

    def test_viet_base_construction(self):
        """Test ViET-Base can be constructed via specs."""
        img_size = 224
        patch_size = 16
        num_classes = 1000
        embed_dim = 768
        depth = 12
        num_heads = 12
        head_dim = 64
        hopfield_hidden_dim = 3072
        et_steps = 4
        et_alpha = 0.125
        model_spec = seq(
            PatchEmbedSpec(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=3,
                embed_dim=embed_dim,
            ),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            loop(
                ETBlockSpec(
                    steps=et_steps,
                    alpha=et_alpha,
                    attention=MHEASpec(num_heads=num_heads, head_dim=head_dim),
                    hopfield=HNSpec(hidden_dim=hopfield_hidden_dim),
                ),
                times=depth,
            ),
            LayerNormSpec(),
            ClassificationHeadSpec(num_classes=num_classes, pool_type="token"),
        )
        spec_model = realise(model_spec)
        direct_model = viet_base(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=num_classes,
        )
        torch.manual_seed(42)
        x = torch.randn(2, 3, img_size, img_size)
        spec_out = spec_model(x)
        direct_out = direct_model(x)
        assert spec_out.shape == direct_out.shape == (2, num_classes)

    def test_viet_cifar_construction(self):
        """Test ViET-CIFAR can be constructed via specs."""
        model_spec = seq(
            PatchEmbedSpec(
                img_size=32, patch_size=4, in_chans=3, embed_dim=192
            ),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            loop(
                ETBlockSpec(
                    steps=6,
                    alpha=10.0,
                    attention=MHEASpec(num_heads=8, head_dim=64),
                    hopfield=HNSpec(multiplier=3.0),
                ),
                times=2,
            ),
            LayerNormSpec(),
            ClassificationHeadSpec(num_classes=100, pool_type="token"),
        )
        spec_model = realise(model_spec)
        direct_model = viet_2l_cifar(num_classes=100)
        x = torch.randn(2, 3, 32, 32)
        assert spec_model(x).shape == direct_model(x).shape

    def test_viset_construction(self):
        """Test ViSET can be constructed via specs."""
        model_spec = seq(
            PatchEmbedSpec(
                img_size=32, patch_size=4, in_chans=3, embed_dim=192
            ),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            loop(
                ETBlockSpec(
                    steps=6,
                    alpha=10.0,
                    attention=MHEASpec(num_heads=8, head_dim=64),
                    hopfield=SHNSpec(
                        coordinates=[
                            (i, j) for i in range(8) for j in range(8)
                        ],
                        max_dim=2,
                        budget=0.2,
                        dim_weights={1: 0.5, 2: 0.5},
                        num_vertices=64,
                        multiplier=3.0,
                    ),
                ),
                times=2,
            ),
            LayerNormSpec(),
            ClassificationHeadSpec(num_classes=100, pool_type="token"),
        )
        spec_model = realise(model_spec)
        direct_model = viset_2l_e50_t50_cifar(num_classes=100)
        x = torch.randn(2, 3, 32, 32)
        assert spec_model(x).shape == direct_model(x).shape


class TestModelBehavior:
    """Test that spec-built models exhibit expected behavior."""

    def test_energy_minimization_occurs(self):
        """Verify ET blocks perform energy minimization."""
        spec = ETBlockSpec(
            steps=4,
            alpha=0.1,
            attention=MHEASpec(num_heads=4, head_dim=32),
            hopfield=HNSpec(multiplier=2.0),
        )
        ctx = Context(dimensions={"embed_dim": 128})
        et_block = realise(spec, ctx)
        x = torch.randn(2, 10, 128)
        initial_energy = et_block._compute_energy(x.clone())
        out = et_block(x)
        final_energy = et_block._compute_energy(out.clone())
        assert final_energy < initial_energy

    def test_attention_affects_output(self):
        """Verify attention mechanism works."""
        spec = MHEASpec(num_heads=4, head_dim=32)
        ctx = Context(dimensions={"embed_dim": 128})
        attn = realise(spec, ctx)
        x = torch.randn(1, 10, 128)
        x[:, 0, :] = 10.0
        with torch.no_grad():
            energy = attn(x)
            assert energy.shape == ()
            assert energy.dtype == torch.float32


class TestErrorCases:
    """Test error handling in spec realisation."""

    def test_missing_dimension_error(self):
        """Test helpful error when required dimension is missing."""
        spec = MHEASpec(num_heads=8, head_dim=64)
        ctx = Context()
        with pytest.raises(ValidationError) as exc_info:
            realise(spec, ctx)
        assert "embed_dim" in str(exc_info.value).lower()

    def test_incompatible_dimensions(self):
        """Test error when dimensions don't match."""
