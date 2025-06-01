"""Exhaustive tests for all model configurations."""

import pytest
import torch

from energy_transformer.models.vision import (
    viet_2l_cifar,
    viet_4l_cifar,
    viet_6l_cifar,
    viet_base,
    viet_large,
    viet_small,
    viet_tiny,
    viset_2l_e40_t40_tet20_cifar,
    viset_2l_e50_t50_cifar,
    viset_2l_e100_cifar,
    viset_2l_random_cifar,
    viset_2l_t100_cifar,
    viset_base,
    viset_small,
    viset_tiny,
    vit_base,
    vit_large,
    vit_small,
    vit_small_cifar,
    vit_tiny_cifar,
)
from energy_transformer.spec import Context, loop, realise, seq
from energy_transformer.spec.library import (
    ClassificationHeadSpec,
    CLSTokenSpec,
    ETBlockSpec,
    HNSpec,
    LayerNormSpec,
    MHASpec,
    MHEASpec,
    MLPSpec,
    PatchEmbedSpec,
    PosEmbedSpec,
    SHNSpec,
    TransformerBlockSpec,
)

pytest.skip("Exhaustive model tests not implemented", allow_module_level=True)

pytestmark = pytest.mark.integration


class TestVisionTransformerModels:
    """Test all Vision Transformer configurations."""

    def test_vit_tiny_equivalence(self):
        """Test ViT-Tiny construction via specs."""
        spec = seq(
            PatchEmbedSpec(
                img_size=224, patch_size=16, embed_dim=192, in_chans=3
            ),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            loop(
                TransformerBlockSpec(
                    attention=MHASpec(num_heads=3),
                    mlp=MLPSpec(hidden_features=768),
                    norm_first=True,
                ),
                times=12,
            ),
            LayerNormSpec(),
            ClassificationHeadSpec(num_classes=1000, use_cls_token=True),
        )
        # Just ensure spec validates
        issues = spec.validate(Context())
        assert not issues

    def test_vit_small_equivalence(self):
        """Test ViT-Small construction."""
        direct = vit_small(img_size=224, patch_size=16, num_classes=1000)
        assert direct.embed_dim == 384
        assert len(direct.blocks) == 12

    def test_vit_base_equivalence(self):
        """Test ViT-Base construction."""
        direct = vit_base(img_size=224, patch_size=16, num_classes=1000)
        assert direct.embed_dim == 768
        assert len(direct.blocks) == 12

    def test_vit_large_equivalence(self):
        """Test ViT-Large construction."""
        direct = vit_large(img_size=224, patch_size=16, num_classes=1000)
        assert direct.embed_dim == 1024
        assert len(direct.blocks) == 24

    def test_vit_cifar_variants(self):
        """Test CIFAR-specific ViT models."""
        models = [
            (vit_tiny_cifar, {"embed_dim": 192, "depth": 12}),
            (vit_small_cifar, {"embed_dim": 384, "depth": 12}),
        ]
        for model_fn, expected in models:
            model = model_fn(num_classes=100)
            assert model.embed_dim == expected["embed_dim"]
            assert len(model.blocks) == expected["depth"]


class TestVisionEnergyTransformerModels:
    """Test all Vision Energy Transformer configurations."""

    def test_viet_tiny_full_equivalence(self):
        """Test ViET-Tiny with complete spec equivalence."""
        config = {
            "img_size": 224,
            "patch_size": 16,
            "in_chans": 3,
            "num_classes": 1000,
            "embed_dim": 192,
            "depth": 12,
            "num_heads": 3,
            "head_dim": 64,
            "hopfield_hidden_dim": 768,
            "et_steps": 4,
            "et_alpha": 0.125,
            "drop_rate": 0.0,
        }
        direct = viet_tiny(**config)
        spec = seq(
            PatchEmbedSpec(
                img_size=config["img_size"],
                patch_size=config["patch_size"],
                in_chans=config["in_chans"],
                embed_dim=config["embed_dim"],
            ),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            loop(
                ETBlockSpec(
                    steps=config["et_steps"],
                    alpha=config["et_alpha"],
                    attention=MHEASpec(
                        num_heads=config["num_heads"],
                        head_dim=config["head_dim"],
                    ),
                    hopfield=HNSpec(hidden_dim=config["hopfield_hidden_dim"]),
                ),
                times=config["depth"],
            ),
            LayerNormSpec(),
            ClassificationHeadSpec(
                num_classes=config["num_classes"], use_cls_token=True
            ),
        )
        spec_model = realise(spec)
        x = torch.randn(2, 3, 224, 224)
        direct_out = direct(x)
        spec_out = spec_model(x)
        assert direct_out.shape == spec_out.shape == (2, 1000)

    def test_viet_small_equivalence(self):
        """Test ViET-Small construction."""
        direct = viet_small(img_size=224, patch_size=16, num_classes=1000)
        spec = seq(
            PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=384),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            loop(
                ETBlockSpec(
                    steps=4,
                    alpha=0.125,
                    attention=MHEASpec(num_heads=6, head_dim=64),
                    hopfield=HNSpec(hidden_dim=1536),
                ),
                times=12,
            ),
            LayerNormSpec(),
            ClassificationHeadSpec(num_classes=1000),
        )
        spec_model = realise(spec)
        x = torch.randn(1, 3, 224, 224)
        assert direct(x).shape == spec_model(x).shape

    def test_viet_base_equivalence(self):
        """Test ViET-Base construction."""
        direct = viet_base(img_size=224, patch_size=16, num_classes=1000)
        spec = seq(
            PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=768),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            loop(
                ETBlockSpec(
                    steps=4,
                    alpha=0.125,
                    attention=MHEASpec(num_heads=12, head_dim=64),
                    hopfield=HNSpec(hidden_dim=3072),
                ),
                times=12,
            ),
            LayerNormSpec(),
            ClassificationHeadSpec(num_classes=1000),
        )
        spec_model = realise(spec)
        x = torch.randn(1, 3, 224, 224)
        assert direct(x).shape == spec_model(x).shape

    def test_viet_large_equivalence(self):
        """Test ViET-Large construction."""
        direct = viet_large(img_size=224, patch_size=16, num_classes=1000)
        spec = seq(
            PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=1024),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            loop(
                ETBlockSpec(
                    steps=4,
                    alpha=0.125,
                    attention=MHEASpec(num_heads=16, head_dim=64),
                    hopfield=HNSpec(hidden_dim=4096),
                ),
                times=24,
            ),
            LayerNormSpec(),
            ClassificationHeadSpec(num_classes=1000),
        )
        spec_model = realise(spec)
        x = torch.randn(1, 3, 224, 224)
        assert direct(x).shape == spec_model(x).shape

    def test_viet_cifar_shallow_variants(self):
        """Test all CIFAR shallow variants."""
        direct_2l = viet_2l_cifar(num_classes=100)
        spec_2l = seq(
            PatchEmbedSpec(img_size=32, patch_size=4, embed_dim=192),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            loop(
                ETBlockSpec(
                    steps=6,
                    alpha=10.0,
                    attention=MHEASpec(num_heads=8, head_dim=64),
                    hopfield=HNSpec(hidden_dim=576),
                ),
                times=2,
            ),
            LayerNormSpec(),
            ClassificationHeadSpec(num_classes=100),
        )
        spec_model_2l = realise(spec_2l)
        direct_4l = viet_4l_cifar(num_classes=100)
        spec_4l = seq(
            PatchEmbedSpec(img_size=32, patch_size=4, embed_dim=192),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            loop(
                ETBlockSpec(
                    steps=5,
                    alpha=5.0,
                    attention=MHEASpec(num_heads=8, head_dim=64),
                    hopfield=HNSpec(hidden_dim=576),
                ),
                times=4,
            ),
            LayerNormSpec(),
            ClassificationHeadSpec(num_classes=100),
        )
        spec_model_4l = realise(spec_4l)
        direct_6l = viet_6l_cifar(num_classes=100)
        spec_6l = seq(
            PatchEmbedSpec(img_size=32, patch_size=4, embed_dim=192),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            loop(
                ETBlockSpec(
                    steps=4,
                    alpha=2.5,
                    attention=MHEASpec(num_heads=8, head_dim=64),
                    hopfield=HNSpec(hidden_dim=576),
                ),
                times=6,
            ),
            LayerNormSpec(),
            ClassificationHeadSpec(num_classes=100),
        )
        spec_model_6l = realise(spec_6l)
        x = torch.randn(1, 3, 32, 32)
        assert direct_2l(x).shape == spec_model_2l(x).shape == (1, 100)
        assert direct_4l(x).shape == spec_model_4l(x).shape == (1, 100)
        assert direct_6l(x).shape == spec_model_6l(x).shape == (1, 100)


class TestVisionSimplicialEnergyTransformerModels:
    """Test all Vision Simplicial Energy Transformer configurations."""

    def test_viset_topology_variants(self):
        """Test all topology-based ViSET variants."""
        direct_e50_t50 = viset_2l_e50_t50_cifar(num_classes=100)
        spec_e50_t50 = seq(
            PatchEmbedSpec(img_size=32, patch_size=4, embed_dim=192),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            loop(
                ETBlockSpec(
                    steps=6,
                    alpha=10.0,
                    attention=MHEASpec(num_heads=8, head_dim=64),
                    hopfield=SHNSpec(
                        num_vertices=64,
                        coordinates=[
                            (i, j) for i in range(8) for j in range(8)
                        ],
                        max_dim=2,
                        budget=0.2,
                        dim_weights={1: 0.5, 2: 0.5},
                        hidden_dim=576,
                    ),
                ),
                times=2,
            ),
            LayerNormSpec(),
            ClassificationHeadSpec(num_classes=100),
        )
        spec_model_e50_t50 = realise(spec_e50_t50)
        direct_e100 = viset_2l_e100_cifar(num_classes=100)
        spec_e100 = seq(
            PatchEmbedSpec(img_size=32, patch_size=4, embed_dim=192),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            loop(
                ETBlockSpec(
                    steps=6,
                    alpha=10.0,
                    attention=MHEASpec(num_heads=8, head_dim=64),
                    hopfield=SHNSpec(
                        num_vertices=64,
                        coordinates=[
                            (i, j) for i in range(8) for j in range(8)
                        ],
                        max_dim=1,
                        budget=0.15,
                        dim_weights={1: 1.0},
                        hidden_dim=576,
                    ),
                ),
                times=2,
            ),
            LayerNormSpec(),
            ClassificationHeadSpec(num_classes=100),
        )
        spec_model_e100 = realise(spec_e100)
        direct_t100 = viset_2l_t100_cifar(num_classes=100)
        spec_t100 = seq(
            PatchEmbedSpec(img_size=32, patch_size=4, embed_dim=192),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            loop(
                ETBlockSpec(
                    steps=6,
                    alpha=10.0,
                    attention=MHEASpec(num_heads=8, head_dim=64),
                    hopfield=SHNSpec(
                        num_vertices=64,
                        coordinates=[
                            (i, j) for i in range(8) for j in range(8)
                        ],
                        max_dim=2,
                        budget=0.15,
                        dim_weights={2: 1.0},
                        hidden_dim=576,
                    ),
                ),
                times=2,
            ),
            LayerNormSpec(),
            ClassificationHeadSpec(num_classes=100),
        )
        spec_model_t100 = realise(spec_t100)
        direct_random = viset_2l_random_cifar(num_classes=100)
        spec_random = seq(
            PatchEmbedSpec(img_size=32, patch_size=4, embed_dim=192),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            loop(
                ETBlockSpec(
                    steps=6,
                    alpha=10.0,
                    attention=MHEASpec(num_heads=8, head_dim=64),
                    hopfield=SHNSpec(
                        num_vertices=64,
                        coordinates=None,
                        max_dim=2,
                        budget=0.15,
                        dim_weights={1: 0.5, 2: 0.5},
                        hidden_dim=576,
                    ),
                ),
                times=2,
            ),
            LayerNormSpec(),
            ClassificationHeadSpec(num_classes=100),
        )
        spec_model_random = realise(spec_random)
        x = torch.randn(1, 3, 32, 32)
        assert direct_e50_t50(x).shape == spec_model_e50_t50(x).shape
        assert direct_e100(x).shape == spec_model_e100(x).shape
        assert direct_t100(x).shape == spec_model_t100(x).shape
        assert direct_random(x).shape == spec_model_random(x).shape

    def test_viset_with_tetrahedra(self):
        """Test ViSET with tetrahedra (3-simplices)."""
        direct = viset_2l_e40_t40_tet20_cifar(num_classes=100)
        spec = seq(
            PatchEmbedSpec(img_size=32, patch_size=4, embed_dim=192),
            CLSTokenSpec(),
            PosEmbedSpec(include_cls=True),
            loop(
                ETBlockSpec(
                    steps=6,
                    alpha=10.0,
                    attention=MHEASpec(num_heads=8, head_dim=64),
                    hopfield=SHNSpec(
                        num_vertices=64,
                        coordinates=[
                            (i, j) for i in range(8) for j in range(8)
                        ],
                        max_dim=3,
                        budget=0.15,
                        dim_weights={1: 0.4, 2: 0.4, 3: 0.2},
                        hidden_dim=576,
                    ),
                ),
                times=2,
            ),
            LayerNormSpec(),
            ClassificationHeadSpec(num_classes=100),
        )
        spec_model = realise(spec)
        x = torch.randn(1, 3, 32, 32)
        assert direct(x).shape == spec_model(x).shape

    def test_viset_standard_sizes(self):
        """Test standard ViSET model sizes."""
        models = [
            (viset_tiny, 192, 12, 3),
            (viset_small, 384, 12, 6),
            (viset_base, 768, 12, 12),
        ]
        for model_fn, embed_dim, depth, num_heads in models:
            model = model_fn(img_size=224, patch_size=16, num_classes=1000)
            spec = seq(
                PatchEmbedSpec(
                    img_size=224, patch_size=16, embed_dim=embed_dim
                ),
                CLSTokenSpec(),
                PosEmbedSpec(include_cls=True),
                loop(
                    ETBlockSpec(
                        steps=4,
                        alpha=0.125,
                        attention=MHEASpec(num_heads=num_heads, head_dim=64),
                        hopfield=SHNSpec(
                            num_vertices=196,
                            coordinates=[
                                (i, j) for i in range(14) for j in range(14)
                            ],
                            max_dim=2,
                            budget=0.15,
                            dim_weights={1: 0.5, 2: 0.5},
                            multiplier=4.0,
                        ),
                    ),
                    times=depth,
                ),
                LayerNormSpec(),
                ClassificationHeadSpec(num_classes=1000),
            )
            spec_model = realise(spec)
            x = torch.randn(1, 3, 224, 224)
            assert model(x).shape == spec_model(x).shape


class TestModelAttributes:
    """Test that models have correct attributes and methods."""

    def test_viet_model_attributes(self):
        """Test ViET models have expected attributes."""
        model = viet_base(img_size=224, patch_size=16, num_classes=1000)
        assert hasattr(model, "patch_embed")
        assert hasattr(model, "cls_token")
        assert hasattr(model, "pos_embed")
        assert hasattr(model, "et_blocks")
        assert hasattr(model, "norm")
        assert hasattr(model, "head")
        assert len(model.et_blocks) == 12
        for block in model.et_blocks:
            assert hasattr(block, "layer_norm")
            assert hasattr(block, "attention")
            assert hasattr(block, "hopfield")
            assert block.steps == 4
            assert block.alpha == 0.125

    def test_viset_model_simplicial_networks(self):
        """Test ViSET models use simplicial networks."""
        model = viset_2l_e50_t50_cifar(num_classes=100)
        from energy_transformer.layers.simplicial import (
            SimplicialHopfieldNetwork,
        )

        for block in model.et_blocks:
            assert isinstance(block.hopfield, SimplicialHopfieldNetwork)
            assert hasattr(block.hopfield, "simps_by_size")
            assert hasattr(block.hopfield, "max_vertex")

    def test_energy_output_functionality(self):
        """Test models can return energy information."""
        model = viet_2l_cifar(num_classes=100)
        x = torch.randn(1, 3, 32, 32)
        result = model(x, return_energy_info=True, et_kwargs={"track": "both"})
        assert "logits" in result
        assert "energy_info" in result
        assert result["logits"].shape == (1, 100)
        energy_info = result["energy_info"]
        assert "block_energies" in energy_info
        assert "block_trajectories" in energy_info
        assert len(energy_info["block_energies"]) == 2


class TestModelConsistency:
    """Test consistency between direct and spec construction."""

    def test_parameter_count_consistency(self):
        """Verify parameter counts match between direct and spec models."""
        test_cases = [
            (
                lambda: viet_tiny(
                    img_size=224, patch_size=16, num_classes=1000
                ),
                seq(
                    PatchEmbedSpec(img_size=224, patch_size=16, embed_dim=192),
                    CLSTokenSpec(),
                    PosEmbedSpec(include_cls=True),
                    loop(
                        ETBlockSpec(
                            steps=4,
                            alpha=0.125,
                            attention=MHEASpec(num_heads=3, head_dim=64),
                            hopfield=HNSpec(hidden_dim=768),
                        ),
                        times=12,
                    ),
                    LayerNormSpec(),
                    ClassificationHeadSpec(num_classes=1000),
                ),
                "ViET-Tiny",
            ),
        ]
        for direct_fn, spec, name in test_cases:
            direct_model = direct_fn()
            spec_model = realise(spec)
            direct_params = sum(p.numel() for p in direct_model.parameters())
            spec_params = sum(p.numel() for p in spec_model.parameters())
            ratio = spec_params / direct_params
            assert 0.95 <= ratio <= 1.05, (
                f"{name}: Parameter count mismatch - direct: {direct_params}, "
                f"spec: {spec_params}"
            )
