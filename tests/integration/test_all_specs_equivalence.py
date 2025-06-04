"""Exhaustive equivalence tests for all specifications in library.py."""

import math

import pytest
import torch
from torch import nn

from energy_transformer.layers import (
    ClassificationHead,
    CLSToken,
    FeatureHead,
    HopfieldNetwork,
    LayerNorm,
    MultiheadEnergyAttention,
    PatchEmbedding,
    PositionalEmbedding2D,
)
from energy_transformer.layers.simplicial import SimplicialHopfieldNetwork
from energy_transformer.models import EnergyTransformer
from energy_transformer.spec import Context, realise
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
)
from energy_transformer.spec.primitives import ValidationError

pytestmark = pytest.mark.integration


class TestLayerSpecs:
    """Test all layer specifications."""

    def test_layer_norm_all_variants(self):
        """Test LayerNormSpec with various configurations."""
        test_cases = [
            {"eps": 1e-5, "embed_dim": 768},
            {"eps": 1e-6, "embed_dim": 512},
            {"eps": 1e-4, "embed_dim": 256},
            {"eps": 1e-3, "embed_dim": 1024},
        ]

        for tc in test_cases:
            direct = LayerNorm(in_dim=tc["embed_dim"], eps=tc["eps"])
            spec = LayerNormSpec(eps=tc["eps"])
            ctx = Context(dimensions={"embed_dim": tc["embed_dim"]})
            from_spec = realise(spec, ctx)

            assert isinstance(from_spec, LayerNorm)
            assert type(from_spec) is type(direct)
            assert from_spec.in_dim == direct.in_dim
            assert from_spec.eps == direct.eps

            x = torch.randn(2, 10, tc["embed_dim"])
            assert direct(x).shape == from_spec(x).shape

    def test_patch_embed_all_variants(self):
        """Test PatchEmbedSpec with various image and patch sizes."""
        test_cases = [
            {
                "img_size": 224,
                "patch_size": 16,
                "in_chans": 3,
                "embed_dim": 768,
            },
            {
                "img_size": 224,
                "patch_size": 14,
                "in_chans": 3,
                "embed_dim": 1024,
            },
            {
                "img_size": 384,
                "patch_size": 16,
                "in_chans": 3,
                "embed_dim": 1024,
            },
            {"img_size": 32, "patch_size": 4, "in_chans": 3, "embed_dim": 192},
            {"img_size": 32, "patch_size": 2, "in_chans": 3, "embed_dim": 384},
            {
                "img_size": (224, 112),
                "patch_size": (16, 8),
                "in_chans": 3,
                "embed_dim": 768,
            },
            {
                "img_size": 224,
                "patch_size": 16,
                "in_chans": 1,
                "embed_dim": 768,
            },
            {
                "img_size": 224,
                "patch_size": 16,
                "in_chans": 4,
                "embed_dim": 768,
            },
            {
                "img_size": 224,
                "patch_size": 16,
                "in_chans": 3,
                "embed_dim": 768,
                "bias": False,
            },
        ]

        for tc in test_cases:
            bias = tc.get("bias", True)
            direct = PatchEmbedding(
                img_size=tc["img_size"],
                patch_size=tc["patch_size"],
                in_chans=tc["in_chans"],
                embed_dim=tc["embed_dim"],
                bias=bias,
            )
            spec = PatchEmbedSpec(
                img_size=tc["img_size"],
                patch_size=tc["patch_size"],
                in_chans=tc["in_chans"],
                embed_dim=tc["embed_dim"],
                bias=bias,
            )
            from_spec = realise(spec)

            assert isinstance(from_spec, PatchEmbedding)
            assert from_spec.num_patches == direct.num_patches
            assert (from_spec.proj.bias is not None) == bias

    def test_cls_token_context_variations(self):
        """Test CLSTokenSpec with different embedding dimensions."""
        for embed_dim in [192, 384, 768, 1024, 1280]:
            direct = CLSToken(embed_dim)
            spec = CLSTokenSpec()
            ctx = Context(dimensions={"embed_dim": embed_dim})
            from_spec = realise(spec, ctx)

            assert isinstance(from_spec, CLSToken)
            assert from_spec.cls_token.shape[2] == embed_dim
            assert from_spec.cls_token.shape == direct.cls_token.shape

    def test_pos_embed_all_configurations(self):
        """Test PosEmbedSpec with various settings."""
        test_cases = [
            {
                "num_patches": 196,
                "embed_dim": 768,
                "include_cls": True,
                "init_std": 0.02,
            },
            {
                "num_patches": 196,
                "embed_dim": 768,
                "include_cls": False,
                "init_std": 0.02,
            },
            {
                "num_patches": 64,
                "embed_dim": 192,
                "include_cls": True,
                "init_std": 0.01,
            },
            {
                "num_patches": 1024,
                "embed_dim": 1024,
                "include_cls": True,
                "init_std": 0.02,
            },
            {
                "num_patches": 256,
                "embed_dim": 512,
                "include_cls": False,
                "init_std": 0.005,
            },
        ]

        for tc in test_cases:
            num_patches_for_module = tc["num_patches"]
            num_patches_for_context = tc["num_patches"] + (
                1 if tc["include_cls"] else 0
            )

            direct = PositionalEmbedding2D(
                num_patches=num_patches_for_module,
                embed_dim=tc["embed_dim"],
                include_cls=tc["include_cls"],
                init_std=tc["init_std"],
            )
            spec = PosEmbedSpec(
                include_cls=tc["include_cls"], init_std=tc["init_std"]
            )
            ctx = Context(
                dimensions={
                    "num_patches": num_patches_for_context,
                    "embed_dim": tc["embed_dim"],
                }
            )
            from_spec = realise(spec, ctx)

            assert isinstance(from_spec, PositionalEmbedding2D)
            assert from_spec.pos_embed.shape == direct.pos_embed.shape

    def test_dropout_spec(self):
        """Test DropoutSpec creates proper nn.Dropout."""
        test_cases = [
            {"p": 0.0, "inplace": False},
            {"p": 0.1, "inplace": False},
            {"p": 0.5, "inplace": True},
            {"p": 0.9, "inplace": False},
        ]

        for tc in test_cases:
            spec = DropoutSpec(p=tc["p"], inplace=tc["inplace"])
            module = realise(spec)

            assert isinstance(module, nn.Dropout)
            assert module.p == tc["p"]
            assert module.inplace == tc["inplace"]

    def test_identity_spec(self):
        """Test IdentitySpec creates nn.Identity."""
        spec = IdentitySpec()
        module = realise(spec)

        assert isinstance(module, nn.Identity)
        x = torch.randn(2, 10, 768)
        assert torch.equal(module(x), x)


class TestAttentionSpecs:
    """Test attention-related specifications."""

    def test_mhea_comprehensive(self):
        """Test MHEASpec with all parameter combinations."""
        test_cases = [
            {
                "in_dim": 768,
                "num_heads": 12,
                "head_dim": 64,
                "beta": None,
                "bias": False,
                "init_std": 0.002,
            },
            {
                "in_dim": 512,
                "num_heads": 8,
                "head_dim": 64,
                "beta": None,
                "bias": True,
                "init_std": 0.002,
            },
            {
                "in_dim": 1024,
                "num_heads": 16,
                "head_dim": 64,
                "beta": 0.5,
                "bias": False,
                "init_std": 0.001,
            },
            {
                "in_dim": 192,
                "num_heads": 3,
                "head_dim": 64,
                "beta": None,
                "bias": False,
                "init_std": 0.002,
            },
            {
                "in_dim": 384,
                "num_heads": 6,
                "head_dim": 64,
                "beta": None,
                "bias": False,
                "init_std": 0.002,
            },
            {
                "in_dim": 768,
                "num_heads": 12,
                "head_dim": 32,
                "beta": None,
                "bias": False,
                "init_std": 0.002,
            },
            {
                "in_dim": 768,
                "num_heads": 8,
                "head_dim": 96,
                "beta": None,
                "bias": False,
                "init_std": 0.002,
            },
        ]

        for tc in test_cases:
            direct = MultiheadEnergyAttention(
                embed_dim=tc["in_dim"],
                num_heads=tc["num_heads"],
                beta=tc["beta"],
                init_std=tc["init_std"],
            )
            spec = MHEASpec(
                num_heads=tc["num_heads"],
                head_dim=tc["head_dim"],
                beta=tc["beta"],
                init_std=tc["init_std"],
            )
            ctx = Context(dimensions={"embed_dim": tc["in_dim"]})
            from_spec = realise(spec, ctx)

            assert isinstance(from_spec, MultiheadEnergyAttention)
            assert from_spec.num_heads == direct.num_heads
            assert from_spec.head_dim == direct.head_dim
            if tc["beta"] is not None:
                assert torch.allclose(from_spec.beta, direct.beta)
            else:
                expected_beta = torch.full(
                    (tc["num_heads"],),
                    1.0 / math.sqrt(tc["in_dim"] // tc["num_heads"]),
                    device=from_spec.beta.device,
                    dtype=from_spec.beta.dtype,
                )
                assert torch.allclose(from_spec.beta, expected_beta)

    def test_mha_spec(self):
        """Test standard MHA spec (should create nn.MultiheadAttention)."""
        test_cases = [
            {
                "embed_dim": 768,
                "num_heads": 12,
                "qkv_bias": True,
                "attn_drop": 0.0,
                "proj_drop": 0.0,
            },
            {
                "embed_dim": 512,
                "num_heads": 8,
                "qkv_bias": False,
                "attn_drop": 0.1,
                "proj_drop": 0.1,
            },
            {
                "embed_dim": 1024,
                "num_heads": 16,
                "qkv_bias": True,
                "attn_drop": 0.0,
                "proj_drop": 0.2,
            },
        ]

        for tc in test_cases:
            spec = MHASpec(
                num_heads=tc["num_heads"],
                qkv_bias=tc["qkv_bias"],
                attn_drop=tc["attn_drop"],
                proj_drop=tc["proj_drop"],
            )
            ctx = Context(dimensions={"embed_dim": tc["embed_dim"]})
            module = realise(spec, ctx)

            assert (
                isinstance(module, nn.MultiheadAttention)
                or module.__class__.__name__ == "MultiheadAttention"
            )


class TestMemorySpecs:
    """Test memory network specifications."""

    def test_hopfield_network_comprehensive(self):
        """Test HNSpec with all configurations."""
        test_cases = [
            {"in_dim": 768, "hidden_dim": 3072},
            {"in_dim": 512, "hidden_dim": 2048},
            {"in_dim": 256, "hidden_dim": 1024},
            {"in_dim": 768, "multiplier": 4.0},
            {"in_dim": 512, "multiplier": 3.0},
            {"in_dim": 384, "multiplier": 5.0},
            {"in_dim": 192, "multiplier": 2.0},
            {"in_dim": 768, "hidden_dim": 3072, "energy_fn": "relu_squared"},
            {"in_dim": 768, "hidden_dim": 3072, "energy_fn": "softmax"},
            {"in_dim": 768, "hidden_dim": 3072, "energy_fn": "tanh"},
        ]

        for tc in test_cases:
            if "hidden_dim" in tc:
                expected_hidden = tc["hidden_dim"]
            else:
                expected_hidden = int(tc["in_dim"] * tc["multiplier"])

            direct = HopfieldNetwork(
                in_dim=tc["in_dim"], hidden_dim=expected_hidden
            )

            if "hidden_dim" in tc:
                spec = HNSpec(
                    hidden_dim=tc["hidden_dim"],
                    energy_fn=tc.get("energy_fn", "relu_squared"),
                )
            else:
                spec = HNSpec(
                    multiplier=tc["multiplier"],
                    energy_fn=tc.get("energy_fn", "relu_squared"),
                )

            ctx = Context(dimensions={"embed_dim": tc["in_dim"]})
            from_spec = realise(spec, ctx)

            assert isinstance(from_spec, HopfieldNetwork)
            assert from_spec.in_dim == direct.in_dim
            assert from_spec.hidden_dim == expected_hidden
            assert from_spec.ξ.shape == (expected_hidden, tc["in_dim"])

    def test_simplicial_hopfield_comprehensive(self):
        """Test SHNSpec with various topological configurations."""
        test_cases = [
            {
                "in_dim": 192,
                "num_vertices": 64,
                "coordinates": [(i, j) for i in range(8) for j in range(8)],
                "max_dim": 2,
                "budget": 0.1,
                "dim_weights": {1: 0.5, 2: 0.5},
                "hidden_dim": 768,
                "temperature": 0.5,
            },
            {
                "in_dim": 192,
                "num_vertices": 64,
                "coordinates": [(i, j) for i in range(8) for j in range(8)],
                "max_dim": 2,
                "budget": 0.3,
                "dim_weights": {1: 0.7, 2: 0.3},
                "multiplier": 3.0,
                "temperature": 1.0,
            },
            {
                "in_dim": 256,
                "num_vertices": 64,
                "coordinates": [(i, j) for i in range(8) for j in range(8)],
                "max_dim": 1,
                "budget": 0.15,
                "dim_weights": {1: 1.0},
                "hidden_dim": 1024,
                "temperature": 0.1,
            },
            {
                "in_dim": 384,
                "num_vertices": 64,
                "coordinates": [(i, j) for i in range(8) for j in range(8)],
                "max_dim": 2,
                "budget": 0.2,
                "dim_weights": {2: 1.0},
                "hidden_dim": 1536,
                "temperature": 0.5,
            },
            {
                "in_dim": 192,
                "num_vertices": 64,
                "coordinates": None,
                "max_dim": 2,
                "budget": 0.1,
                "dim_weights": {1: 0.5, 2: 0.5},
                "hidden_dim": 768,
                "temperature": 0.5,
            },
            {
                "in_dim": 192,
                "num_vertices": 64,
                "coordinates": [(i, j) for i in range(8) for j in range(8)],
                "max_dim": 3,
                "budget": 0.15,
                "dim_weights": {1: 0.4, 2: 0.4, 3: 0.2},
                "multiplier": 4.0,
                "temperature": 0.5,
            },
        ]

        for tc in test_cases:
            if "hidden_dim" in tc:
                expected_hidden = tc["hidden_dim"]
            else:
                expected_hidden = int(tc["in_dim"] * tc["multiplier"])

            spec_kwargs = {
                "num_vertices": tc["num_vertices"],
                "coordinates": tc["coordinates"],
                "max_dim": tc["max_dim"],
                "budget": tc["budget"],
                "dim_weights": tc["dim_weights"],
                "temperature": tc["temperature"],
            }

            if "hidden_dim" in tc:
                spec_kwargs["hidden_dim"] = tc["hidden_dim"]
            else:
                spec_kwargs["multiplier"] = tc["multiplier"]

            spec = SHNSpec(**spec_kwargs)
            ctx = Context(dimensions={"embed_dim": tc["in_dim"]})
            from_spec = realise(spec, ctx)

            assert isinstance(from_spec, SimplicialHopfieldNetwork)
            assert from_spec.in_dim == tc["in_dim"]
            assert from_spec.hidden_dim == expected_hidden
            assert tc["temperature"] == from_spec.T
            assert from_spec.max_vertex < tc["num_vertices"]


class TestHeadSpecs:
    """Test output head specifications."""

    def test_classification_head_all_variants(self):
        """Test ClassificationHeadSpec with all configurations."""
        test_cases = [
            {
                "embed_dim": 768,
                "num_classes": 1000,
                "representation_size": None,
                "drop_rate": 0.0,
                "use_cls_token": True,
            },
            {
                "embed_dim": 512,
                "num_classes": 100,
                "representation_size": None,
                "drop_rate": 0.1,
                "use_cls_token": True,
            },
            {
                "embed_dim": 1024,
                "num_classes": 21843,
                "representation_size": None,
                "drop_rate": 0.0,
                "use_cls_token": False,
            },
            {
                "embed_dim": 768,
                "num_classes": 1000,
                "representation_size": 1024,
                "drop_rate": 0.0,
                "use_cls_token": True,
            },
            {
                "embed_dim": 768,
                "num_classes": 1000,
                "representation_size": 512,
                "drop_rate": 0.2,
                "use_cls_token": True,
            },
            {
                "embed_dim": 192,
                "num_classes": 10,
                "representation_size": None,
                "drop_rate": 0.0,
                "use_cls_token": True,
            },
        ]

        for tc in test_cases:
            spec = ClassificationHeadSpec(
                num_classes=tc["num_classes"],
                representation_size=tc["representation_size"],
                drop_rate=tc["drop_rate"],
                use_cls_token=tc["use_cls_token"],
            )
            ctx = Context(dimensions={"embed_dim": tc["embed_dim"]})
            from_spec = realise(spec, ctx)

            assert isinstance(from_spec, ClassificationHead)
            assert from_spec.head.out_features == tc["num_classes"]
            assert from_spec.use_cls_token == tc["use_cls_token"]
            if tc["representation_size"] is not None:
                assert hasattr(from_spec, "pre_logits")
                assert isinstance(from_spec.pre_logits, nn.Sequential)

    def test_feature_head_spec(self):
        """Test FeatureHeadSpec."""
        test_cases = [{"use_cls_token": True}, {"use_cls_token": False}]

        for tc in test_cases:
            spec = FeatureHeadSpec(use_cls_token=tc["use_cls_token"])
            ctx = Context(dimensions={"embed_dim": 768})
            from_spec = realise(spec, ctx)

            assert isinstance(from_spec, FeatureHead)
            assert from_spec.use_cls_token == tc["use_cls_token"]


class TestCompositeSpecs:
    """Test composite specifications."""

    def test_et_block_all_configurations(self):
        """Test ETBlockSpec with various configurations."""
        test_cases = [
            {
                "embed_dim": 768,
                "steps": 4,
                "alpha": 0.125,
                "num_heads": 12,
                "head_dim": 64,
                "hidden_dim": 3072,
            },
            {
                "embed_dim": 512,
                "steps": 6,
                "alpha": 0.1,
                "num_heads": 8,
                "head_dim": 64,
                "hidden_dim": 2048,
            },
            {
                "embed_dim": 192,
                "steps": 10,
                "alpha": 10.0,
                "num_heads": 3,
                "head_dim": 64,
                "hidden_dim": 768,
            },
            {
                "embed_dim": 384,
                "steps": 5,
                "alpha": 1.0,
                "num_heads": 6,
                "head_dim": 64,
                "multiplier": 4.0,
            },
        ]

        for tc in test_cases:
            direct = EnergyTransformer(
                layer_norm=LayerNorm(tc["embed_dim"]),
                attention=MultiheadEnergyAttention(
                    embed_dim=tc["embed_dim"],
                    num_heads=tc["num_heads"],
                ),
                hopfield=HopfieldNetwork(
                    in_dim=tc["embed_dim"],
                    hidden_dim=tc.get(
                        "hidden_dim",
                        int(tc["embed_dim"] * tc.get("multiplier", 4.0)),
                    ),
                ),
                steps=tc["steps"],
                alpha=tc["alpha"],
            )
            hopfield_spec = (
                HNSpec(hidden_dim=tc.get("hidden_dim"))
                if "hidden_dim" in tc
                else HNSpec(multiplier=tc["multiplier"])
            )
            spec = ETBlockSpec(
                steps=tc["steps"],
                alpha=tc["alpha"],
                layer_norm=LayerNormSpec(),
                attention=MHEASpec(
                    num_heads=tc["num_heads"], head_dim=tc["head_dim"]
                ),
                hopfield=hopfield_spec,
            )
            ctx = Context(dimensions={"embed_dim": tc["embed_dim"]})
            from_spec = realise(spec, ctx)

            assert isinstance(from_spec, EnergyTransformer)
            assert from_spec.steps == direct.steps
            assert from_spec.alpha == direct.alpha
            x = torch.randn(2, 10, tc["embed_dim"])
            direct_out = direct(x)
            spec_out = from_spec(x)
            assert direct_out.shape == spec_out.shape

    def test_vision_embedding_spec(self):
        """Test VisionEmbeddingSpec composite."""
        test_cases = [
            {
                "img_size": 224,
                "patch_size": 16,
                "embed_dim": 768,
                "in_chans": 3,
                "use_cls_token": True,
                "drop_rate": 0.0,
            },
            {
                "img_size": 32,
                "patch_size": 4,
                "embed_dim": 192,
                "in_chans": 3,
                "use_cls_token": True,
                "drop_rate": 0.1,
            },
            {
                "img_size": 384,
                "patch_size": 32,
                "embed_dim": 1024,
                "in_chans": 3,
                "use_cls_token": False,
                "drop_rate": 0.0,
            },
        ]

        for tc in test_cases:
            spec = VisionEmbeddingSpec(
                img_size=tc["img_size"],
                patch_size=tc["patch_size"],
                embed_dim=tc["embed_dim"],
                in_chans=tc["in_chans"],
                use_cls_token=tc["use_cls_token"],
                drop_rate=tc["drop_rate"],
            )
            ctx = spec.apply_context(Context())

            assert ctx.get_dim("embed_dim") == tc["embed_dim"]
            expected_patches = (tc["img_size"] // tc["patch_size"]) ** 2
            if tc["use_cls_token"]:
                expected_patches += 1
            assert ctx.get_dim("num_patches") == expected_patches

    def test_transformer_block_spec(self):
        """Test standard TransformerBlockSpec."""
        test_cases = [
            {
                "embed_dim": 768,
                "num_heads": 12,
                "mlp_ratio": 4.0,
                "drop_path": 0.0,
                "norm_first": True,
            },
            {
                "embed_dim": 512,
                "num_heads": 8,
                "mlp_ratio": 3.0,
                "drop_path": 0.1,
                "norm_first": False,
            },
        ]

        for tc in test_cases:
            spec = TransformerBlockSpec(
                attention=MHASpec(num_heads=tc["num_heads"]),
                mlp=MLPSpec(
                    hidden_features=int(tc["embed_dim"] * tc["mlp_ratio"])
                ),
                drop_path=tc["drop_path"],
                norm_first=tc["norm_first"],
            )
            ctx = Context(dimensions={"embed_dim": tc["embed_dim"]})
            issues = spec.validate(ctx)
            assert len(issues) == 0

    def test_mlp_spec(self):
        """Test MLPSpec with various configurations."""
        test_cases = [
            {
                "embed_dim": 768,
                "hidden_features": 3072,
                "activation": "gelu",
                "drop": 0.0,
            },
            {
                "embed_dim": 512,
                "hidden_features": 2048,
                "activation": "relu",
                "drop": 0.1,
            },
            {
                "embed_dim": 768,
                "hidden_features": None,
                "activation": "swish",
                "drop": 0.0,
            },
        ]

        for tc in test_cases:
            spec = MLPSpec(
                hidden_features=tc["hidden_features"],
                activation=tc["activation"],
                drop=tc["drop"],
            )
            ctx = Context(dimensions={"embed_dim": tc["embed_dim"]})
            issues = spec.validate(ctx)
            assert len(issues) == 0


class TestParameterValidation:
    """Test parameter validation and error cases."""

    def test_dimension_constraints(self):
        """Test dimension validation constraints."""
        with pytest.raises(ValidationError):
            realise(
                PatchEmbedSpec(
                    img_size=224, patch_size=16, embed_dim=-768, in_chans=3
                )
            )

        spec = MHEASpec(num_heads=12, head_dim=64)
        ctx = Context()
        with pytest.raises(ValidationError) as exc_info:
            realise(spec, ctx)
        assert "embed_dim" in str(exc_info.value).lower()

    def test_parameter_ranges(self):
        """Test parameter range validation."""
        with pytest.raises(ValidationError):
            realise(DropoutSpec(p=1.5))

        with pytest.raises(ValidationError):
            realise(LayerNormSpec(eps=-1e-5))

    def test_missing_required_params(self):
        """Test missing required parameters."""
        with pytest.raises(ValidationError):
            ClassificationHeadSpec(num_classes=None)  # type: ignore
