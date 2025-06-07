import pytest
import torch
from torch import nn

from energy_transformer.layers.embeddings import (
    ConvPatchEmbed,
    PatchifyEmbed,
    PosEmbed2D,
    _to_pair,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("inp", "expected"), [(3, (3, 3)), ((2, 5), (2, 5))])
def test_to_pair(inp: int | tuple[int, int], expected: tuple[int, int]) -> None:
    assert _to_pair(inp) == expected


def test_patch_embedding_output_shape() -> None:
    patch = ConvPatchEmbed(img_size=4, patch_size=2, in_chans=3, embed_dim=8)
    x = torch.randn(1, 3, 4, 4)
    out = patch(x)
    assert out.shape == (1, 4, 8)


def test_patch_embedding_raises_for_wrong_size() -> None:
    patch = ConvPatchEmbed(img_size=4, patch_size=2, in_chans=3, embed_dim=8)
    x = torch.randn(1, 3, 5, 4)
    with pytest.raises(
        ValueError, match="ConvPatchEmbed: input dimension 2 mismatch"
    ):
        patch(x)


def test_cls_token_prepends_token() -> None:
    cls_token = nn.Parameter(torch.zeros(1, 1, 5))
    with torch.no_grad():
        cls_token.fill_(2.0)
    x = torch.zeros(2, 3, 5)
    cls_tokens = cls_token.expand(x.size(0), -1, -1)
    out = torch.cat([cls_tokens, x], dim=1)
    assert out.shape == (2, 4, 5)
    assert torch.allclose(out[:, 0], torch.full((2, 5), 2.0))
    assert torch.all(out[:, 1:] == 0)


def test_positional_embedding_adds_values() -> None:
    pos = PosEmbed2D(num_patches=4, embed_dim=2, cls_token=True)
    with torch.no_grad():
        pos.pos_embed.fill_(1.0)
    x = torch.zeros(1, 5, 2)
    out = pos(x)
    assert out.shape == (1, 5, 2)
    assert torch.allclose(out, torch.ones_like(out))


def test_patch_embedding_num_patches_and_conv_params() -> None:
    patch = ConvPatchEmbed(
        img_size=(4, 6),
        patch_size=(2, 3),
        in_chans=1,
        embed_dim=4,
    )
    assert patch.num_patches == 4
    assert patch.proj.kernel_size == (2, 3)
    assert patch.proj.stride == (2, 3)


def test_patch_embedding_non_square_image_and_patch() -> None:
    patch = ConvPatchEmbed(
        img_size=(4, 6),
        patch_size=(2, 3),
        in_chans=1,
        embed_dim=5,
    )
    x = torch.randn(2, 1, 4, 6)
    out = patch(x)
    assert out.shape == (2, 4, 5)


def test_patch_embedding_bias_optional() -> None:
    patch = ConvPatchEmbed(
        img_size=2,
        patch_size=1,
        in_chans=1,
        embed_dim=3,
        bias=False,
    )
    assert patch.proj.bias is None


def test_positional_embedding_parameter_shapes() -> None:
    pos_no_cls = PosEmbed2D(
        num_patches=3,
        embed_dim=2,
        cls_token=False,
    )
    pos_with_cls = PosEmbed2D(
        num_patches=3,
        embed_dim=2,
        cls_token=True,
    )
    assert pos_no_cls.pos_embed.shape == (3, 2)
    assert pos_with_cls.pos_embed.shape == (4, 2)


def test_positional_embedding_preserves_dtype() -> None:
    pos = PosEmbed2D(num_patches=2, embed_dim=3)
    x = torch.zeros(1, 2, 3, dtype=torch.float16)
    out = pos(x)
    assert out.dtype == torch.float16


def test_patchify_embed_roundtrip() -> None:
    embed = PatchifyEmbed(img_size=4, patch_size=2, in_chans=1, embed_dim=4)
    x = torch.randn(1, 1, 4, 4)
    patches = embed.patchify(x)
    assert patches.shape == (1, 4, 1, 2, 2)
    recon = embed.unpatchify(patches)
    assert torch.allclose(recon, x)


@pytest.mark.parametrize("batched", [True, False])
def test_positional_embedding_length_mismatch(batched: bool) -> None:
    pos = PosEmbed2D(num_patches=3, embed_dim=2)
    x = torch.zeros(1, 4, 2) if batched else torch.zeros(4, 2)
    with pytest.raises(ValueError, match="PosEmbed2D: input dimension"):
        pos(x)


def test_embedding_properties() -> None:
    conv = ConvPatchEmbed(
        img_size=(4, 6), patch_size=(2, 3), in_chans=1, embed_dim=4
    )
    assert conv.patch_area == 6
    assert conv.grid_shape == (2, 2)
    assert conv.sequence_length == 4
    assert conv.receptive_field == (2, 3)

    patchify = PatchifyEmbed(img_size=4, patch_size=2, in_chans=1, embed_dim=8)
    assert patchify.patch_area == 4
    assert patchify.tokens_per_image == 4
    total_pixels = 4 * 4 * 1
    total_features = 4 * 8
    assert patchify.compression_ratio == pytest.approx(
        total_pixels / total_features
    )

    pos = PosEmbed2D(num_patches=4, embed_dim=2, cls_token=True)
    assert pos.max_sequence_length == 5
    assert pos.has_cls_token
    assert pos.num_content_positions == 4
    assert isinstance(pos.device, torch.device)
