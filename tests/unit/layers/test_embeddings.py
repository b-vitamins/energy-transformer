import pytest
import torch

from energy_transformer.layers.embeddings import (
    PatchEmbedding,
    PositionalEmbedding2D,
    _to_pair,
)
from energy_transformer.layers.tokens import CLSToken


@pytest.mark.parametrize(("inp", "expected"), [(3, (3, 3)), ((2, 5), (2, 5))])
def test_to_pair(inp: int | tuple[int, int], expected: tuple[int, int]) -> None:
    assert _to_pair(inp) == expected


def test_patch_embedding_output_shape() -> None:
    patch = PatchEmbedding(img_size=4, patch_size=2, in_chans=3, embed_dim=8)
    x = torch.randn(1, 3, 4, 4)
    out = patch(x)
    assert out.shape == (1, 4, 8)


def test_patch_embedding_raises_for_wrong_size() -> None:
    patch = PatchEmbedding(img_size=4, patch_size=2, in_chans=3, embed_dim=8)
    x = torch.randn(1, 3, 5, 4)
    with pytest.raises(AssertionError):
        patch(x)


def test_cls_token_prepends_token() -> None:
    tok = CLSToken(embed_dim=5)
    with torch.no_grad():
        tok.cls_token.fill_(2.0)
    x = torch.zeros(2, 3, 5)
    out = tok(x)
    assert out.shape == (2, 4, 5)
    assert torch.allclose(out[:, 0], torch.full((2, 5), 2.0))
    assert torch.all(out[:, 1:] == 0)


def test_positional_embedding_adds_values() -> None:
    pos = PositionalEmbedding2D(num_patches=4, embed_dim=2, include_cls=True)
    with torch.no_grad():
        pos.pos_embed.fill_(1.0)
    x = torch.zeros(1, 5, 2)
    out = pos(x)
    assert out.shape == (1, 5, 2)
    assert torch.allclose(out, torch.ones_like(out))


def test_patch_embedding_num_patches_and_conv_params() -> None:
    patch = PatchEmbedding(
        img_size=(4, 6),
        patch_size=(2, 3),
        in_chans=1,
        embed_dim=4,
    )
    assert patch.num_patches == 4
    assert patch.proj.kernel_size == (2, 3)
    assert patch.proj.stride == (2, 3)


def test_patch_embedding_non_square_image_and_patch() -> None:
    patch = PatchEmbedding(
        img_size=(4, 6),
        patch_size=(2, 3),
        in_chans=1,
        embed_dim=5,
    )
    x = torch.randn(2, 1, 4, 6)
    out = patch(x)
    assert out.shape == (2, 4, 5)


def test_patch_embedding_bias_optional() -> None:
    patch = PatchEmbedding(
        img_size=2,
        patch_size=1,
        in_chans=1,
        embed_dim=3,
        bias=False,
    )
    assert patch.proj.bias is None


def test_positional_embedding_parameter_shapes() -> None:
    pos_no_cls = PositionalEmbedding2D(
        num_patches=3,
        embed_dim=2,
        include_cls=False,
    )
    pos_with_cls = PositionalEmbedding2D(
        num_patches=3,
        embed_dim=2,
        include_cls=True,
    )
    assert pos_no_cls.pos_embed.shape == (1, 3, 2)
    assert pos_with_cls.pos_embed.shape == (1, 4, 2)


def test_positional_embedding_preserves_dtype() -> None:
    pos = PositionalEmbedding2D(num_patches=2, embed_dim=3)
    x = torch.zeros(1, 2, 3, dtype=torch.float16)
    out = pos(x)
    assert out.dtype == torch.float16
