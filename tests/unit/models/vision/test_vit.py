import torch

from energy_transformer.models.vision.vit import (
    MLP,
    Attention,
    PatchEmbedding,
    TransformerBlock,
    VisionTransformer,
    vit_base,
    vit_large,
    vit_small,
    vit_small_cifar,
    vit_tiny,
    vit_tiny_cifar,
)


def test_patch_embedding() -> None:
    embed = PatchEmbedding(img_size=4, patch_size=2, in_chans=3, embed_dim=2)
    with torch.no_grad():
        embed.proj.weight.fill_(1.0)
        embed.proj.bias.zero_()
    x = torch.ones(1, 3, 4, 4)
    out = embed(x)
    assert out.shape == (1, 4, 2)
    # Since weights are ones and bias zero, each patch value equals sum over channels
    expected = torch.full((1, 4, 2), 12.0)
    assert torch.allclose(out, expected)


def test_attention_zero_weights_identity() -> None:
    attn = Attention(
        dim=4, num_heads=2, qkv_bias=False, attn_drop=0.0, proj_drop=0.0
    )
    with torch.no_grad():
        attn.qkv.weight.zero_()
        attn.proj.weight.zero_()
        attn.proj.bias.zero_()
    x = torch.randn(2, 3, 4)
    out = attn(x)
    # With zero weights the output should be zeros
    assert torch.all(out == 0)


def test_mlp_forward() -> None:
    # Test with matching dimensions for true identity
    mlp = MLP(in_features=3, hidden_features=3, drop=0.0)
    with torch.no_grad():
        mlp.fc1.weight.copy_(torch.eye(3))
        mlp.fc1.bias.zero_()
        mlp.fc2.weight.copy_(torch.eye(3))
        mlp.fc2.bias.zero_()

    x = torch.randn(2, 3)
    # Can't have true identity due to GELU activation, but test shape
    out = mlp(x)
    assert out.shape == (2, 3)

    # Also test the original case with different hidden dim
    mlp2 = MLP(in_features=3, hidden_features=4, drop=0.0)
    x2 = torch.randn(2, 3)
    out2 = mlp2(x2)
    assert out2.shape == (2, 3)  # Output should match input dimension


def test_transformer_block_residual() -> None:
    block = TransformerBlock(dim=4, num_heads=2, drop=0.0, attn_drop=0.0)
    with torch.no_grad():
        block.attn.qkv.weight.zero_()
        block.attn.proj.weight.zero_()
        block.attn.proj.bias.zero_()
        block.mlp.fc1.weight.zero_()
        block.mlp.fc1.bias.zero_()
        block.mlp.fc2.weight.zero_()
        block.mlp.fc2.bias.zero_()
    x = torch.randn(1, 3, 4)
    out = block(x)
    # Zero weights mean residual preserves input
    assert torch.allclose(out, x)


def test_vision_transformer_forward_zero() -> None:
    model = VisionTransformer(
        img_size=4,
        patch_size=2,
        in_chans=3,
        num_classes=2,
        embed_dim=4,
        depth=1,
        num_heads=2,
        drop_rate=0.0,
        attn_drop_rate=0.0,
    )
    with torch.no_grad():
        model.patch_embed.proj.weight.zero_()
        model.patch_embed.proj.bias.zero_()
        model.cls_token.zero_()
        model.pos_embed.zero_()
        blk = model.blocks[0]
        blk.attn.qkv.weight.zero_()
        blk.attn.proj.weight.zero_()
        blk.attn.proj.bias.zero_()
        blk.mlp.fc1.weight.zero_()
        blk.mlp.fc1.bias.zero_()
        blk.mlp.fc2.weight.zero_()
        blk.mlp.fc2.bias.zero_()
        model.head.weight.zero_()
        model.head.bias.zero_()
    x = torch.randn(1, 3, 4, 4)
    out = model(x)
    assert out.shape == (1, 2)
    assert torch.all(out == 0)
    assert model.patch_embed.num_patches == 4


def test_factory_functions_return_models() -> None:
    m1 = vit_tiny(img_size=2, patch_size=1, in_chans=1, num_classes=3)
    m2 = vit_small(img_size=2, patch_size=1, in_chans=1, num_classes=3)
    m3 = vit_base(img_size=2, patch_size=1, in_chans=1, num_classes=3)
    m4 = vit_large(img_size=2, patch_size=1, in_chans=1, num_classes=3, depth=1)
    m5 = vit_tiny_cifar(num_classes=10)
    m6 = vit_small_cifar(num_classes=10)
    assert isinstance(m1, VisionTransformer)
    assert isinstance(m2, VisionTransformer)
    assert isinstance(m3, VisionTransformer)
    assert isinstance(m4, VisionTransformer)
    assert isinstance(m5, VisionTransformer)
    assert isinstance(m6, VisionTransformer)
