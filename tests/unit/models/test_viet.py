import pytest
import torch

from energy_transformer.models.vision.viet import (
    VisionEnergyTransformer,
    viet_base_patch16_224,
    viet_large_patch16_224,
    viet_small_patch16_224,
    viet_tiny_patch16_224,
)


def _make_model() -> VisionEnergyTransformer:
    return VisionEnergyTransformer(
        img_size=8,
        patch_size=4,
        embed_dim=8,
        depth=1,
        num_classes=5,
        num_heads=2,
        head_dim=4,
        hopfield_hidden_dim=16,
        et_steps=1,
    )


def test_forward_logits_shape() -> None:
    model = _make_model()
    x = torch.randn(2, 3, 8, 8)
    logits = model(x)
    assert logits.shape == (2, 5)


def test_forward_features_and_energy_info() -> None:
    model = _make_model()
    x = torch.randn(1, 3, 8, 8)
    out = model(x, return_features=True, return_energy_info=True)
    assert set(out.keys()) == {"features", "energy_info"}
    assert out["features"].shape == (1, 8)
    info = out["energy_info"]
    assert len(info["block_energies"]) == 1
    assert len(info["block_trajectories"]) == 1
    assert info["total_energy"] == pytest.approx(sum(info["block_energies"]))


def test_forward_raises_for_wrong_size() -> None:
    model = _make_model()
    x = torch.randn(1, 3, 4, 8)
    with pytest.raises(ValueError):
        model(x)


def test_freeze_and_unfreeze_patch_embed() -> None:
    model = _make_model()
    model.freeze_patch_embed()
    assert all(not p.requires_grad for p in model.patch_embed.parameters())
    model.unfreeze_patch_embed()
    assert all(p.requires_grad for p in model.patch_embed.parameters())


def test_freeze_and_unfreeze_backbone() -> None:
    model = _make_model()
    model.freeze_backbone()
    for name, param in model.named_parameters():
        if name.startswith("head."):
            assert param.requires_grad
        else:
            assert not param.requires_grad
    model.unfreeze_backbone()
    assert all(p.requires_grad for p in model.parameters())


def test_factory_function_returns_small_model() -> None:
    model = viet_small_patch16_224()
    assert isinstance(model, VisionEnergyTransformer)
    assert model.embed_dim == 384
    assert model.img_size == 224


def test_get_attention_maps_returns_placeholder() -> None:
    model = _make_model()
    x = torch.randn(1, 3, 8, 8)
    maps = model.get_attention_maps(x)
    assert isinstance(maps, dict)
    assert "message" in maps
    assert "Energy-based" in maps["message"]


def test_forward_returns_logits_and_energy_info() -> None:
    model = _make_model()
    x = torch.randn(1, 3, 8, 8)
    out = model(x, return_energy_info=True)
    assert set(out.keys()) == {"logits", "energy_info"}
    assert out["logits"].shape == (1, 5)
    info = out["energy_info"]
    assert len(info["block_energies"]) == 1
    assert len(info["block_trajectories"]) == 1
    assert info["total_energy"] == pytest.approx(sum(info["block_energies"]))


def test_forward_only_features_returns_tensor() -> None:
    model = _make_model()
    x = torch.randn(2, 3, 8, 8)
    feats = model(x, return_features=True)
    assert isinstance(feats, torch.Tensor)
    assert feats.shape == (2, 8)


@pytest.mark.parametrize(
    "factory,embed,depth",
    [
        (viet_tiny_patch16_224, 192, 12),
        (viet_base_patch16_224, 768, 12),
        (viet_large_patch16_224, 1024, 24),
    ],
)
def test_other_factory_functions(factory, embed, depth) -> None:
    model = factory()
    assert isinstance(model, VisionEnergyTransformer)
    assert model.embed_dim == embed
    assert model.depth == depth
