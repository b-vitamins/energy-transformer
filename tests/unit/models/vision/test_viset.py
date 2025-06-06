"""Unit tests for Vision Simplicial Energy Transformer (ViSET)."""

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from energy_transformer.models.vision import viset


@dataclass
class ETOutput:
    tokens: torch.Tensor
    final_energy: torch.Tensor | None
    trajectory: torch.Tensor | None


pytestmark = pytest.mark.unit


class DummyET(nn.Module):
    """Simple ET block that increments tokens and optionally returns energy."""

    def __init__(self, energy: float, trajectory: list[float]):
        super().__init__()
        self.energy = energy
        self.trajectory = trajectory

    def forward(
        self,
        x: torch.Tensor,
        track: str = "none",
        **_: object,
    ) -> torch.Tensor | ETOutput:
        x = x + 1

        return_energy = track in ("energy", "both")
        return_trajectory = track in ("trajectory", "both")

        energy = torch.tensor(self.energy) if return_energy else None
        traj = torch.tensor(self.trajectory) if return_trajectory else None

        if return_energy or return_trajectory:
            return ETOutput(tokens=x, final_energy=energy, trajectory=traj)
        return x


def _make_model(depth: int = 0) -> viset.VisionSimplicialTransformer:
    return viset.VisionSimplicialTransformer(
        img_size=4,
        patch_size=2,
        in_chans=3,
        num_classes=5,
        embed_dim=8,
        depth=depth,
        num_heads=1,
        _head_dim=4,
        hopfield_hidden_dim=8,
        et_steps=1,
        et_alpha=0.1,
        order=3,
        drop_rate=0.0,
    )


def test_process_blocks_without_energy_info() -> None:
    """Test processing blocks without energy tracking."""
    model = _make_model()
    model.et_blocks = nn.ModuleList([DummyET(0.5, [0.1]), DummyET(1.0, [0.2])])
    x = torch.zeros(1, 5, 8)
    out, info = model._process_et_blocks(x, False, {})
    assert torch.allclose(out, torch.full_like(x, 2.0))
    assert info == {}


def test_process_blocks_with_energy_info() -> None:
    """Test processing blocks with energy tracking."""
    model = _make_model()
    model.et_blocks = nn.ModuleList([DummyET(0.5, [0.1]), DummyET(1.0, [0.2])])
    x = torch.zeros(1, 5, 8)
    out, info = model._process_et_blocks(x, True, {})
    assert torch.allclose(out, torch.full_like(x, 2.0))

    assert info["block_energies"] == pytest.approx([0.5, 1.0])
    assert info["total_energy"] == pytest.approx(1.5)
    assert len(info["block_trajectories"]) == 2

    import numpy as np

    np.testing.assert_array_almost_equal(
        info["block_trajectories"][0], np.array([0.1])
    )
    np.testing.assert_array_almost_equal(
        info["block_trajectories"][1], np.array([0.2])
    )


def test_forward_raises_for_wrong_image_size() -> None:
    """Test that wrong image size raises ValueError."""
    model = _make_model()
    img = torch.zeros(1, 3, 2, 4)
    with pytest.raises(ValueError, match="Input size"):
        model(img)


def test_forward_returns_logits() -> None:
    """Test forward pass returns logits."""
    model = _make_model()
    model.et_blocks = nn.ModuleList([DummyET(0.2, [0.1])])
    img = torch.zeros(1, 3, 4, 4)
    out = model(img)
    assert out.shape == (1, 5)
    assert torch.all(out == 0)


def test_forward_features_and_energy_info() -> None:
    """Test returning features and energy info."""
    model = _make_model()
    model.et_blocks = nn.ModuleList([DummyET(0.2, [0.1])])
    img = torch.zeros(1, 3, 4, 4)
    result = model(img, return_features=True, return_energy_info=True)
    assert set(result.keys()) == {"features", "energy_info"}
    assert result["features"].shape == (1, 8)

    assert result["energy_info"]["block_energies"] == pytest.approx([0.2])


def test_factory_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test factory functions create correct configurations."""
    factories = [
        (
            viset.viset_tiny,
            {
                "embed_dim": 192,
                "depth": 12,
                "num_heads": 3,
                "_head_dim": 64,
                "hopfield_hidden_dim": 768,
                "et_steps": 4,
                "et_alpha": 0.125,
                "order": 3,
                "in_chans": 3,
            },
        ),
        (
            viset.viset_small,
            {
                "embed_dim": 384,
                "depth": 12,
                "num_heads": 6,
                "_head_dim": 64,
                "hopfield_hidden_dim": 1536,
                "et_steps": 4,
                "et_alpha": 0.125,
                "order": 3,
                "in_chans": 3,
            },
        ),
        (
            viset.viset_base,
            {
                "embed_dim": 768,
                "depth": 12,
                "num_heads": 12,
                "_head_dim": 64,
                "hopfield_hidden_dim": 3072,
                "et_steps": 4,
                "et_alpha": 0.125,
                "order": 3,
                "in_chans": 3,
            },
        ),
        (
            viset.viset_large,
            {
                "embed_dim": 1024,
                "depth": 24,
                "num_heads": 16,
                "_head_dim": 64,
                "hopfield_hidden_dim": 4096,
                "et_steps": 4,
                "et_alpha": 0.125,
                "order": 3,
                "in_chans": 3,
            },
        ),
        (
            viset.viset_tiny_cifar,
            {
                "img_size": 32,
                "patch_size": 4,
                "in_chans": 3,
                "num_classes": 100,
                "embed_dim": 192,
                "depth": 12,
                "num_heads": 3,
                "_head_dim": 64,
                "hopfield_hidden_dim": 768,
                "et_steps": 4,
                "et_alpha": 0.125,
                "order": 3,
                "drop_rate": 0.1,
            },
        ),
        (
            viset.viset_small_cifar,
            {
                "img_size": 32,
                "patch_size": 4,
                "in_chans": 3,
                "num_classes": 100,
                "embed_dim": 384,
                "depth": 12,
                "num_heads": 6,
                "_head_dim": 64,
                "hopfield_hidden_dim": 1536,
                "et_steps": 4,
                "et_alpha": 0.125,
                "order": 3,
                "drop_rate": 0.1,
            },
        ),
        (
            viset.viset_2l_cifar,
            {
                "img_size": 32,
                "patch_size": 4,
                "in_chans": 3,
                "num_classes": 100,
                "embed_dim": 192,
                "depth": 2,
                "num_heads": 8,
                "_head_dim": 64,
                "hopfield_hidden_dim": 576,
                "et_steps": 6,
                "et_alpha": 10.0,
                "order": 3,
                "drop_rate": 0.1,
            },
        ),
        (
            viset.viset_4l_cifar,
            {
                "img_size": 32,
                "patch_size": 4,
                "in_chans": 3,
                "num_classes": 100,
                "embed_dim": 192,
                "depth": 4,
                "num_heads": 8,
                "_head_dim": 64,
                "hopfield_hidden_dim": 576,
                "et_steps": 5,
                "et_alpha": 5.0,
                "order": 3,
                "drop_rate": 0.1,
            },
        ),
        (
            viset.viset_6l_cifar,
            {
                "img_size": 32,
                "patch_size": 4,
                "in_chans": 3,
                "num_classes": 100,
                "embed_dim": 192,
                "depth": 6,
                "num_heads": 8,
                "_head_dim": 64,
                "hopfield_hidden_dim": 576,
                "et_steps": 4,
                "et_alpha": 2.5,
                "order": 3,
                "drop_rate": 0.1,
            },
        ),
    ]

    def capture(**kwargs: object) -> dict[str, object]:
        return kwargs

    monkeypatch.setattr(viset, "VisionSimplicialTransformer", capture)

    for factory, expected in factories:
        cfg = factory(extra=1)
        assert cfg == {**expected, "extra": 1}


def test_viset_with_different_orders() -> None:
    """Test ViSET with different simplicial orders."""
    for order in [2, 3, 4, 5]:
        model = viset.viset_2l_cifar(num_classes=10, order=order)
        assert model.order == order
        for et_block in model.et_blocks:
            assert et_block.hopfield.order == order
        img = torch.randn(2, 3, 32, 32)
        out = model(img)
        assert out.shape == (2, 10)


def test_viset_with_softmax_hopfield() -> None:
    """Test ViSET with softmax activation in Simplicial Hopfield."""
    model = viset.viset_2l_cifar(
        num_classes=10,
        hopfield_activation="softmax",
        hopfield_beta=0.2,
    )
    for et_block in model.et_blocks:
        assert et_block.hopfield.activation == "softmax"
        assert et_block.hopfield.temperature == pytest.approx(0.2)

    img = torch.randn(2, 3, 32, 32)
    out = model(img)
    assert out.shape == (2, 10)


def test_viset_real_forward_pass() -> None:
    """Test a real forward pass with actual ET blocks."""
    model = viset.viset_2l_cifar(num_classes=10)

    img = torch.randn(4, 3, 32, 32)
    out = model(img)
    assert out.shape == (4, 10)

    features = model(img, return_features=True)
    assert features.shape == (4, 192)

    result = model(img, return_energy_info=True)
    assert "logits" in result
    assert "energy_info" in result
    assert result["logits"].shape == (4, 10)
    assert len(result["energy_info"]["block_energies"]) == 2


def test_viset_patch_embedding() -> None:
    """Test patch embedding dimensions."""
    model = viset.viset_2l_cifar(num_classes=10)
    assert model.patch_embed.num_patches == 64
    assert model.patch_embed.patch_size == (4, 4)

    img = torch.randn(2, 3, 32, 32)
    patches = model.patch_embed(img)
    assert patches.shape == (2, 64, 192)


def test_viset_cls_token_and_positional_embedding() -> None:
    """Test CLS token and positional embeddings."""
    model = viset.viset_2l_cifar(num_classes=10)
    assert model.cls_token.shape == (1, 1, 192)
    assert model.pos_embed.pos_embed.shape == (65, 192)
    assert model.pos_embed.cls_token is True


def test_viset_memory_efficiency() -> None:
    """Test that ViSET can handle reasonable batch sizes."""
    model = viset.viset_2l_cifar(num_classes=10)
    img = torch.randn(32, 3, 32, 32)
    with torch.no_grad():
        out = model(img, et_kwargs={"detach": True})
    assert out.shape == (32, 10)


def test_viset_gradient_flow() -> None:
    """Test that gradients flow through ViSET."""
    model = viset.viset_2l_cifar(num_classes=10)
    img = torch.randn(2, 3, 32, 32, requires_grad=True)
    out = model(img)
    loss = out.mean()
    loss.backward()
    assert img.grad is not None
    # Classification head should receive gradients during training
    assert model.head.fc.weight.grad is not None


def test_viset_initialization() -> None:
    """Test proper initialization of ViSET components."""
    model = viset.viset_2l_cifar(num_classes=10)
    assert model.cls_token.abs().max() < 0.1
    assert model.pos_embed.pos_embed.abs().max() < 0.1
    assert torch.allclose(
        model.head.fc.weight, torch.zeros_like(model.head.fc.weight)
    )
