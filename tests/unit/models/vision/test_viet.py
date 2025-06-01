import pytest
import torch
from torch import nn

from energy_transformer.models.base import ETOutput
from energy_transformer.models.vision import viet


class DummyET(nn.Module):
    """Simple ET block that increments tokens and optionally returns energy."""

    def __init__(self, energy: float, trajectory: list[float]):
        super().__init__()
        self.energy = energy
        self.trajectory = trajectory

    def forward(
        self,
        x: torch.Tensor,
        return_energy: bool = False,
        return_trajectory: bool = False,
        **_: object,
    ) -> torch.Tensor | ETOutput:
        x = x + 1
        energy = torch.tensor(self.energy) if return_energy else None
        traj = torch.tensor(self.trajectory) if return_trajectory else None
        if return_energy or return_trajectory:
            return ETOutput(tokens=x, final_energy=energy, trajectory=traj)
        return x


def _make_model(depth: int = 0) -> viet.VisionEnergyTransformer:
    return viet.VisionEnergyTransformer(
        img_size=4,
        patch_size=2,
        in_chans=3,
        num_classes=5,
        embed_dim=8,
        depth=depth,
        num_heads=1,
        head_dim=4,
        hopfield_hidden_dim=8,
        et_steps=1,
        et_alpha=0.1,
        drop_rate=0.0,
    )


def test_process_blocks_without_energy_info() -> None:
    model = _make_model()
    model.et_blocks = nn.ModuleList([DummyET(0.5, [0.1]), DummyET(1.0, [0.2])])
    x = torch.zeros(1, 5, 8)
    out, info = model._process_et_blocks(x, False, {})
    assert torch.allclose(out, torch.full_like(x, 2.0))
    assert info == {}


def test_process_blocks_with_energy_info() -> None:
    model = _make_model()
    model.et_blocks = nn.ModuleList([DummyET(0.5, [0.1]), DummyET(1.0, [0.2])])
    x = torch.zeros(1, 5, 8)
    out, info = model._process_et_blocks(x, True, {})
    assert torch.allclose(out, torch.full_like(x, 2.0))

    # Use pytest.approx for floating point comparisons
    assert info["block_energies"] == pytest.approx([0.5, 1.0])
    assert info["total_energy"] == pytest.approx(1.5)
    assert len(info["block_trajectories"]) == 2

    # Compare numpy arrays properly
    import numpy as np

    np.testing.assert_array_almost_equal(
        info["block_trajectories"][0],
        np.array([0.1]),
    )
    np.testing.assert_array_almost_equal(
        info["block_trajectories"][1],
        np.array([0.2]),
    )


def test_forward_raises_for_wrong_image_size() -> None:
    model = _make_model()
    img = torch.zeros(1, 3, 2, 4)
    with pytest.raises(ValueError):
        model(img)


def test_forward_returns_logits() -> None:
    model = _make_model()
    model.et_blocks = nn.ModuleList([DummyET(0.2, [0.1])])
    img = torch.zeros(1, 3, 4, 4)
    out = model(img)
    assert out.shape == (1, 5)
    assert torch.all(out == 0)


def test_forward_features_and_energy_info() -> None:
    model = _make_model()
    model.et_blocks = nn.ModuleList([DummyET(0.2, [0.1])])
    img = torch.zeros(1, 3, 4, 4)
    result = model(img, return_features=True, return_energy_info=True)
    assert set(result.keys()) == {"features", "energy_info"}
    assert result["features"].shape == (1, 8)

    # Use pytest.approx for floating point comparison
    assert result["energy_info"]["block_energies"] == pytest.approx([0.2])


def test_factory_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    factories = [
        (
            viet.viet_tiny,
            {
                "embed_dim": 192,
                "depth": 12,
                "num_heads": 3,
                "head_dim": 64,
                "hopfield_hidden_dim": 768,
                "et_steps": 4,
                "et_alpha": 0.125,
            },
        ),
        (
            viet.viet_small,
            {
                "embed_dim": 384,
                "depth": 12,
                "num_heads": 6,
                "head_dim": 64,
                "hopfield_hidden_dim": 1536,
                "et_steps": 4,
                "et_alpha": 0.125,
            },
        ),
        (
            viet.viet_base,
            {
                "embed_dim": 768,
                "depth": 12,
                "num_heads": 12,
                "head_dim": 64,
                "hopfield_hidden_dim": 3072,
                "et_steps": 4,
                "et_alpha": 0.125,
            },
        ),
        (
            viet.viet_large,
            {
                "embed_dim": 1024,
                "depth": 24,
                "num_heads": 16,
                "head_dim": 64,
                "hopfield_hidden_dim": 4096,
                "et_steps": 4,
                "et_alpha": 0.125,
            },
        ),
        (
            viet.viet_tiny_cifar,
            {
                "img_size": 32,
                "patch_size": 4,
                "in_chans": 3,
                "num_classes": 100,
                "embed_dim": 192,
                "depth": 12,
                "num_heads": 3,
                "head_dim": 64,
                "hopfield_hidden_dim": 768,
                "et_steps": 4,
                "et_alpha": 0.125,
                "drop_rate": 0.1,
            },
        ),
        (
            viet.viet_small_cifar,
            {
                "img_size": 32,
                "patch_size": 4,
                "in_chans": 3,
                "num_classes": 100,
                "embed_dim": 384,
                "depth": 12,
                "num_heads": 6,
                "head_dim": 64,
                "hopfield_hidden_dim": 1536,
                "et_steps": 4,
                "et_alpha": 0.125,
                "drop_rate": 0.1,
            },
        ),
        (
            viet.viet_2l_cifar,
            {
                "img_size": 32,
                "patch_size": 4,
                "in_chans": 3,
                "num_classes": 100,
                "embed_dim": 192,
                "depth": 2,
                "num_heads": 8,
                "head_dim": 64,
                "hopfield_hidden_dim": 576,
                "et_steps": 6,
                "et_alpha": 10.0,
                "drop_rate": 0.1,
            },
        ),
        (
            viet.viet_4l_cifar,
            {
                "img_size": 32,
                "patch_size": 4,
                "in_chans": 3,
                "num_classes": 100,
                "embed_dim": 192,
                "depth": 4,
                "num_heads": 8,
                "head_dim": 64,
                "hopfield_hidden_dim": 576,
                "et_steps": 5,
                "et_alpha": 5.0,
                "drop_rate": 0.1,
            },
        ),
        (
            viet.viet_6l_cifar,
            {
                "img_size": 32,
                "patch_size": 4,
                "in_chans": 3,
                "num_classes": 100,
                "embed_dim": 192,
                "depth": 6,
                "num_heads": 8,
                "head_dim": 64,
                "hopfield_hidden_dim": 576,
                "et_steps": 4,
                "et_alpha": 2.5,
                "drop_rate": 0.1,
            },
        ),
    ]

    def capture(**kwargs: object) -> dict[str, object]:
        return kwargs

    monkeypatch.setattr(viet, "VisionEnergyTransformer", capture)

    for factory, expected in factories:
        cfg = factory(extra=1)
        assert cfg == {**expected, "extra": 1}
