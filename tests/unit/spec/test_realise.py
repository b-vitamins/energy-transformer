import builtins
import sys
import types
from dataclasses import FrozenInstanceError, dataclass

import pytest
import torch
import torch.nn as nn

from energy_transformer.spec import (
    CLSTokenSpec,
    ETSpec,
    HNSpec,
    LayerNormSpec,
    MHEASpec,
    PatchEmbedSpec,
    PosEmbedSpec,
    ValidationError,
    parallel,
    seq,
)
from energy_transformer.spec.primitives import Spec
from energy_transformer.spec.realise import (
    _REALISERS,
    ParallelModule,
    RealisationError,
    Realise,
    SpecInfo,
    realise,
    register_realiser,
)


class DummyHopfield(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()


@dataclass(frozen=True)
class DummySpec(Spec):
    """Simple custom spec for testing registration."""

    def _validate_parameters(self) -> None:
        pass


@register_realiser(DummySpec)
def _realise_dummy(
    spec: DummySpec, info: SpecInfo
) -> nn.Module:  # pragma: no cover - executed via decorator
    return nn.Identity()


def test_realisation_error_str() -> None:
    err = RealisationError("m", spec_type="S", suggestion="hint")
    assert "S: m" in str(err)
    assert "Suggestion: hint" in str(err)


def test_specinfo_update_copy_and_repr() -> None:
    info = SpecInfo()
    spec = PatchEmbedSpec(img_size=2, patch_size=1, embed_dim=3, in_chans=1)
    info.update_from_spec(spec)
    assert info.embedding_dim == 3
    assert info.token_count == 4
    copy = info.copy()
    copy.embedding_dim = 6
    assert info.embedding_dim == 3
    assert str(info) == "SpecInfo(embed_dim=3, token_count=4)"
    assert "SpecInfo(" in repr(info)


def test_specinfo_validate() -> None:
    info = SpecInfo(embedding_dim=3)
    info.validate_spec(LayerNormSpec())  # should not raise
    with pytest.raises(RealisationError):
        SpecInfo().validate_spec(LayerNormSpec())


def test_register_and_realise_custom() -> None:
    spec = DummySpec()
    module = realise(spec)
    assert isinstance(module, nn.Identity)
    _REALISERS.pop(DummySpec, None)


def test_realise_various_paths() -> None:
    assert Realise(None) is None
    mod = nn.Linear(1, 1)
    assert realise(mod) is mod
    with pytest.raises(TypeError):
        realise(123)


def test_realise_sequential_error_propagation() -> None:
    with pytest.raises(RealisationError):
        realise(seq(CLSTokenSpec()))


def test_parallel_module_combine_modes() -> None:
    pm = ParallelModule([nn.Identity(), nn.Identity()], join_mode="concat")
    x = torch.randn(1, 2)
    assert pm(x).shape[-1] == 4
    pm = ParallelModule([nn.Identity(), nn.Identity()], join_mode="add")
    assert torch.allclose(pm(x), x * 2)
    pm = ParallelModule([nn.Identity(), nn.Identity()], join_mode="multiply")
    assert torch.allclose(pm(x), x * x)
    pm = ParallelModule([nn.Identity()], join_mode="bad")
    with pytest.raises(ValueError):
        pm(x)


def _patch_import(monkeypatch, target: str):
    original = builtins.__import__

    def fake(name, globals=None, locals=None, fromlist=(), level=0):
        if name == target:
            raise ImportError("boom")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake)


def test_layer_norm_fallback(monkeypatch) -> None:
    info = SpecInfo(embedding_dim=3)
    _patch_import(monkeypatch, "energy_transformer.layers.layer_norm")
    mod = realise(LayerNormSpec(), info)
    assert isinstance(mod, nn.LayerNorm)


def test_mhea_paths(monkeypatch) -> None:
    info = SpecInfo(embedding_dim=4)
    mod = realise(MHEASpec(num_heads=1, head_dim=2), info)
    assert mod.__class__.__name__ == "MultiHeadEnergyAttention"
    with pytest.raises(RealisationError):
        realise(MHEASpec(), SpecInfo())
    _patch_import(monkeypatch, "energy_transformer.layers")
    with pytest.raises(RealisationError):
        realise(MHEASpec(), info)


def test_hn_paths(monkeypatch) -> None:
    info = SpecInfo(embedding_dim=2)
    mod = realise(HNSpec(hidden_dim=2), info)
    assert mod.__class__.__name__ == "HopfieldNetwork"
    with pytest.raises(RealisationError):
        realise(HNSpec(), SpecInfo())
    _patch_import(monkeypatch, "energy_transformer.layers.hopfield")
    with pytest.raises(RealisationError):
        realise(HNSpec(), info)

    # Create a mock module where HopfieldNetwork raises KeyError
    mock_module = types.ModuleType("mock_hopfield")
    mock_module.HopfieldNetwork = type(
        "HopfieldNetwork",
        (),
        {
            "__init__": lambda *args, **kwargs: (_ for _ in ()).throw(
                KeyError("bad")
            )
        },
    )
    monkeypatch.setitem(
        sys.modules, "energy_transformer.layers.hopfield", mock_module
    )
    with pytest.raises(RealisationError):
        realise(HNSpec(), info)


def test_et_paths(monkeypatch) -> None:
    info = SpecInfo(embedding_dim=2)
    mod = realise(ETSpec(steps=1), info)
    assert mod.__class__.__name__ == "EnergyTransformer"
    with pytest.raises(RealisationError):
        realise(ETSpec(), SpecInfo())
    _patch_import(monkeypatch, "energy_transformer.models.base")
    with pytest.raises(RealisationError):
        realise(ETSpec(), info)

    def none_realiser(spec, info):
        return None

    old = _REALISERS[LayerNormSpec]
    _REALISERS[LayerNormSpec] = none_realiser
    with pytest.raises(RealisationError):
        realise(ETSpec(steps=1), info)
    _REALISERS[LayerNormSpec] = old


def test_cls_token_paths(monkeypatch) -> None:
    info = SpecInfo(embedding_dim=3)
    mod = realise(CLSTokenSpec(), info)
    assert mod.__class__.__name__ == "CLSToken"
    with pytest.raises(RealisationError):
        realise(CLSTokenSpec(), SpecInfo())
    _patch_import(monkeypatch, "energy_transformer.layers.tokens")
    with pytest.raises(RealisationError):
        realise(CLSTokenSpec(), info)


def test_patch_embed_paths(monkeypatch) -> None:
    mod = realise(
        PatchEmbedSpec(img_size=2, patch_size=1, embed_dim=2, in_chans=1)
    )
    assert mod.__class__.__name__ == "PatchEmbedding"
    _patch_import(monkeypatch, "energy_transformer.layers.embeddings")
    with pytest.raises(RealisationError):
        realise(
            PatchEmbedSpec(img_size=2, patch_size=1, embed_dim=2, in_chans=1)
        )


def test_pos_embed_paths(monkeypatch) -> None:
    info = SpecInfo(embedding_dim=2, token_count=3)
    mod = realise(PosEmbedSpec(), info)
    assert mod.__class__.__name__ == "PositionalEmbedding2D"
    with pytest.raises(RealisationError):
        realise(PosEmbedSpec(), SpecInfo(embedding_dim=2))
    with pytest.raises(RealisationError):
        realise(PosEmbedSpec(), SpecInfo(token_count=2))
    _patch_import(monkeypatch, "energy_transformer.layers.embeddings")
    with pytest.raises(RealisationError):
        realise(PosEmbedSpec(), info)


# Tests from the second file that are unique or provide additional coverage


def test_spec_immutability() -> None:
    """Test that specs are immutable (frozen dataclasses)."""
    spec = MHEASpec(num_heads=2, head_dim=8)
    with pytest.raises(FrozenInstanceError):
        spec.num_heads = 4  # type: ignore[misc]


def test_pos_embed_validation_error() -> None:
    """Test PosEmbedSpec validation."""
    with pytest.raises(ValidationError):
        PosEmbedSpec(init_std=0.0)


def test_seq_dimension_propagation_and_validation() -> None:
    """Test dimension propagation through sequential spec."""
    model = seq(
        PatchEmbedSpec(img_size=8, patch_size=4, embed_dim=16),
        CLSTokenSpec(),
        PosEmbedSpec(),
        LayerNormSpec(),
    )
    assert model.get_embedding_dim() == 16
    assert model.get_token_count() == 5  # 4 patches + 1 CLS token
    model.validate()  # should not raise


def test_parallel_add_mode_dimension() -> None:
    """Test parallel spec with add mode dimension handling."""
    p = parallel(
        PatchEmbedSpec(img_size=8, patch_size=4, embed_dim=16),
        PatchEmbedSpec(img_size=8, patch_size=4, embed_dim=16),
        join_mode="add",
    )
    assert p.get_embedding_dim() == 16


def test_parallel_add_mode_unknown_dimension() -> None:
    """Test parallel spec with mixed known/unknown dimensions."""
    p = parallel(
        PatchEmbedSpec(img_size=8, patch_size=4, embed_dim=16),
        LayerNormSpec(),  # LayerNorm doesn't define embedding_dim
        join_mode="add",
    )
    assert p.get_embedding_dim() == 16


def test_realise_simple_sequence() -> None:
    """Test realising a simple sequential model."""
    spec = seq(
        PatchEmbedSpec(img_size=4, patch_size=2, embed_dim=8, in_chans=3),
        CLSTokenSpec(),
        PosEmbedSpec(),
        LayerNormSpec(),
    )
    model = realise(spec)
    x = torch.randn(1, 3, 4, 4)
    out = model(x)
    assert out.shape == (1, 5, 8)  # 4 patches + 1 CLS, embedding dim 8


def test_realise_requires_context_error() -> None:
    """Test that realising CLSTokenSpec without context fails."""
    with pytest.raises(RealisationError):
        realise(CLSTokenSpec())


def test_realise_parallel_no_branches() -> None:
    """Test parallel spec with no valid branches."""

    # Create a spec that passes validation but returns None from realiser
    @dataclass(frozen=True)
    class NoneSpec(Spec):
        def _validate_parameters(self) -> None:
            pass

    # Register a realiser that returns None
    @register_realiser(NoneSpec)
    def _realise_none(spec: NoneSpec, info: SpecInfo) -> nn.Module | None:
        return None

    try:
        # Create parallel spec where all branches return None
        p = parallel(NoneSpec(), NoneSpec())
        with pytest.raises(RealisationError) as exc_info:
            realise(p)
        assert "No valid branches to combine" in str(exc_info.value)
    finally:
        # Clean up
        _REALISERS.pop(NoneSpec, None)


def test_realise_updates_context() -> None:
    """Test that realise properly updates context after each spec."""
    spec = seq(
        PatchEmbedSpec(img_size=4, patch_size=2, embed_dim=16, in_chans=3),
        CLSTokenSpec(),  # adds 1 token
    )
    info = SpecInfo()
    realise(spec, info)
    # Context should be updated with final dimensions
    assert info.embedding_dim == 16
    assert info.token_count == 5  # 4 patches + 1 CLS
