from dataclasses import FrozenInstanceError

import pytest
import torch

from energy_transformer.spec import (
    CLSTokenSpec,
    LayerNormSpec,
    MHEASpec,
    PatchEmbedSpec,
    PosEmbedSpec,
    ValidationError,
    parallel,
    seq,
)
from energy_transformer.spec.realise import RealisationError, realise


def test_spec_immutability():
    spec = MHEASpec(num_heads=2, head_dim=8)
    with pytest.raises(FrozenInstanceError):
        spec.num_heads = 4  # type: ignore[misc]


def test_pos_embed_validation_error():
    with pytest.raises(ValidationError):
        PosEmbedSpec(init_std=0.0)


def test_seq_dimension_propagation_and_validation():
    model = seq(
        PatchEmbedSpec(img_size=8, patch_size=4, embed_dim=16),
        CLSTokenSpec(),
        PosEmbedSpec(),
        LayerNormSpec(),
    )
    assert model.get_embedding_dim() == 16
    assert model.get_token_count() == 5
    model.validate()  # should not raise


def test_parallel_add_mode_dimension():
    p = parallel(
        PatchEmbedSpec(img_size=8, patch_size=4, embed_dim=16),
        PatchEmbedSpec(img_size=8, patch_size=4, embed_dim=16),
        join_mode="add",
    )
    assert p.get_embedding_dim() == 16


def test_parallel_add_mode_unknown_dimension():
    p = parallel(
        PatchEmbedSpec(img_size=8, patch_size=4, embed_dim=16),
        LayerNormSpec(),
        join_mode="add",
    )
    assert p.get_embedding_dim() == 16


def test_realise_simple_sequence():
    spec = seq(
        PatchEmbedSpec(img_size=4, patch_size=2, embed_dim=8, in_chans=3),
        CLSTokenSpec(),
        PosEmbedSpec(),
        LayerNormSpec(),
    )
    model = realise(spec)
    x = torch.randn(1, 3, 4, 4)
    out = model(x)
    assert out.shape == (1, 5, 8)


def test_realise_requires_context_error():
    with pytest.raises(RealisationError):
        realise(CLSTokenSpec())
