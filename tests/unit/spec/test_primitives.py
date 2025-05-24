import math

import pytest

from energy_transformer.spec import (
    CLSTokenSpec,
    ETSpec,
    HNSpec,
    LayerNormSpec,
    MHEASpec,
    PatchEmbedSpec,
    PosEmbedSpec,
    ValidationError,
    to_pair,
    validate_positive,
    validate_probability,
)


@pytest.mark.parametrize("value", [1, 0.1, 5])
def test_validate_positive_pass(value: float | int) -> None:
    validate_positive(value, "test")


@pytest.mark.parametrize("value", [0, -1, "a"])
def test_validate_positive_fail(value: object) -> None:
    with pytest.raises(ValidationError):
        validate_positive(value, "val")


@pytest.mark.parametrize("value", [0, 0.5, 1])
def test_validate_probability_pass(value: float | int) -> None:
    validate_probability(value, "p")


@pytest.mark.parametrize("value", [-0.1, 1.1, "b"])
def test_validate_probability_fail(value: object) -> None:
    with pytest.raises(ValidationError):
        validate_probability(value, "p")


def test_to_pair_from_int() -> None:
    assert to_pair(4) == (4, 4)


def test_to_pair_from_tuple() -> None:
    assert to_pair((2, 3)) == (2, 3)


@pytest.mark.parametrize("value", [(-1, 2), (1, -2), (1, 2, 3), "x"])
def test_to_pair_invalid(value: object) -> None:
    with pytest.raises(ValidationError):
        to_pair(value)


def test_base_validate_requires_context() -> None:
    spec = LayerNormSpec()
    with pytest.raises(ValidationError):
        spec.validate()
    # Providing embedding dim should pass
    spec.validate(upstream_embedding_dim=16)


def test_layer_norm_estimate() -> None:
    spec = LayerNormSpec()
    assert spec.estimate_params(32) == 64


def test_mhea_effective_beta() -> None:
    spec = MHEASpec(num_heads=2, head_dim=16)
    expected = 1.0 / math.sqrt(16)
    assert math.isclose(spec.get_effective_beta(), expected)


def test_mhea_estimate_params_with_bias() -> None:
    spec = MHEASpec(num_heads=2, head_dim=4, bias=True)
    params = spec.estimate_params(32)
    assert params == (2 * 2 * 4 * 32) + (2 * 2 * 4)


@pytest.mark.parametrize(
    "num_heads,head_dim,dropout",
    [(0, 4, 0.0), (2, 0, 0.0), (2, 4, -0.1), (2, 4, 1.1)],
)
def test_mhea_validate_bad(num_heads: int, head_dim: int, dropout: float) -> None:
    with pytest.raises(ValidationError):
        MHEASpec(num_heads=num_heads, head_dim=head_dim, dropout=dropout)


def test_mhea_validate_total_dim() -> None:
    with pytest.raises(ValidationError):
        MHEASpec(num_heads=64, head_dim=100)  # total_dim > 4096


def test_hnspec_hidden_dim_and_estimate() -> None:
    spec = HNSpec(hidden_dim=10, bias=True)
    assert spec.get_effective_hidden_dim(32) == 10
    assert spec.estimate_params(32) == (10 * 32) + 10


def test_hnspec_multiplier_limits() -> None:
    with pytest.raises(ValidationError):
        HNSpec(multiplier=0)
    with pytest.raises(ValidationError):
        HNSpec(multiplier=9.0)


def test_et_spec_estimate_params() -> None:
    spec = ETSpec(
        steps=2,
        alpha=0.5,
        layer_norm=LayerNormSpec(),
        attention=MHEASpec(num_heads=1, head_dim=8),
        hopfield=HNSpec(multiplier=2.0),
    )
    total = spec.estimate_params(32)
    ln = 64
    attn = 2 * 1 * 8 * 32
    hn_hidden = int(32 * 2.0)
    hn = hn_hidden * 32
    assert total == ln + attn + hn


def test_cls_token_spec_behaviour() -> None:
    spec = CLSTokenSpec()
    assert spec.requires_embedding_dim()
    assert spec.adds_tokens() == 1
    assert not spec.modifies_tokens()
    assert spec.estimate_params(32) == 32


def test_patch_embed_spec_tokens_and_params() -> None:
    spec = PatchEmbedSpec(img_size=8, patch_size=4, embed_dim=16, in_chans=3, bias=False)
    assert spec.get_token_count() == 4
    assert spec.get_embedding_dim() == 16
    assert not spec.modifies_tokens()
    patch_area = 4 * 4
    assert spec.estimate_params() == 3 * patch_area * 16


def test_patch_embed_validation_divisible() -> None:
    with pytest.raises(ValidationError):
        PatchEmbedSpec(img_size=7, patch_size=4, embed_dim=8)


def test_pos_embed_spec_requirements_and_estimate() -> None:
    spec = PosEmbedSpec(include_cls=True)
    assert spec.requires_embedding_dim()
    assert spec.requires_token_count()
    assert spec.modifies_tokens()
    assert spec.estimate_params_with_context(8, 4) == 5 * 8
