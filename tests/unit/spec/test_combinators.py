import pytest

from energy_transformer.spec import (
    CLSTokenSpec,
    ETSpec,
    LayerNormSpec,
    PatchEmbedSpec,
    ValidationError,
    parallel,
    repeat,
    seq,
)


def make_patch(embed_dim: int = 64) -> PatchEmbedSpec:
    return PatchEmbedSpec(img_size=32, patch_size=16, embed_dim=embed_dim)


def test_seq_flattens_and_len() -> None:
    pe = make_patch()
    cls = CLSTokenSpec()
    ln = LayerNormSpec()
    nested = seq(cls, ln)
    model = seq(pe, nested)
    assert len(model) == 3
    assert model.parts == (pe, cls, ln)
    assert cls in model
    assert model[0] == pe
    assert isinstance(model[1:], type(model))


def test_seq_embedding_and_token_count() -> None:
    model = seq(make_patch(), CLSTokenSpec(), LayerNormSpec())
    assert model.get_embedding_dim() == 64
    # PatchEmbed produces 4 tokens for 32/16, plus 1 from CLS
    assert model.get_token_count() == 5


def test_seq_modifies_tokens() -> None:
    model = seq(make_patch(), CLSTokenSpec(), LayerNormSpec())
    assert model.modifies_tokens()


def test_seq_validate_requires_upstream() -> None:
    model = seq(CLSTokenSpec())
    with pytest.raises(ValidationError):
        model.validate()


def test_seq_validate_success() -> None:
    model = seq(make_patch(), CLSTokenSpec())
    model.validate()  # should not raise


def test_repeat_basic_and_flatten() -> None:
    ln = LayerNormSpec()
    repeated = repeat(ln, 3)
    assert len(repeated) == 3
    assert repeated.parts == (ln, ln, ln)

    block = seq(ln, CLSTokenSpec())
    repeated_block = repeat(block, 2)
    assert len(repeated_block) == 4


@pytest.mark.parametrize("times", [0, 1])
def test_repeat_zero_and_one(times: int) -> None:
    ln = LayerNormSpec()
    r = repeat(ln, times)
    assert len(r) == times


def test_repeat_invalid() -> None:
    with pytest.raises(ValueError):
        repeat(LayerNormSpec(), -1)
    with pytest.raises(TypeError):
        repeat(123, 2)


def test_parallel_concat_embedding_dim() -> None:
    p = parallel(make_patch(64), make_patch(32), join_mode="concat")
    assert p.get_embedding_dim() == 96


def test_parallel_add_dimension_mismatch_validation() -> None:
    p = parallel(make_patch(64), make_patch(32), join_mode="add")
    with pytest.raises(ValidationError):
        p.validate(upstream_embedding_dim=64)


def test_parallel_add_operator() -> None:
    b1 = parallel(make_patch(64), join_mode="concat")
    b2 = make_patch(32)
    combined = b1 + b2
    assert len(combined) == 2
    combined2 = b1 + parallel(b2, join_mode="concat")
    assert len(combined2) == 2


def test_sequential_getitem_and_slice() -> None:
    model = seq(make_patch(), CLSTokenSpec(), LayerNormSpec())
    assert model[0] is model.parts[0]
    sliced = model[1:]
    assert isinstance(sliced, type(model))
    assert sliced.parts == model.parts[1:]


def test_sequential_find_and_count_parts() -> None:
    model = seq(make_patch(), repeat(ETSpec(), 2), LayerNormSpec())
    parts = model.find_parts_by_type(ETSpec)
    assert len(parts) == 2
    assert model.count_parts_by_type(ETSpec) == 2


def test_parallel_getitem_and_slice() -> None:
    p = parallel(make_patch(), make_patch(32), join_mode="concat")
    assert p[0] is p.branches[0]
    sliced = p[1:]
    assert isinstance(sliced, type(p))
    assert sliced.branches == p.branches[1:]


def test_parallel_find_branches_by_type() -> None:
    p = parallel(
        make_patch(), LayerNormSpec(), make_patch(32), join_mode="concat"
    )
    found = p.find_branches_by_type(PatchEmbedSpec)
    assert len(found) == 2
