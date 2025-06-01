from dataclasses import dataclass

from energy_transformer.spec import Context, Sequential, Spec, param
from energy_transformer.spec.realise import ModuleCache


@dataclass(frozen=True)
class DummySpec(Spec):
    size: int = param(default=1)


def test_simplified_cache_keys():
    cache = ModuleCache()
    simple_spec = DummySpec(size=16)
    key1 = cache._make_key(simple_spec, Context())
    assert isinstance(key1, tuple)
    assert len(key1) == 2
    assert isinstance(key1[0], str)
    assert isinstance(key1[1], str)

    complex_spec = Sequential(
        [
            DummySpec(size=32),
            DummySpec(size=64),
            Sequential(
                [
                    DummySpec(size=128),
                    DummySpec(size=256),
                ]
            ),
        ]
    )
    key2 = cache._make_key(complex_spec, Context())
    assert isinstance(key2, tuple)
    assert len(key2) == 2
    assert key1 != key2
