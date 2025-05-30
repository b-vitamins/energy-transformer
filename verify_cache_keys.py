#!/usr/bin/env python3
"""Verify cache key generation handles nested structures."""

from energy_transformer.spec import Context, realise
from energy_transformer.spec.library import IdentitySpec
from energy_transformer.spec.realise import _config

ctx1 = Context(dimensions={"a": 1, "b": 2}, metadata={"nested": {"x": 1, "y": 2}})
ctx2 = Context(dimensions={"b": 2, "a": 1}, metadata={"nested": {"y": 2, "x": 1}})

spec = IdentitySpec()

_config.cache.clear()

model1 = realise(spec, context=ctx1)
print(f"Cache stats after ctx1: hits={_config.cache._hit_count}, misses={_config.cache._miss_count}")
model2 = realise(spec, context=ctx2)
print(f"Cache stats after ctx2: hits={_config.cache._hit_count}, misses={_config.cache._miss_count}")

if _config.cache._hit_count > 0:
    print("Success: Cache key ignores dictionary ordering!")
else:
    print("FAIL: Cache key is sensitive to ordering")
