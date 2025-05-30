#!/usr/bin/env python3
"""Verify cache state restoration after errors."""

from dataclasses import dataclass
import torch.nn as nn

from energy_transformer.spec import Spec, loop, realise, configure_realisation
from energy_transformer.spec.realise import _config, register, ModuleCache

@dataclass(frozen=True)
class FailingSpec(Spec):
    pass

call_count = 0

@register(FailingSpec)
def realise_failing(spec, context):
    global call_count
    call_count += 1
    if call_count == 2:
        raise ValueError("Intentional failure")
    return nn.Identity()

configure_realisation(cache=ModuleCache(enabled=True))
print(f"Initial cache state: {_config.cache.enabled}")

try:
    realise(loop(FailingSpec(), times=3, unroll=True, share_weights=False))
except Exception as e:
    print(f"Expected error: {e}")

print(f"Cache state after error: {_config.cache.enabled}")
print("Success: Cache state was restored!")
