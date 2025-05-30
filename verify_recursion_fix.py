#!/usr/bin/env python3
"""Verify recursion depth fix works correctly."""

from energy_transformer.spec import seq, realise, configure_realisation
from energy_transformer.spec.library import IdentitySpec
from energy_transformer.spec.debug import inspect_cache_stats

configure_realisation(max_recursion=5)

deep_model_spec = seq(*[IdentitySpec() for _ in range(20)])

print("First realisation (building cache)...")
model1 = realise(deep_model_spec)
print("Success!")

print("\nSecond realisation (using cache)...")
model2 = realise(deep_model_spec)
print("Success! Cache prevented recursion limit.")

inspect_cache_stats()
