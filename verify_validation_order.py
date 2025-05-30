#!/usr/bin/env python3
"""Verify validation order is fixed."""

from energy_transformer.spec import Context, seq
from energy_transformer.spec.library import LayerNormSpec, MLPSpec

spec = seq(
    LayerNormSpec(),
    MLPSpec(),
)

ctx = Context(dimensions={"embed_dim": 768})
issues = spec.validate(ctx)

print(f"Validation issues: {issues}")
print(f"Success: {len(issues) == 0}")
