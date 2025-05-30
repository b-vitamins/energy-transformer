#!/usr/bin/env python3
"""Verify the security fix works correctly."""

from energy_transformer.spec.primitives import Dimension, Context

# This should NOT print anything or execute code
ctx = Context(dimensions={"x": 10})
evil = Dimension("evil", formula="print('HACKED!') or x")
result = evil.resolve(ctx)
print(f"Evil formula result: {result}")  # Should print None

# This should work correctly
good = Dimension("good", formula="x * 2")
result = good.resolve(ctx)
print(f"Good formula result: {result}")  # Should print 20
