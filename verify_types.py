#!/usr/bin/env python3
"""Verify type union fixes work."""

from energy_transformer.spec.library import validate_positive

# These should all work without crashing
print(f"validate_positive(5): {validate_positive(5)}")
print(f"validate_positive(5.5): {validate_positive(5.5)}")
print(f"validate_positive((2, 3)): {validate_positive((2, 3))}")
print(f"validate_positive(-1): {validate_positive(-1)}")
