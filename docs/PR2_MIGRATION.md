# PR2 Migration Guide

## Breaking Changes

### 1. GraphModule Forward Signature
Graph execution now respects graph topology. Nodes receive data from their
predecessors instead of always using the original input.

**Before (BROKEN):**
```python
graph_module(x)  # every node saw x
```

**After (FIXED):**
```python
graph_module(x)  # data flows along edges correctly
```

### 2. Validation Order
Parent specs apply context updates before validating children. Specs that
previously failed validation may now pass.

### 3. Cycle Detection
Graphs with cycles fail immediately during construction.
```python
Graph(nodes=nodes, edges=cyclic_edges)  # raises ValidationError
```

## New Features

1. **Edge Transformations** – Graph edges can now apply transformations such as
   `relu`, `normalize`, or slicing syntax.
2. **Improved Error Messages** – Validation errors include hierarchy paths and
   cycle detection reports the offending path.
3. **Parallel Merge Validation** – Additional checks ensure merges are
   dimensionally compatible and weights are correct.

## Testing Your Code

Run the new test suite:

```bash
pytest tests/test_graph_and_validation.py -v
```

Verify your existing models still function as expected, especially those using
graph specifications or nested specs.

