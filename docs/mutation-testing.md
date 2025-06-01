# Mutation Testing Guide

## What is Mutation Testing?

Mutation testing verifies that your tests actually catch bugs by:
1. Making small changes to code (mutations)
2. Running tests to see if they fail
3. If tests pass with mutated code, they're not effective

## Running Mutation Tests with Our Test Structure

Our tests are organized into categories:
- **unit/**: Fast, isolated component tests
- **integration/**: Component interaction tests
- **functional/**: End-to-end feature tests
- **security/**: Security validation tests
- **regression/**: Bug fix verification tests

### Test changed files (recommended for PRs):
```bash
python scripts/run_mutation_tests.py
```

### Test specific files with automatic test discovery:
```bash
# Automatically finds relevant tests based on file location
python scripts/run_mutation_tests.py --files energy_transformer/layers/attention.py
```

### Test with specific test category:
```bash
# Run only unit tests
python scripts/run_mutation_tests.py --files energy_transformer/layers/attention.py --category unit

# Run only integration tests
python scripts/run_mutation_tests.py --files energy_transformer/models/base.py --category integration
```

### Test everything (very slow):
```bash
python scripts/run_mutation_tests.py --all
```

### Get improvement suggestions:
```bash
python scripts/run_mutation_tests.py --suggest
```

### View results:
```bash
# Show summary
mutmut results

# Show specific mutant
mutmut show 1

# Generate HTML report
mutmut html
open html/index.html
```

## Test Discovery Logic

The mutation testing script automatically finds relevant tests:

1. **Layer modules** (e.g., `layers/attention.py`):
   - Primary: `tests/unit/layers/test_attention.py`
   - Secondary: `tests/integration/**/test_*attention*.py`

2. **Model modules** (e.g., `models/vision/vit.py`):
   - Primary: `tests/unit/models/vision/test_vit.py`
   - Secondary: `tests/integration/test_model_building.py`

3. **Spec modules** (e.g., `spec/primitives.py`):
   - Primary: `tests/unit/spec/test_primitives.py`
   - Secondary: `tests/integration/test_spec_to_model.py`
   - Tertiary: `tests/functional/test_*primitives*.py`

4. **Security-sensitive modules**:
   - Always includes: `tests/security/`

## Common Mutation Types

1. **Boundary Mutations**: `>` → `>=`, `<` → `<=`
2. **Arithmetic Mutations**: `+` → `-`, `*` → `/`
3. **Constant Mutations**: `1.0` → `1.1`, `0` → `1`
4. **Boolean Mutations**: `and` → `or`, `True` → `False`
5. **Return Mutations**: Changing return values

## Killing Mutants by Test Category

### Unit Test Examples

```python
# Code in energy_transformer/layers/attention.py
if epoch >= max_epochs:
    break

# Unit test to kill mutant (>= → >)
# In tests/unit/layers/test_attention.py
def test_stops_at_exact_max_epochs():
    result = train(max_epochs=10)
    assert result.epochs_completed == 10  # Not 11
```

### Integration Test Examples

```python
# Code in energy_transformer/models/base.py
scale = 1.0 / math.sqrt(head_dim)

# Integration test to kill mutant (1.0 → 0.9)
# In tests/integration/test_model_building.py
def test_attention_scaling_in_full_model():
    model = build_model(head_dim=64)
    assert abs(model.attention.scale - 0.125) < 1e-6
```

### Security Test Examples

```python
# Code in energy_transformer/spec/primitives.py
if not isinstance(x, torch.Tensor):
    raise TypeError("Expected tensor")

# Security test to kill mutant (TypeError → ValueError)
# In tests/security/test_type_safety.py
def test_type_error_prevents_injection():
    with pytest.raises(TypeError, match="Expected tensor"):
        function("malicious_string")
```

## Best Practices

1. **Test Selection**: Let the script automatically select relevant tests
2. **Category Focus**: Use `--category` to focus on specific test types
3. **Incremental Testing**: Test changed files first, then expand
4. **Kill Important Mutants**: Focus on mutants in critical code paths
5. **Document Acceptable Survivors**: Some equivalent mutants are OK

## Performance Tips

1. **Use categories**: `--category unit` is faster than running all tests
2. **Test changed files**: Default behavior in CI
3. **Set timeouts**: Use `--timeout` to prevent hanging
4. **Parallelize locally**: Run multiple files in parallel terminals

## CI Integration

Mutation testing runs automatically on PRs:
- Tests only changed files
- Uses intelligent test discovery
- Posts results as PR comment
- Allows category selection via workflow dispatch

## Interpreting Results by Test Type

- **Unit test failures**: Add more isolated component tests
- **Integration test failures**: Add more interaction tests
- **Security test failures**: Add more validation tests
- **No test failures**: Module might need tests in multiple categories

## Module-Specific Guidelines

### Layers (`energy_transformer/layers/`)
- Focus on unit tests for mathematical correctness
- Add integration tests for gradient flow
- Test boundary conditions and edge cases

### Models (`energy_transformer/models/`)
- Unit tests for model construction
- Integration tests for forward passes
- Performance tests for inference speed

### Spec System (`energy_transformer/spec/`)
- Unit tests for individual specs
- Integration tests for spec composition
- Functional tests for spec-to-model conversion

### Security-Sensitive Code
- Always include security tests
- Test input validation thoroughly
- Verify error messages don't leak information
