# Energy Transformer Tests

## Test Organization

The test suite is organized by test type to promote clarity, maintainability, and efficient execution:

```
tests/
├── unit/           # Fast, isolated unit tests
├── integration/    # Tests verifying component interactions
├── functional/     # End-to-end functional tests
├── performance/    # Performance and benchmark tests
├── security/       # Security and safety tests
├── regression/     # Tests for specific bug fixes
└── fixtures/       # Shared test data and utilities
```

## Running Tests

### Quick Test Runs

```bash
# Run only fast unit tests (for development)
pytest -m "unit and not slow"

# Run unit tests for a specific module
pytest tests/unit/layers/

# Run with parallel execution
pytest -n auto
```

### Comprehensive Test Runs

```bash
# Run all tests
pytest

# Run all tests with coverage
pytest --cov=energy_transformer --cov-report=html

# Run specific test categories
pytest tests/integration/
pytest tests/security/
pytest tests/performance/
```

### Test Markers

Tests are marked with various attributes for selective execution:

```bash
# Run only slow tests
pytest -m slow

# Run tests that don't require GPU
pytest -m "not gpu"

# Run smoke tests for quick validation
pytest -m smoke

# Run security tests
pytest -m security

# Combine markers
pytest -m "unit and not slow and not gpu"
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Characteristics**: Fast (<100ms), no external dependencies, heavily mocked
- **Examples**: Individual layer tests, spec validation tests

### Integration Tests (`tests/integration/`)
- **Purpose**: Verify interactions between components
- **Characteristics**: Test realistic workflows, may be slower than unit tests
- **Examples**: Model building, spec-to-model conversion, graph execution

### Functional Tests (`tests/functional/`)
- **Purpose**: End-to-end testing of complete features
- **Characteristics**: Test public APIs, user-facing functionality
- **Examples**: Complete vision model workflows, import behavior

### Performance Tests (`tests/performance/`)
- **Purpose**: Benchmark and monitor performance
- **Characteristics**: Measure speed, memory usage, scalability
- **Examples**: Cache performance, model inference speed

### Security Tests (`tests/security/`)
- **Purpose**: Verify security measures and input validation
- **Characteristics**: Test against malicious inputs, code injection
- **Examples**: eval() prevention, type safety validation

### Regression Tests (`tests/regression/`)
- **Purpose**: Prevent reintroduction of fixed bugs
- **Characteristics**: Target specific issues that were previously fixed
- **Examples**: Deep recursion handling, validation order fixes

## Writing New Tests

### 1. Determine Test Category
- Is it testing a single component? → `unit/`
- Is it testing component interactions? → `integration/`
- Is it testing user-facing features? → `functional/`
- Is it measuring performance? → `performance/`
- Is it testing security? → `security/`
- Is it preventing a bug regression? → `regression/`

### 2. Follow Naming Conventions
- Test files: `test_{feature}_{aspect}.py`
- Test classes: `Test{Feature}{Aspect}`
- Test methods: `test_{behavior}_{expected_outcome}`

### 3. Use Appropriate Markers
```python
import pytest

# Mark individual tests
@pytest.mark.slow
@pytest.mark.gpu
def test_complex_model_training():
    pass

# Mark entire modules
pytestmark = [pytest.mark.unit, pytest.mark.fast]
```

### 4. Use Shared Fixtures
```python
# Use fixtures from conftest.py
def test_model_forward(simple_image_batch):
    model = create_model()
    output = model(simple_image_batch)
    assert output.shape == (4, 1000)
```

## Coverage Requirements

- Overall coverage must be ≥ 90%
- New code must have ≥ 90% coverage
- Branch coverage is required

Check coverage reports:
```bash
# Generate HTML report
pytest --cov=energy_transformer --cov-report=html
open htmlcov/index.html

# See uncovered lines in terminal
pytest --cov=energy_transformer --cov-report=term-missing
```

## CI/CD Integration

Tests run automatically on:
- Every push to master/main/develop
- Every pull request
- Different test suites run based on context

GitHub Actions workflow runs:
1. Linting and formatting checks
2. Type checking with mypy
3. Unit tests (fast)
4. Integration and functional tests
5. Coverage reporting to Codecov

## Best Practices

1. **Keep Tests Fast**: Use mocks and fixtures to avoid slow operations
2. **Test One Thing**: Each test should verify a single behavior
3. **Use Descriptive Names**: Test names should explain what they test
4. **Avoid Test Dependencies**: Tests should be independent and runnable in any order
5. **Use Fixtures**: Share common setup code through pytest fixtures
6. **Mock External Dependencies**: Don't rely on external services or files
7. **Test Edge Cases**: Include tests for error conditions and boundary values
8. **Document Complex Tests**: Add docstrings explaining non-obvious test logic

## Debugging Tests

```bash
# Run with verbose output
pytest -vv tests/unit/layers/test_attention.py

# Run with print statements visible
pytest -s tests/integration/

# Run with debugger on failure
pytest --pdb tests/regression/

# Run specific test method
pytest tests/unit/layers/test_attention.py::TestAttention::test_energy_matches_manual
```

## Adding Test Dependencies

Test dependencies go in `pyproject.toml`:
- `[tool.poetry.group.dev.dependencies]` for test frameworks
- `[tool.poetry.group.examples.dependencies]` for example-specific deps

Remember to run `poetry install --with dev` after adding dependencies.
