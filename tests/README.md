# Energy Transformer Tests

## Running Tests

### Run all tests:
```bash
pytest
```

### Run only fast tests:
```bash
pytest -m "not slow"
```

### Run specific test markers:
```bash
pytest -m "gpu"  # GPU tests only
pytest -m "integration"  # Integration tests only
pytest -m "not slow and not gpu"  # Fast CPU tests
```

### Run with coverage report:
```bash
pytest --cov=energy_transformer --cov-report=html
open htmlcov/index.html
```

## Test Markers

- `@pytest.mark.slow`: Tests that take > 1 second
- `@pytest.mark.gpu`: Tests requiring CUDA/GPU
- `@pytest.mark.integration`: Multi-component integration tests
- `@pytest.mark.benchmark`: Performance benchmark tests

## Coverage Requirements

All code must maintain ≥ 90% test coverage. New PRs will fail if they reduce coverage below this threshold.
