# Testing

## Required

For **any code**: unit + integration + property tests  
For **bug fixes**: also regression test

```bash
# Development
pytest tests/unit/path/to/test_file.py -xvs

# Pre-commit
pytest tests/ -k "module_name" -x

# Full
pytest tests/
```

## Structure

```
tests/
├── conftest.py          # Shared fixtures
├── unit/                # Isolated, fast
├── integration/         # Component interaction
├── functional/          # End-to-end workflows
├── performance/         # Benchmarks
├── regression/          # Bug-specific
└── fixtures/            # Test data
```

## Naming

### Files

Source → Test mapping:
```
energy_transformer/layers/attention.py → tests/unit/layers/test_attention.py
energy_transformer/models/vision/viet.py → tests/unit/models/vision/test_viet.py
```

### Functions

```python
# Pattern: test_{function}_{scenario}_{outcome}
def test_forward_with_mask_returns_correct_shape():
def test_energy_minimization_converges():
def test_invalid_dimensions_raises_value_error():
```

### Classes

```python
class TestMultiheadEnergyAttention:  # Test{ClassName}
    def test_forward_returns_scalar_energy(self):
    def test_gradient_matches_autograd(self):
```

## Markers

```python
# Speed (auto-applied by runtime)
@pytest.mark.fast      # <100ms
@pytest.mark.slow      # >1s

# Resources
@pytest.mark.gpu       # Requires CUDA
@pytest.mark.memory    # >1GB RAM

# Priority
@pytest.mark.smoke     # Critical path
@pytest.mark.core      # Core functionality
```

## Unit

**Requirements**: Fast (<100ms), isolated, deterministic, focused

```python
# tests/unit/layers/test_attention.py
import pytest
import torch
from energy_transformer.layers.attention import MultiheadEnergyAttention

class TestMultiheadEnergyAttention:
    
    @pytest.fixture
    def attention(self):
        return MultiheadEnergyAttention(embed_dim=128, num_heads=8)
    
    @pytest.fixture
    def sample_input(self):
        return torch.randn(2, 10, 128)  # (B, N, D)
    
    def test_forward_returns_scalar_energy(self, attention, sample_input):
        energy = attention(sample_input)
        assert energy.ndim == 0
        assert energy.dtype == torch.float32
        assert torch.isfinite(energy)
    
    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 1),    # Edge case
        (4, 128),  # Typical
        (32, 512), # Large
    ])
    def test_various_sizes(self, attention, batch_size, seq_len):
        x = torch.randn(batch_size, seq_len, 128)
        energy = attention(x)
        assert torch.isfinite(energy)
    
    def test_gradient_computation(self, attention, sample_input):
        # Explicit
        grad_explicit = attention.compute_grad(sample_input)
        
        # Autograd
        sample_input.requires_grad_(True)
        energy = attention(sample_input)
        grad_auto = torch.autograd.grad(energy, sample_input)[0]
        
        torch.testing.assert_close(grad_explicit, grad_auto, rtol=1e-5, atol=1e-6)
    
    def test_invalid_input_dimension(self, attention):
        wrong_input = torch.randn(10, 128)  # Missing batch
        
        with pytest.raises(ValueError) as exc_info:
            attention(wrong_input)
            
        assert "Expected 3D input" in str(exc_info.value)
        assert "got 2D" in str(exc_info.value)
```

**DO**: fixtures, edge cases, error messages, `torch.testing.assert_close`, CPU+GPU  
**DON'T**: test PyTorch internals, use files/network, depend on order, use delays

## Integration

**Requirements**: Test component interaction, realistic configs

```python
# tests/integration/test_energy_transformer_block.py
class TestEnergyTransformerIntegration:
    
    @pytest.fixture
    def et_block(self):
        return EnergyTransformer(
            layer_norm=EnergyLayerNorm(256),
            attention=MultiheadEnergyAttention(256, num_heads=8),
            hopfield=HopfieldNetwork(256, hidden_dim=1024),
            steps=5,
            alpha=0.1
        )
    
    def test_energy_decreases(self, et_block):
        from energy_transformer.testing import assert_energy_decreases
        x = torch.randn(4, 50, 256)
        assert_energy_decreases(et_block, x, tolerance=1e-6)
    
    @pytest.mark.slow
    def test_convergence_improves_with_steps(self):
        x = torch.randn(2, 50, 256)
        energies = []
        
        for steps in [1, 5, 10, 20]:
            et = EnergyTransformer(..., steps=steps)
            # Track final energy
            energies.append(final_energy)
        
        # Monotonic improvement
        assert all(e1 >= e2 for e1, e2 in zip(energies[:-1], energies[1:]))
```

**DO**: test data flow, config compatibility, use real components  
**DON'T**: test all combinations, duplicate unit tests, mock components

## Property

**Requirements**: Mathematical invariants, behavioral properties

```python
# tests/unit/layers/test_layer_norm_properties.py
from hypothesis import given, strategies as st

class TestEnergyLayerNormProperties:
    
    @given(
        batch_size=st.integers(1, 32),
        seq_len=st.integers(1, 128),
        embed_dim=st.integers(16, 512)
    )
    def test_gradient_is_lagrangian_derivative(self, batch_size, seq_len, embed_dim):
        layer = EnergyLayerNorm(embed_dim)
        x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
        
        g = layer(x)
        L = layer.compute_energy(x)
        grad_L = torch.autograd.grad(L.sum(), x, create_graph=True)[0]
        
        torch.testing.assert_close(g, grad_L, rtol=1e-5, atol=1e-6)
```

**DO**: test invariants, use hypothesis, verify conservation laws  
**DON'T**: exact float equality, implementation-specific properties

## Regression

**Requirements**: Document bug, test exact trigger, reference issue

```python
# tests/regression/test_issue_42_attention_mask.py
"""Issue #42: Attention mask dimension bug.

Bug: (B, 1, N, N) mask shape failed with unclear error.
Fixed: commit abc123
"""

def test_attention_mask_broadcasting():
    attn = MultiheadEnergyAttention(embed_dim=128, num_heads=8)
    x = torch.randn(2, 10, 128)
    
    # Both shapes work
    mask_3d = torch.ones(2, 10, 10, dtype=torch.bool)
    mask_4d = torch.ones(2, 1, 10, 10, dtype=torch.bool)
    
    energy_3d = attn(x, attn_mask=mask_3d)
    energy_4d = attn(x, attn_mask=mask_4d)
    
    torch.testing.assert_close(energy_3d, energy_4d)
```

**DO**: reference issue, document bug, test trigger + edge cases  
**DON'T**: delete tests, over-specify implementation

## Performance

```python
# tests/performance/test_attention_speed.py
@pytest.mark.benchmark
def test_attention_throughput(benchmark):
    attn = MultiheadEnergyAttention(512, 8)
    x = torch.randn(32, 128, 512)
    
    # Warmup
    for _ in range(10):
        _ = attn(x)
        
    result = benchmark(lambda: attn(x))
    assert result.median < 0.01  # 10ms budget
```

## Fixtures

### Global (conftest.py)

```python
import pytest
import torch

@pytest.fixture(autouse=True)
def set_random_seed():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def simple_batch():
    return {
        'images': torch.randn(4, 3, 32, 32),
        'labels': torch.randint(0, 10, (4,))
    }

def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA")
    config.addinivalue_line("markers", "slow: takes >1s")
```

## Patterns

### Error Testing

```python
def test_dimension_mismatch_helpful_error():
    layer = SomeLayer(expected_dim=256)
    wrong_input = torch.randn(10, 128)
    
    with pytest.raises(ValueError) as exc_info:
        layer(wrong_input)
    
    error = str(exc_info.value)
    assert all(x in error for x in ["Expected", "256", "got 128", "Possible fix:"])
```

### Device Compatibility

```python
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.gpu)
])
def test_device_compatibility(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    model = MyModel().to(device)
    x = torch.randn(2, 10, 128).to(device)
    output = model(x)
    assert output.device.type == device
```

### Numerical Stability

```python
def test_numerical_stability():
    layer = EnergyLayer()
    
    # Extreme values
    for value in [1e-8, 1e8]:
        x = torch.full((2, 10, 128), value)
        out = layer(x)
        assert torch.isfinite(out).all()
```

## Utilities

```python
from energy_transformer.testing import (
    assert_energy_decreases,
    assert_dimension_preserved,
    create_mock_spec
)

def test_energy_optimization():
    model = EnergyTransformer(...)
    x = torch.randn(4, 100, 256)
    
    assert_energy_decreases(
        model, x, 
        steps=20,
        tolerance=1e-6,
        strict=True  # Monotonic
    )
```

## CI

```yaml
# Fast (<2min)
push:
  pytest -m "not slow and not gpu" -n auto

# Full (<10min)
pull_request:
  pytest -m "not gpu" --cov=energy_transformer

# Nightly
schedule:
  pytest -m gpu
```

## Coverage

- Minimum: 90%
- New code: 90%
- Critical paths: 100%

```bash
pytest --cov=energy_transformer --cov-report=html
```

## Debug

```bash
# Verbose
pytest test_file.py::test_function -xvs

# Debugger on failure
pytest --pdb test_file.py

# Show locals
pytest -l test_file.py

# Last failed
pytest --lf
```

## Checklist

- [ ] Unit tests for all public methods
- [ ] Integration test if interacts
- [ ] Property tests for invariants
- [ ] Regression test if bug fix
- [ ] Docstrings on test functions
- [ ] Coverage ≥90%
- [ ] No flaky tests
- [ ] Performance tests for critical paths

Good tests: fast, reliable, informative, maintainable.