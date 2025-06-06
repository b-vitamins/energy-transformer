# Contributor Guide

Research project exploring energy-based alternatives to standard transformer components.

## Setup

### Recommended: Guix

`manifest.scm` provides reproducible development environment.

#### Install
- Follow: https://guix.gnu.org/manual/en/html_node/Binary-Installation.html
- Works alongside existing package managers

#### Environment

1. **Add channel** to `~/.config/guix/channels.scm`:
   ```scheme
   (cons* (channel
           (name 'myguix)
           (url "https://github.com/b-vitamins/myguix.git")
           (branch "master")
           (introduction
            (make-channel-introduction
             "85d58b09dc71e9dc9834b666b658f79d2e212d65"
             (openpgp-fingerprint
              "883B CA6B D275 A5F2 673C  C5DD 2AD3 2FC0 2A50 01F7"))))
          %default-channels)
   ```

2. **Update**:
   ```bash
   guix pull
   ```

3. **Enter**:
   ```bash
   guix shell -m manifest.scm
   ```

### Alternative: Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry install --with dev,examples
poetry shell
```

### Last Resort: pip

```bash
pip install -e .
```

## Adage

Follow PyTorch conventions. When in doubt, check PyTorch core.

### Architecture

- **Composition > Inheritance**: Build complexity from simple parts
- **Explicit > Implicit**: No hidden state or magic
- **Fail Fast**: Validate early with helpful errors
- **Zero Surprises**: If it looks like PyTorch, it acts like PyTorch

### Variables

```python
# GOOD: Concise PyTorch style
x = torch.randn(B, N, D)
q, k, v = self.qkv(x).chunk(3, dim=-1)
attn = (q @ k.transpose(-2, -1)) * self.scale

# BAD: Verbose
input_tensor = torch.randn(batch_size, sequence_length, embedding_dimension)
```

### Abbreviations

```
B = batch_size          N = num_tokens/sequence_length
D = embedding_dim       H = num_heads
L = num_layers         C = num_classes
T = time_steps         K = hidden_dim/memory_dim
```

### Comments

```python
x = self.proj(x)  # (B, N, D) -> (B, N, 3*D)
x = x.view(B, N, 3, H, D // H)  # (B, N, 3, H, D_h)
x = x.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D_h)
```

### __init__.py

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .attention import MultiheadEnergyAttention

__all__ = ["MultiheadEnergyAttention"]

def __getattr__(name):
    if name == "MultiheadEnergyAttention":
        from .attention import MultiheadEnergyAttention
        return MultiheadEnergyAttention
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

### Energy

- Energy functions return scalars
- Provide explicit gradients when possible
- Test energy decreases during optimization
- Ensure numerical stability

### Documentation

NumPy-style docstrings:
- One-line summary
- Parameters with types
- Returns with shapes
- Examples when helpful
- Paper references when relevant

Doc files use hyphens: `docs/getting-started.md`

### Misc

1. **No TODO/FIXME**: Fix now or create issue
2. **No commented code**: Git remembers
3. **No magic numbers**: Name constants
4. **No duplication**: Extract immediately
5. **No mixed concerns**: One responsibility per function

## Add

1. Study existing patterns
2. Write tests first
3. Validate inputs with helpful errors
4. Document thoroughly

## Test

See `tests/README.md`.

```bash
# Development
pytest tests/unit/your_test.py -xvs

# Pre-commit
pytest -m "not slow"

# Full
pytest tests/
```

Requirements:
- 90% coverage minimum
- Test edge cases
- Verify error messages help

## Commit

```bash
ruff check . --fix && ruff format . && mypy . && pytest tests/
```

## Pull

**Title:** `[component] Action taken`

Examples:
- `[models] Add ViSET-XL variant`
- `[tests] Improve attention coverage`

**Checklist:**
- [ ] Tests pass
- [ ] Types complete
- [ ] Docs updated
- [ ] No performance regression

## Vibe

`AGENTS.md` contains AI pair programming instructions.

1. Provide clear context
2. Verify generated code  
3. Run full test suite
4. Review for consistency

## Version

Follow [Semantic Versioning](https://semver.org/):

```markdown
## [0.2.0] - 2024-01-15
### Added
- Simplicial Hopfield networks

### Changed  
- Improved convergence criteria

### Fixed
- Memory leak in attention
```

Update `pyproject.toml` and `CHANGELOG.md` for releases.

## Resources

- [Energy Transformer](https://arxiv.org/abs/2302.07253)
- Examples: `examples/cifar/`
- Tests: `tests/`

Questions? Open an issue or start a discussion.