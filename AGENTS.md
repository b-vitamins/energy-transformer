# Contributor Guide

## Project Overview
Energy Transformer implements energy-based transformers with associative memory. The codebase uses PyTorch and follows a declarative specification system for model construction.

## Quick Setup
```bash
# Install with Poetry
poetry install

# Install with examples
poetry install --with examples

# Activate virtual environment
poetry shell
```

## Development Workflow

### Pre-commit Checks
Always run these before committing:
```bash
# Auto-fix code style issues
ruff check . --fix
ruff format .

# Type checking
mypy .

# Run tests (adjust path as needed)
pytest tests/

# If all pass, commit
git add -A && git commit -m "..."
```

## Coding Conventions

### PyTorch First
Follow PyTorch's conventions for everything - API design, naming, code organization. When in doubt, check how PyTorch core does it.

## Coding Conventions

### PyTorch First
Follow PyTorch's conventions for everything - API design, naming, code organization. When in doubt, check how PyTorch core does it.

### Modern Python (3.11+)

Use modern Python features for cleaner, more maintainable code:

```python
# GOOD: Type parameters (3.12+)
from typing import Generic, TypeVar

T = TypeVar('T', bound=Tensor)

class EnergyModule(nn.Module, Generic[T]):
    """Base class for energy-based modules."""
    
    def forward(self, x: T) -> T:
        return self.minimize_energy(x)
    
    def minimize_energy(self, x: T) -> T:
        """Minimize energy via gradient descent."""
        # Implementation preserves type
        return x

# GOOD: Self type (3.11+)
from typing import Self

class ModelBuilder:
    def with_layers(self, n: int) -> Self:
        self.n_layers = n
        return self
    
    def with_heads(self, h: int) -> Self:
        self.n_heads = h
        return self

# GOOD: TypedDict for configs
from typing import TypedDict, NotRequired

class AttentionConfig(TypedDict):
    num_heads: int
    head_dim: int
    dropout: NotRequired[float]  # Optional in TypedDict
    
def build_attention(config: AttentionConfig) -> nn.Module:
    return MultiheadAttention(
        embed_dim=config['num_heads'] * config['head_dim'],
        num_heads=config['num_heads'],
        dropout=config.get('dropout', 0.0)
    )

# GOOD: Pattern matching
match spec:
    case ETBlockSpec(attention=attn, hopfield=hop):
        return EnergyTransformer(attn, hop)
    case MHEASpec(num_heads=n, head_dim=d):
        return MultiheadEnergyAttention(n * d, n)
    case _:
        raise ValueError(f"Unknown spec: {spec}")

# BAD: Old-style type checking
if isinstance(spec, ETBlockSpec):
    return EnergyTransformer(spec.attention, spec.hopfield)
elif isinstance(spec, MHEASpec):
    return MultiheadEnergyAttention(spec.num_heads * spec.head_dim, spec.num_heads)
else:
    raise ValueError(f"Unknown spec: {spec}")
```

### Protocols over ABCs

```python
# GOOD: Protocol for type checking
from typing import Protocol

class EnergyFunction(Protocol):
    def compute_energy(self, x: Tensor) -> Tensor: ...
    def compute_grad(self, x: Tensor) -> Tensor: ...

# BAD: Abstract base class
from abc import ABC, abstractmethod

class EnergyFunction(ABC):
    @abstractmethod
    def compute_energy(self, x: Tensor) -> Tensor:
        pass
```

### Properties and Descriptors

```python
# GOOD: Properties for computed attributes
class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.w_q = nn.Linear(d_model, d_k)
        
    @property
    def scale(self) -> float:
        """Scaling factor for dot product attention."""
        return self.d_k ** -0.5

# BAD: Method for simple computed value
def get_scale(self) -> float:
    return self.d_k ** -0.5
```

### Magic Methods

```python
# GOOD: Rich comparison and representation
class EnergyState:
    def __init__(self, energy: float, iteration: int):
        self.energy = energy
        self.iteration = iteration
    
    def __repr__(self) -> str:
        return f"EnergyState(energy={self.energy:.6f}, iteration={self.iteration})"
    
    def __lt__(self, other: EnergyState) -> bool:
        return self.energy < other.energy

# BAD: No useful representation
class EnergyState:
    def __init__(self, energy: float, iteration: int):
        self.energy = energy
        self.iteration = iteration
```

### Variable Naming
```python
# GOOD: Concise, PyTorch-style
x = torch.randn(B, N, D)  # input tensor
q, k, v = self.qkv(x).chunk(3, dim=-1)  # query, key, value
attn = (q @ k.transpose(-2, -1)) * self.scale

# BAD: Verbose, non-standard
input_tensor = torch.randn(batch_size, sequence_length, embedding_dimension)
```

### Shape Comments
Only for tensor operations - concise and consistent:
```python
x = self.proj(x)  # (B, N, D) -> (B, N, 3*D)
x = x.view(B, N, 3, H, D // H)  # (B, N, 3, H, D_h)
x = x.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D_h)
```

### Minimal Comments
```python
# GOOD: Code explains itself
def forward(self, x: Tensor) -> Tensor:
    B, N, D = x.shape
    
    # Single-line shape tracking only
    h = self.norm(x)  # (B, N, D)
    h = self.fc1(h)  # (B, N, 4*D)
    h = self.act(h)
    return self.fc2(h)  # (B, N, D)

# BAD: Over-commented
def forward(self, x: Tensor) -> Tensor:
    # Get the batch size, sequence length, and dimension
    B, N, D = x.shape
    
    # First we normalize the input
    h = self.norm(x)  # Apply layer normalization
```

### Intent as Code
```python
# GOOD: Intent is clear from structure
class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5
```

### Context Managers

```python
# GOOD: Context manager for state
from contextlib import contextmanager

@contextmanager
def energy_tracking(model: EnergyTransformer):
    """Track energy during forward pass."""
    tracker = EnergyTracker()
    handle = model.register_step_hook(lambda m, info: tracker.update(info))
    try:
        yield tracker
    finally:
        handle.remove()

# Usage
with energy_tracking(model) as tracker:
    output = model(x)
    print(f"Final energy: {tracker.final_energy}")
```

### Dataclasses and Slots

```python
# GOOD: Dataclass with slots for efficiency
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class AttentionConfig:
    num_heads: int
    head_dim: int
    dropout: float = 0.0
    
    @property
    def embed_dim(self) -> int:
        return self.num_heads * self.head_dim

# BAD: Regular class with boilerplate
class AttentionConfig:
    def __init__(self, num_heads: int, head_dim: int, dropout: float = 0.0):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
```

### Technical Debt Prevention

1. **No TODO/FIXME in main**: Fix it now or create an issue
2. **No commented code**: Use git history instead
3. **No magic numbers**: Use named constants
4. **No duplicate code**: Extract common patterns
5. **No mixed concerns**: Single responsibility per function/class

### Common Patterns

```python
# Parameter initialization (PyTorch style)
def reset_parameters(self) -> None:
    nn.init.xavier_uniform_(self.weight)
    if self.bias is not None:
        nn.init.zeros_(self.bias)

# Factory functions for model variants
def vit_base(**kwargs) -> VisionTransformer:
    return VisionTransformer(
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs
    )
```

### Common Patterns

```python
# Parameter initialization (PyTorch style)
def reset_parameters(self) -> None:
    nn.init.xavier_uniform_(self.weight)
    if self.bias is not None:
        nn.init.zeros_(self.bias)

# Factory functions for model variants
def vit_base(**kwargs) -> VisionTransformer:
    return VisionTransformer(
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs
    )

# GOOD: Clean class structure with properties
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = MultiheadAttention(d_model, n_heads)
        self.mlp = MLP(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# BAD: Mixing concerns, unclear structure
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Everything in one place
        self.config = config
        self.layers = nn.ModuleList()
        # Unclear what this does
        self._setup_layers()
```

## Documentation

### Docstring Standards

Follow NumPy style for consistency with PyTorch and scientific Python:

```python
def forward(
    self,
    x: Tensor,
    mask: Tensor | None = None,
    return_attention: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Apply multi-head attention mechanism.
    
    Computes scaled dot-product attention with optional masking.
    When return_attention is True, also returns attention weights
    for visualization.
    
    Parameters
    ----------
    x : Tensor
        Input tensor of shape (B, N, D) where B is batch size,
        N is sequence length, and D is embedding dimension.
    mask : Tensor | None, optional
        Attention mask of shape (B, N, N) or (B, 1, N, N).
        Values should be True for positions to mask.
    return_attention : bool, default=False
        Whether to return attention weights along with output.
    
    Returns
    -------
    Tensor | tuple[Tensor, Tensor]
        If return_attention is False:
            Output tensor of shape (B, N, D).
        If return_attention is True:
            Tuple of (output, attention_weights) where attention_weights
            has shape (B, H, N, N) with H being number of heads.
    
    Examples
    --------
    >>> attn = MultiheadAttention(embed_dim=512, num_heads=8)
    >>> x = torch.randn(32, 100, 512)
    >>> output = attn(x)
    >>> output.shape
    torch.Size([32, 100, 512])
    
    Notes
    -----
    The attention computation follows Vaswani et al. (2017) with
    the modification that layer normalization is applied before
    attention (pre-norm) rather than after.
    """
```

### Module-Level Documentation

```python
"""Multi-head energy attention mechanism.

This module implements an energy-based variant of multi-head attention
where the attention computation is framed as energy minimization over
the Stiefel manifold.

The key differences from standard attention:
- Attention weights are computed via gradient descent on an energy landscape
- Orthogonality constraints ensure diverse attention heads
- Explicit temperature control for sharpness of attention

Classes
-------
MultiheadEnergyAttention
    Main attention module with energy-based computation.
EnergyAttentionHead
    Single attention head with Stiefel manifold constraints.

Functions
---------
compute_attention_energy
    Compute the energy function for attention weights.

Examples
--------
Basic usage with default settings:

>>> attn = MultiheadEnergyAttention(embed_dim=512, num_heads=8)
>>> x = torch.randn(32, 100, 512)
>>> output = attn(x, et_kwargs={"steps": 20})

References
----------
.. [1] Hoover et al., "Energy Transformer", 2023.
   https://arxiv.org/abs/2302.07253
"""
```

### Building Documentation

Add to module docstrings for automated API docs:

```python
# In pyproject.toml or setup.cfg, configure Sphinx
# [tool.sphinx]
# source-suffix = '.rst'
# master-doc = 'index'
# autodoc-member-order = 'bysource'

# In docs/conf.py
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # For NumPy-style docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',   # For LaTeX math
    'sphinx_autodoc_typehints',
]

# Napoleon settings for NumPy style
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
```


### Documentation Checklist

For every public function/class:
- [ ] One-line summary (what it does)
- [ ] Detailed description (how/why it works)
- [ ] Parameters section with types and descriptions
- [ ] Returns section with types and shapes
- [ ] Examples section with doctests
- [ ] Notes section for implementation details
- [ ] References section for papers/resources

### LaTeX Math in Docstrings

```python
def compute_attention_scores(q: Tensor, k: Tensor) -> Tensor:
    r"""Compute scaled dot-product attention scores.
    
    The attention scores are computed as:
    
    .. math::
        \text{scores}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}
    
    where :math:`d_k` is the key dimension.
    """
```

### Core Philosophy
**No Surprises**: Code should do what it looks like it does. If it looks like PyTorch, it should behave like PyTorch.

### Factory Functions
```python
# GOOD: PyTorch-style factories with consistent naming
def vit_tiny(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """Vision Transformer Tiny (Vit-Ti/16)"""
    model = VisionTransformer(
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        model.load_state_dict(load_checkpoint('vit_tiny'))
    return model

def vit_small(**kwargs) -> VisionTransformer:
    """Vision Transformer Small (Vit-S/16)"""
    return VisionTransformer(embed_dim=384, depth=12, num_heads=6, **kwargs)

def vit_base(**kwargs) -> VisionTransformer:
    """Vision Transformer Base (Vit-B/16)"""
    return VisionTransformer(embed_dim=768, depth=12, num_heads=12, **kwargs)

# GOOD: Configuration with defaults
def create_model(name: str = 'vit_base', **kwargs) -> nn.Module:
    """Create model by name with optional config overrides."""
    match name:
        case 'vit_tiny':
            return vit_tiny(**kwargs)
        case 'vit_small':
            return vit_small(**kwargs)
        case 'vit_base':
            return vit_base(**kwargs)
        case _:
            raise ValueError(f"Unknown model: {name}")

# BAD: Inconsistent naming and parameters
def get_tiny_vit(img_size, num_classes):  # Missing kwargs
    return VisionTransformer(192, 12, 3, img_size, num_classes)

def build_ViT_small(**config):  # Inconsistent casing
    return VisionTransformer(**config)

def VIT_BASE(**params):  # All caps is wrong
    return VisionTransformer(embed_dim=768, **params)
```

### Device & Dtype Handling
```python
# GOOD: Explicit and safe device/dtype handling
def forward(self, x: Tensor) -> Tensor:
    weight = self.weight.to(device=x.device, dtype=x.dtype)
    return F.linear(x, weight)

# GOOD: Preserve input properties
class AdaptiveModule(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: Tensor) -> Tensor:
        # Ensure scale matches input
        scale = self.scale.to(device=x.device, dtype=x.dtype)
        return x * scale

# GOOD: Handle mixed precision
def attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    # Use same dtype for computation
    dtype = q.dtype
    # Upcast to float32 for stability if needed
    if dtype in (torch.float16, torch.bfloat16):
        q, k, v = q.float(), k.float(), v.float()
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores.to(dtype)
    else:
        scores = torch.matmul(q, k.transpose(-2, -1))
    return scores

# BAD: Assumes device/dtype
def forward(self, x: Tensor) -> Tensor:
    return F.linear(x, self.weight)  # Fails if x and weight on different devices

# BAD: Forces specific dtype
def forward(self, x: Tensor) -> Tensor:
    x = x.float()  # Don't force dtype unless necessary
    return self.layer(x)
```

### Performance Patterns
```python
# GOOD: PyTorch-efficient operations
x = x.view(B, N, H, -1).transpose(1, 2)  # (B, H, N, D_h)
scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, N, N)

# GOOD: Reuse allocations when possible
class EfficientAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.register_buffer('_buffer', torch.empty(0))
    
    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        # Reuse buffer if possible
        if self._buffer.shape != (B, N, N):
            self._buffer = torch.empty(B, N, N, device=x.device, dtype=x.dtype)
        torch.matmul(x, x.transpose(-2, -1), out=self._buffer)
        return self._buffer

# BAD: Unnecessary copies and reshapes
x = x.reshape(B, N, H, -1)  # Creates copy if not contiguous
x = x.permute(0, 2, 1, 3)   # Another potential copy
scores = q @ k.permute(0, 1, 3, 2)  # Permute is often unnecessary

# BAD: Not considering memory layout
for i in range(N):
    for j in range(N):
        scores[i, j] = (q[i] * k[j]).sum()  # Extremely slow
```

### Error Handling
```python
# GOOD: PyTorch-style validation with helpful context
if x.dim() != 3:
    raise ValueError(
        f"Expected 3D input (batch, seq, features), got {x.dim()}D. "
        f"Shape: {x.shape}. Did you forget to add batch dimension?"
    )

if x.size(-1) != self.d_model:
    raise ValueError(
        f"Expected features of size {self.d_model}, got {x.size(-1)}. "
        f"Possible fix: Add embedding layer before this module."
    )

# GOOD: Input validation with recovery hints
def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
    if lengths is not None and lengths.dim() != 1:
        raise ValueError(
            f"lengths must be 1D tensor of batch size, got shape {lengths.shape}. "
            f"Example: lengths = torch.tensor([10, 8, 12]) for batch of 3."
        )

# BAD: Generic errors without context
if len(x.shape) != 3:
    raise Exception("Wrong shape")

# BAD: No helpful information
if x.size(-1) != self.d_model:
    raise RuntimeError("Size mismatch")
```

### Module Organization
```python
# GOOD: Clean import order (PyTorch style)
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, TypeVar, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import MultiheadAttention
from .embeddings import PatchEmbed, PositionalEncoding
from .utils import to_2tuple

# GOOD: Conditional imports for optional dependencies
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# BAD: Mixed import styles and poor organization
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import math
from .layers import *  # Never use star imports
from typing import Any
import numpy as np
from .utils import to_2tuple
from collections.abc import Sequence

# BAD: Importing implementation details
from .attention import _scaled_dot_product  # Don't import private functions
from .layers.base import BaseLayer  # Too specific
```

### Common Abbreviations
Stick to PyTorch's standard abbreviations:
```
B = batch_size          N = num_tokens/sequence_length
D = embedding_dim       H = num_heads
L = num_layers         C = num_classes
T = time_steps         K = hidden_dim/memory_dim
```

### Working Principles

1. **Energy Functions**: Always return scalar values
2. **Gradient Safety**: Avoid autograd when explicit gradients are available
3. **Dimension Tracking**: Use the Context system for dimension propagation
4. **Lazy Loading**: Maintain fast imports - heavy dependencies should be lazy

## Testing

See `tests/README.md` for comprehensive testing guidelines, organization, and conventions.

### Quick Test Commands
```bash
# Run fast unit tests during development
pytest -m "unit and not slow" -n auto

# Run all tests before committing
pytest tests/

# Run with coverage
pytest --cov=energy_transformer --cov-report=term-missing

# Run specific test file
pytest tests/unit/layers/test_attention.py -v
```

### Key Principles
- **90% coverage minimum** for new code
- **Test categories**: unit, integration, functional, performance, security, regression
- **Use markers**: `@pytest.mark.unit`, `@pytest.mark.slow`, `@pytest.mark.gpu`
- **Test energy properties**: Use `assert_energy_decreases` from `energy_transformer.testing`

## PR Guidelines

### Title Format
`[component] Action taken`

Examples:
- `[models] Add new vision transformer variant`
- `[spec] Fix validation bug in Sequential`
- `[tests] Improve coverage for attention layers`

### PR Checklist
- [ ] Pre-commit checks pass
- [ ] Tests added/updated
- [ ] Breaking changes documented
- [ ] Performance impact considered

## Project Structure

### Key Directories
- `energy_transformer/` - Main package code
- `tests/` - Test suite (see tests/README.md)
- `examples/` - Example scripts and experiments
- `docs/` - Project documentation
- `.github/workflows/` - CI configuration

### Key Documentation
- `tests/README.md` - Testing practices and organization
- `docs/` - Autogenerated API docs

## Quality Checks

### Code Quality
```bash
# Format and lint
ruff check . --fix
ruff format .

# Type checking
mypy .

# Import sorting
ruff check . --select I --fix
```

### Performance Profiling
```python
# GOOD: Profile critical paths
import cProfile
import pstats

with cProfile.Profile() as pr:
    model(x)
    
stats = pstats.Stats(pr)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

## Debugging Tips

1. **Energy Issues**: Use the observer hooks to track energy trajectories
2. **Dimension Errors**: Check Context propagation through the spec system
3. **Import Problems**: Verify lazy loading in `__init__.py` files
4. **Memory Issues**: Consider gradient checkpointing for deep models

## Quick Exploration

### Try the Examples
```bash
# Quick CIFAR-100 test
cd examples/cifar
python quick.py --model viet --epochs 5

```

### Basic Usage
```python
from energy_transformer.models.vision import viet_base

model = viet_base(img_size=64, patch_size=16, num_classes=10)
images = torch.randn(2, 3, 64, 64)
logits = model(images, et_kwargs={"detach": False})
```

## Architecture Decisions

- **Declarative > Imperative**: Prefer specs over direct construction
- **Composition > Inheritance**: Use combinators for complex models
- **Explicit > Implicit**: Make dimension requirements clear
- **Validation > Assumption**: Check inputs at boundaries
- **Composition > Configuration**: Build complex behavior from simple modules (PyTorch way)

### Zero Tolerance for Code Smell
- **DRY violations**: Extract shared logic immediately
- **Long methods**: Break down into focused functions
- **Deep nesting**: Flatten with early returns
- **Unclear ownership**: Every line should have obvious responsibility
- **Inconsistent patterns**: Align with existing code or refactor both

## Working with AI Agents

When asking for help:
1. **Be Specific**: Point to files, functions, or error messages
2. **Provide Context**: Include stack traces, test failures, or energy trajectories
3. **Verify Results**: Always run pre-commit checks on generated code
4. **Iterate**: Break complex tasks into smaller, testable chunks

### Effective Prompting
- **Clear code pointers**: Use greppable identifiers, full stack traces, or code snippets
- **Include verification**: Provide reproduction steps and expected outcomes
- **Split large tasks**: Break down complex work into focused steps
- **Leverage for debugging**: Paste detailed logs for parallel analysis

## Migration Notes
<!-- Add temporary sections here during large refactors -->

## Quick Commands

```bash
# Find all TODOs
grep -r "TODO\|FIXME\|HACK" . --include="*.py"

# Check for circular imports
python -c "import energy_transformer; print('No circular imports')"

# Profile import time
python -X importtime -c "import energy_transformer" 2>&1 | grep energy_transformer

# Run specific test pattern
pytest tests/ -k "attention" -v
```

## Resources

- Main paper: "Energy Transformer" (Hoover et al., 2023)
- Issues: Check GitHub issues for known problems
- Discussions: See GitHub discussions for design decisions