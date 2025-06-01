# Specification Realisation System

## Overview

The Energy Transformer uses a specification system that separates model
architecture definition (specs) from concrete implementations. A spec describes
what to build and the realiser converts it into PyTorch modules.

## Core Components

### Specifications
- **ETBlockSpec**: Energy Transformer block specification
- **MHEASpec**: Multi-Head Energy Attention specification
- **HNSpec**: Hopfield Network specification
- **SHNSpec**: Simplicial Hopfield Network specification

### Realisation
Specifications are converted to modules using registered realisers or the
auto-import mechanism.

```python
spec = MHEASpec(num_heads=8, head_dim=64)
module = realise(spec, embed_dim=512)
```

For composite specs like `ETBlockSpec`, a custom realiser constructs the full
`EnergyTransformer`:

```python
@register(ETBlockSpec)
def realise_et_block(spec, context):
    attention = realise(spec.attention, context)
    hopfield = realise(spec.hopfield, context)
    return EnergyTransformer(attention=attention, hopfield=hopfield, ...)
```

## Extending the System

1. Define a Spec dataclass in `spec/library.py`.
2. Add an auto-import mapping or register a custom realiser with `@register`.
