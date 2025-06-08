Concepts
========

Energy Transformer builds on explicit energy functions to compute attention,
layer normalization and memory updates. Each layer minimizes its energy to
produce new representations, enabling associative memory and efficient
learning.

Key components include:

- :class:`energy_transformer.layers.MultiheadEnergyAttention`
- :class:`energy_transformer.layers.EnergyLayerNorm`
- :class:`energy_transformer.layers.HopfieldNetwork`
- :class:`energy_transformer.layers.SimplicialHopfieldNetwork`

For an overview of the architecture see Hoover et al., *Energy Transformer*,
2023.
