Quick Start Guide
=================

This guide will help you get started with Energy Transformer.

Basic Usage
-----------

Vision Transformer (ViT)
~~~~~~~~~~~~~~~~~~~~~~~~

Standard Vision Transformer baseline:

.. code-block:: python

   from energy_transformer.models.vision import vit_small
   
   model = vit_small(
       img_size=224,
       patch_size=16,
       num_classes=1000
   )

Vision Energy Transformer (ViET)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Energy-based transformer with Hopfield memory:

.. code-block:: python

   from energy_transformer.models.vision import viet_small
   
   model = viet_small(
       img_size=224,
       patch_size=16,
       num_classes=1000
   )
   
   # Get energy information
   logits, (e_att, e_hop) = model(images, return_energies=True)

Vision Simplicial Transformer (ViSET)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With higher-order interactions:

.. code-block:: python

   from energy_transformer.models.vision import viset_small
   
   model = viset_small(
       img_size=224,
       patch_size=16,
       num_classes=1000,
       triangle_fraction=0.5  # Mix of edges and triangles
   )

Custom Models
-------------

Building from components:

.. code-block:: python

   from energy_transformer.layers import (
       EnergyLayerNorm,
       MultiheadEnergyAttention,
       HopfieldNetwork
   )
   from energy_transformer.models import EnergyTransformer
   
   et_block = EnergyTransformer(
       layer_norm=EnergyLayerNorm(768),
       attention=MultiheadEnergyAttention(768, num_heads=12),
       hopfield=HopfieldNetwork(768, hidden_dim=3072),
       steps=4
   )
