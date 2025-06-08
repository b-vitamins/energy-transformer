Energy Transformer Documentation
================================

**Energy Transformer** is a PyTorch implementation of energy-based transformers with associative memory.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   concepts
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

Key Features
------------

- **Energy-based Components** – Attention, layer normalization, and memory are formulated as energy minimization
- **Direct Gradient Computation** – Efficient implementation without autograd overhead
- **Simplicial Networks** – Higher-order interactions through simplicial complexes
- **Vision Models** – Pre-configured ViET and ViSET models for computer vision

Installation
------------

.. code-block:: bash

   pip install energy-transformer

Quick Example
-------------

.. code-block:: python

   import torch
   from energy_transformer.models.vision import viet_base

   # Create model
   model = viet_base(img_size=224, patch_size=16, num_classes=1000)
   
   # Forward pass
   images = torch.randn(1, 3, 224, 224)
   logits = model(images)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
