Installation
============

Requirements
------------

- Python 3.11 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU support)

Using pip
---------

.. code-block:: bash

   pip install energy-transformer

Using Poetry
------------

.. code-block:: bash

   poetry add energy-transformer

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/b-vitamins/energy-transformer.git
   cd energy-transformer
   poetry install

Verifying Installation
----------------------

.. code-block:: python

   import energy_transformer
   print(energy_transformer.__version__)
