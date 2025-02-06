.. include:: ../README.md
   :parser: myst_parser.sphinx_

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Tutorial:
   :glob:

   notebook/*.ipynb
 
.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API:
   :glob:
   :exclude: tests, __init__.py

   featuremap/*.py
