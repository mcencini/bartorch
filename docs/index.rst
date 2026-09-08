bartorch
========

MRI reconstruction, expressed in Python and executed by BART on PyTorch tensors.

Use a prepackaged reconstruction, compose an encoding operator, or connect a
PyTorch signal model to BART's solvers. The embedded library runs in your Python
process; tensors carry data between the tools and your application.

.. note::

   This documentation describes the development checkout (|release|). The API
   and wheel distribution are under development. Gallery pages distinguish
   synthetic examples from research reproduction plans.

.. code-block:: python

   import bartorch.tools as bt

   kspace = bt.phantom([32, 32], kspace=True, ncoils=4)
   maps = bt.ecalib(kspace, calib_size=16, maps=1)
   image = bt.pics(kspace, maps, l=2, lambda_=0.001, iter_=30)

Start with :doc:`guides/user/installation`, follow the
:doc:`auto_examples/index`, or look up a callable in :doc:`api/index`.

.. toctree::
   :maxdepth: 2
   :caption: Guides

   guides/user/index
   guides/developer/index

.. toctree::
   :maxdepth: 2
   :caption: Example Gallery

   auto_examples/index
   gallery/encoding
   gallery/research
   gallery/learning_resources

.. toctree::
   :maxdepth: 2
   :caption: API References

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Miscellaneous

   misc/license
   misc/related_projects
   misc/contributors

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
