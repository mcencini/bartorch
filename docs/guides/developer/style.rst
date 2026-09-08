Style guide
===========

Python follows the repository's Ruff configuration: 100-column lines,
formatter-managed layout and sorted imports. Use descriptive names and public
type hints. Keep tensor shapes, devices, normalization, mutation and ownership
explicit at native boundaries. Use plain C in the exported ABI, follow existing
local implementation conventions, and keep upstream BART unchanged.

Wrapper docstrings use NumPy style: a one-line summary followed by Parameters,
Returns and Raises where applicable. Document dtype, C-order shape, units,
default behavior and differentiability when they affect the caller. For example
(a documentation template, not a public function):

.. code-block:: python

   def apply_encoding(image, operator):
       """Apply a fixed MRI encoding operator.

       Parameters
       ----------
       image : torch.Tensor
           Complex64 image in C order, matching ``operator.ishape``.
       operator : LinearOperator
           Fixed encoding with matching device-resident operands.

       Returns
       -------
       torch.Tensor
           Complex64 samples of shape ``operator.oshape``.

       Raises
       ------
       ValueError
           If the input cannot be represented in the operator's domain.
       """
       return operator(image)

Document implemented behavior. Separate proposed methods from available APIs
and paper reproductions from teaching phantoms. Use RST for pages and gallery
narrative blocks for examples. Label plots with units and normalization.
