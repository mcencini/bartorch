"""
bartorch.core — BartTensor, execution context, and DSL dispatch graph.

Classes defined here are the foundation of the hot-path DSL:

* ``BartTensor``:  a ``torch.Tensor`` subclass tagged for hot-path dispatch.
* ``BartContext``: thread-local execution context (manages the in-memory CFL
  registry session for a batch of chained BART operations).
* ``dispatch``:    routes an operation to the hot path or fallback based on
  the types of its inputs.
"""
