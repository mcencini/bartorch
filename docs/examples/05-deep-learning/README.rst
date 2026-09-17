Deep learning
-------------

Reconstructions whose regularizer is learned rather than written down.

An unrolled network is an iteration with a network inside it: the data term
stays the encoding operator and the solver, and what is learned is the
proximal step and the scalars around it. The page here builds MoDL that way,
with BART's alternating-direction iteration as the backbone, and trains it with
``lightning``, ``torchio`` and a denoiser from ``deepinv``.

Running it needs those three and ``monai``::

    pip install lightning torchio monai deepinv
