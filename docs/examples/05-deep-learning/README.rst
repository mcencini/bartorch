Deep learning
-------------

Reconstructions whose regularizer is learned rather than specified.

An unrolled network is an iteration with a network in its proximal step: the
data term remains the encoding operator and the solver, and what is learned is
the proximal step and the scalars accompanying it. MoDL is constructed this
way below, over BART's alternating-direction iteration, with ``deepinv``'s
DnCNN as the denoiser, and trained against fully sampled images with
``lightning`` and ``torchio``.

Running the section additionally requires::

    pip install lightning torchio monai deepinv
