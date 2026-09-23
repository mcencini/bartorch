

.. _sphx_glr_auto_examples_05-deep-learning:

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


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An unrolled network for undersampled Cartesian SENSE: a convolutional denoiser in the proximal step of BART&#x27;s alternating-direction iteration, trained end to end against fully sampled images.">

.. only:: html

  .. image:: /auto_examples/05-deep-learning/images/thumb/sphx_glr_01-modl-with-admm_thumb.png
    :alt:

  :doc:`/auto_examples/05-deep-learning/01-modl-with-admm`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MoDL, on BART's ADMM</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/05-deep-learning/01-modl-with-admm

