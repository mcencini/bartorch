:orphan:

Examples
========

Reconstruction workflows written with bartorch, grouped by what they build.

**Basics** covers the data layout, BART's applications as functions, the same
reconstruction assembled from an encoding operator and a solver, and noise
prewhitening.

**Non-Cartesian imaging** covers trajectories, the non-uniform Fourier
transform and its point spread function, and a radial SENSE reconstruction.

**Applications** covers two problems whose encoding is more than sampling and
coils: a dynamic series reconstructed from golden-angle radial data, and a
subspace-constrained :math:`T_1` mapping experiment.

**Model-based reconstruction** estimates the unknowns jointly with the coil
sensitivities, or estimates parameter maps directly from k-space through a
signal model.

**Deep learning** replaces a specified regularizer with a learned one, by
unrolling a BART iteration and training its proximal step through it.

The concepts these examples use -- the encoding model, regularized
least squares, non-uniform transforms, nonlinear inversion -- are introduced in
:doc:`../explanation/index`.

Running them needs a built ``bartorch``, ``brainweb-dl``, which every example
builds its phantom from, and ``matplotlib`` and ``cmap`` for the figures::

    pip install bartorch brainweb-dl matplotlib cmap

The deep-learning section additionally requires::

    pip install lightning torchio monai deepinv


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

Basics
------

The conventions, the two ways of writing a reconstruction, and the channel
noise model of a SENSE reconstruction.

The first example follows undersampled Cartesian k-space to an image: the
sampling pattern, channel compression, ESPIRiT calibration, and reconstruction
with :func:`bartorch.tools.pics`. The second assembles the same problem out of
an encoding operator and a solver from :mod:`bartorch.optim`, which is how a
reconstruction BART has no application for is written. The third measures the effect
of correlated channel noise on a SENSE reconstruction, and its removal by
prewhitening with a noise measurement.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reconstruction of an undersampled Cartesian acquisition, from the measured k-space to a coil-combined image.">

.. only:: html

  .. image:: /auto_examples/01-basics/images/thumb/sphx_glr_01-from-kspace-to-image_thumb.png
    :alt:

  :doc:`/auto_examples/01-basics/01-from-kspace-to-image`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">From k-space to image</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The same reconstruction written as an encoding operator and a solver rather than as a call to a BART application.">

.. only:: html

  .. image:: /auto_examples/01-basics/images/thumb/sphx_glr_02-operators-and-solvers_thumb.png
    :alt:

  :doc:`/auto_examples/01-basics/02-operators-and-solvers`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Operators and solvers</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The effect of channel-noise correlation on a SENSE reconstruction, and its removal by prewhitening with a noise measurement.">

.. only:: html

  .. image:: /auto_examples/01-basics/images/thumb/sphx_glr_03-noise-prewhitening_thumb.png
    :alt:

  :doc:`/auto_examples/01-basics/03-noise-prewhitening`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Noise prewhitening</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Non-Cartesian imaging
---------------------

Sampling off the Cartesian grid.

The first example covers the trajectory and transform interfaces: the
trajectories :func:`bartorch.tools.traj` generates, the non-uniform Fourier
transform, density compensation, and the point spread function the normal
operator convolves with. The second reconstructs an undersampled radial
acquisition with :class:`bartorch.linop.NoncartesianSense`.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The non-Cartesian interfaces: the trajectories bartorch.tools.traj generates, the non-uniform Fourier transform along one, the density compensation an adjoint reconstruction needs, and the point spread function the normal operator convolves with.">

.. only:: html

  .. image:: /auto_examples/02-non-cartesian/images/thumb/sphx_glr_01-trajectories-and-transforms_thumb.png
    :alt:

  :doc:`/auto_examples/02-non-cartesian/01-trajectories-and-transforms`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Trajectories and transforms</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An undersampled golden-angle radial acquisition reconstructed by regularized least squares, with the non-Cartesian SENSE operator">

.. only:: html

  .. image:: /auto_examples/02-non-cartesian/images/thumb/sphx_glr_02-radial-sense_thumb.png
    :alt:

  :doc:`/auto_examples/02-non-cartesian/02-radial-sense`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Radial SENSE reconstruction</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Applications
------------

Two acquisitions whose encoding carries more than sampling and coils.

A golden-angle radial series is reconstructed jointly under a temporal total
variation penalty, and an inversion-recovery experiment is reconstructed into the
coefficients of a signal subspace, from which :math:`T_1` is estimated.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A continuously acquired golden-angle radial scan reconstructed as a time series, with a temporal regularizer compensating for the undersampling of each frame.">

.. only:: html

  .. image:: /auto_examples/03-applications/images/thumb/sphx_glr_01-dynamic-golden-angle_thumb.png
    :alt:

  :doc:`/auto_examples/03-applications/01-dynamic-golden-angle`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dynamic golden-angle radial MRI</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An inversion-recovery FLASH acquisition of four hundred frames, one spoke each, reconstructed into the coefficients of a signal subspace and fitted for T_1.">

.. only:: html

  .. image:: /auto_examples/03-applications/images/thumb/sphx_glr_02-subspace-t1-mapping_thumb.png
    :alt:

  :doc:`/auto_examples/03-applications/02-subspace-t1-mapping`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Subspace-constrained T1 mapping</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Model-based reconstruction
--------------------------

Reconstructions whose forward operator is nonlinear in the unknowns.

Nonlinear inversion estimates the image and the coil sensitivities jointly
from undersampled k-space. A signal model in front of the encoding estimates
parameter maps from the measured data directly, without an intermediate series
of images.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Estimating the image and the coil sensitivities together, from undersampled data whose fully sampled central region is too small for a separate calibration.">

.. only:: html

  .. image:: /auto_examples/04-model-based/images/thumb/sphx_glr_01-nonlinear-inversion_thumb.png
    :alt:

  :doc:`/auto_examples/04-model-based/01-nonlinear-inversion`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nonlinear inversion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A multi-echo spin-echo acquisition fitted for T_2 in two ways: by reconstructing the echo images and fitting them afterwards, and by putting the signal model inside the forward operator and fitting the k-space directly.">

.. only:: html

  .. image:: /auto_examples/04-model-based/images/thumb/sphx_glr_02-quantitative-models_thumb.png
    :alt:

  :doc:`/auto_examples/04-model-based/02-quantitative-models`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Parameter maps straight from k-space</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

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
   :includehidden:


   /auto_examples/01-basics/index.rst
   /auto_examples/02-non-cartesian/index.rst
   /auto_examples/03-applications/index.rst
   /auto_examples/04-model-based/index.rst
   /auto_examples/05-deep-learning/index.rst



.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
