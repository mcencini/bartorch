

.. _sphx_glr_auto_examples_01-basics:

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


.. toctree::
   :hidden:

   /auto_examples/01-basics/01-from-kspace-to-image
   /auto_examples/01-basics/02-operators-and-solvers
   /auto_examples/01-basics/03-noise-prewhitening

