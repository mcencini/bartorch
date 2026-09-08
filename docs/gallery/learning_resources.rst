Learning from BART workshops and webinars
=========================================

The `BART tutorial catalogue <https://mrirecon.codeberg.page/tutorials.html>`_
and `webinar archive <https://mrirecon.codeberg.page/webinars.html>`_ provide
recordings, scripts and datasets. The following reading order connects those
materials to bartorch. Existing CLI/MATLAB/Python command scripts require
translation of filenames, dimensions and option syntax; they are not native
bartorch apps without that adaptation.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Learning goal
     - Upstream material
     - bartorch progression
   * - First reconstruction
     - `Webinar 4 (July 2021)
       <https://gitlab.tugraz.at/ibi/mrirecon/tutorials/bart-webinars/tree/master/webinar4>`_
     - Phantom, calibration, coil preparation, PICS gallery.
   * - Noise processing and parallel imaging
     - `ISMRM 2019 workshop
       <https://gitlab.tugraz.at/ibi/mrirecon/tutorials/bart-workshop/tree/master/ismrm2019>`_
     - Whiten and compress consistently; later add measured data and g-factor.
   * - Non-Cartesian reconstruction
     - Webinar 7 (December 2022), linked in the webinar archive
     - Radial operator and direct Fourier check, then density compensation,
       sampling weights and measured trajectories.
   * - Temporal subspaces and signal models
     - `Webinar 3 (March 2021)
       <https://gitlab.tugraz.at/ibi/mrirecon/tutorials/bart-webinars/tree/master/webinar3>`_
     - Model-based gallery, then basis contraction and quantitative mapping.
   * - Dynamic imaging
     - Webinar 2 (December 2020), linked in the webinar archive
     - Explicit time axes, data-consistent regularization, then GRASP-like apps.
   * - Wave reconstruction
     - `ISMRM 2016 Wave-CS
       <https://gitlab.tugraz.at/ibi/mrirecon/tutorials/bart-workshop/tree/master/ismrm2016/wave-cs>`_
     - Hybrid-space wave model and measured-data recipe in :doc:`research`.
   * - Learning-based reconstruction
     - Webinar 6 (March 2022), linked in the webinar archive
     - Compare BART's network workflows with the gallery's explicit
       DeepInverse physics and autograd adapter.
   * - Volumetric reconstruction
     - Webinar 8 (April 2024), linked in the webinar archive
     - Extend axis-aware examples to 3-D and measure memory scaling.

Follow the upstream links for current data locations and licenses. These pages
link to teaching material; the gallery scripts are newly written synthetic
examples and do not redistribute workshop datasets or figures.
