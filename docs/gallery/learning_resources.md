# BART workshops and webinars

The [BART tutorial catalogue](https://mrirecon.codeberg.page/tutorials.html)
and [webinar archive](https://mrirecon.codeberg.page/webinars.html) provide
recordings, scripts and datasets.  This reading order connects them to
bartorch.  Their command-line, MATLAB and Python scripts need their filenames,
dimensions and option syntax translated before they run here.

| Learning goal | Upstream material | bartorch progression |
| --- | --- | --- |
| First reconstruction | [Webinar 4 (July 2021)](https://gitlab.tugraz.at/ibi/mrirecon/tutorials/bart-webinars/tree/master/webinar4) | Phantom, calibration, coil preparation, the `pics` example. |
| Noise processing and parallel imaging | [ISMRM 2019 workshop](https://gitlab.tugraz.at/ibi/mrirecon/tutorials/bart-workshop/tree/master/ismrm2019) | Whiten and compress consistently; then measured data and g-factor. |
| Non-Cartesian reconstruction | Webinar 7 (December 2022), in the webinar archive | The radial operator and its direct Fourier check, then density compensation, sampling weights and measured trajectories. |
| Temporal subspaces and signal models | [Webinar 3 (March 2021)](https://gitlab.tugraz.at/ibi/mrirecon/tutorials/bart-webinars/tree/master/webinar3) | The model-based example, then basis contraction and quantitative mapping. |
| Dynamic imaging | Webinar 2 (December 2020), in the webinar archive | Explicit time axes, data-consistent regularization, then GRASP-like applications. |
| Wave reconstruction | [ISMRM 2016 Wave-CS](https://gitlab.tugraz.at/ibi/mrirecon/tutorials/bart-workshop/tree/master/ismrm2016/wave-cs) | The hybrid-space wave model and the measured-data recipe in {doc}`research`. |
| Learning-based reconstruction | Webinar 6 (March 2022), in the webinar archive | BART's network workflows beside the gallery's DeepInverse physics and autograd. |
| Volumetric reconstruction | Webinar 8 (April 2024), in the webinar archive | Axis-aware examples in 3-D, with memory measured. |

Follow the upstream links for current data locations and licenses.  The
gallery scripts are newly written synthetic examples and redistribute no
workshop data or figures.
