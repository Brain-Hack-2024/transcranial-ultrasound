# Transcranial Functional Ultrasound

Code and other assets from the [brain hack](https://brainhack.vercel.app) for functional transcranial ultrasound.

The structure of this repo is as follows:
- `data` contains scan data for a cross-section of a human skull.
- `imaging` contains an implementation of image reconstruction via beamforming. The implementation is taken from [NeurotechDevKit](https://github.com/agencyenterprise/neurotechdevkit), with minor modifications to improve speed. We thank `NeurotechDevKit` for their robust beamforming implementation.
- `notebooks` contains various Jupyter notebooks used in image reconstruction experiments.
- `skull-attenuation-analysis` contains data and analysis code for an experiment quantifying attenuation of signals through the skull.
- `skull-propagation` contains a notebook for simulating skull propagation through the skull in 2D
- `syringepump` contains code to control a syringe pump and pump fluid through an artificial vein at a controlled rate.
- `utils` contains various code we wrote to assist with wave simulations, beamforming, and other tasks.

We use `jwave` and `kwave` for wave simulations, and standard libraries (`jax`, `matplotlib`, `numpy`, `pandas`, `scipy`) for scientific computing and data manipulation.
