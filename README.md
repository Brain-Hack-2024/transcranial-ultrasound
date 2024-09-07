# Transcranial Functional Ultrasound

Code and other assets from the 2024 brain hack (todo: insert link to blog post) for functional transcranial ultrasound.

The structure of this repo is as follows:
- `imaging` contains an implementation of image reconstruction via beamforming. The implementation is taken from [NeurotechDevKit](https://github.com/agencyenterprise/neurotechdevkit), with minor modifications to improve speed. We thank `NeurotechDevKit` for their robust beamforming implementation.
- `notebooks` contains various Jupyter notebooks used in image reconstruction experiments.
- `skull-attenuation` contains data and analysis code for an experiment quantifying attenuation of signals through the skull.
- `syringepump` contains code to control a syringe pump and pump fluid through an artificial vein at a controlled rate.

root -> utils and fix notebook imports