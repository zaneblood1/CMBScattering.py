# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). No version has been tagged yet: everything below is the development history of the upcoming `0.1.0`, newest first, with the date each change landed.

## [Unreleased]

### Added

- **2026-11-17** `CHANGELOG.md`, `CONTRIBUTING.md`, `PROJECT_SUMMARY.md` (a briefing for AI agents and new contributors) and `CLAUDE.md`.
- **2026-11-17** `docs/data_generation.ipynb`: an executed guide that simulates a data set on a 42.7 degree patch with 20' pixels and plots the B-mode spectrum of each component and the CMB, dust and data maps.
- **2026-11-17** The Gaussian prior on the dust frequency exponent (BK18's beta_d = 1.49 +/- 0.13) is part of the log-posterior.
- **2026-11-17** Separate amplitude and power-law index for the EE and BB dust spectra (`dust_amplitude_ee`, `dust_knee_ee`, `dust_amplitude_bb`, `dust_knee_bb`), with Planck 2018 XI's measured slopes and BB/EE ratio as reference constants.
- **2026-11-17** Links to the source paper next to every physical constant.
- **2026-11-17** `cmb_scattering/likelihoods.py`: `gaussian_posterior`, the joint log-density of the data, the CMB field and the dust field, and `frequency_exponent_logprior`.
- **2026-11-17** `cmb_scattering/simulate_dust.py`: the three-frequency, E + B data model `d = M B (f + a g) + n` with a frequency-independent Gaussian CMB of covariance `C_scalar + r * C_tensor`, a Gaussian dust field scaled between channels by a modified blackbody in uK_CMB, per-channel white noise and identity mask and beam (`simulate_data`, `cmb_covariances`, `dust_frequency_scaling`, `white_noise_covariance`, `data_model`).
- **2026-11-17** Python packaging: `pyproject.toml` (setuptools), the `cmb_scattering` package, import smoke tests and `.gitignore`. CMBLensing.py is a dependency, installed from GitHub.

### Changed

- **2026-11-17** The dust power law is defined in `D_ell = ell (ell + 1) C_ell / (2 pi)`, the BK18 convention, instead of in `C_ell`.
- **2026-11-17** `frequency_exponent_logprior` is `-(x - mean)^2 / (2 width^2)` with `width` the 1-sigma uncertainty; the first stub divided by `width` once.
