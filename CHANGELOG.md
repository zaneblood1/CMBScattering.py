# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). No version has been tagged yet: everything below is the development history of the upcoming `0.1.0`, newest first, with the date each change landed.

## [Unreleased]

### Added

- **2026-11-17** `docs/01_simulating_data.ipynb` has a new section on non-Gaussian dust, "Non-Gaussian dust from PySM": the full-sky PySM `d9` polarized intensity at 353 GHz before and after the Galactic plane mask with the outline of the flat patch, the projected E and B patch before and after the mean subtraction and taper, and a check that two seeds give identical dust (`d9` has no seed, so there is one dust realization). The notebook now needs the `pysm` extra to execute.

- **2026-11-17** `docs/03_sampling_pysm_dust.ipynb`: an executed comparison of the Gaussian sampler on two data sets that share their CMB and noise, one with PySM dust and one with Gaussian dust whose power laws are fitted to it. On the one map tested, 2000 iterations each, r is 0.0323 +/- 0.0043 with Gaussian dust and 0.0324 +/- 0.0045 with PySM dust (truth 0.03).

- **2026-11-17** Non-Gaussian dust in the simulated data: `simulate_data(..., dust_model = "pysm")` replaces the Gaussian dust draw with the 353 GHz dust of a PySM model (default `d9`) on the BICEP/Keck field. The Galactic plane is masked, the map is split into E and B on the sphere, and the two scalar maps are read onto the flat patch, mean-subtracted and tapered to zero at its edges (`galactic_plane_mask`, `pysm_dust_eb_maps`, `flat_patch_sky_coordinates`, `edge_taper`, `pysm_dust_field` in `simulate_dust.py`), then scaled across the channels with the same `a_freq` as the Gaussian dust. The sampler's command line has matching `--dust_model` and `--pysm_preset` flags, and `pyproject.toml` an optional `pysm` extra (`pysm3`, `healpy`).

- **2026-11-17** `/update-docs`, a project skill (`.claude/skills/update-docs/SKILL.md`) that tells Claude Code how to bring the markdown documentation in line with the code. `.gitignore` now tracks `.claude/skills/` while still ignoring the rest of `.claude/`.
- **2026-11-17** `docs/02_sampling_gaussian_dust.ipynb`: an executed guide that simulates one data set at the fiducial parameters, runs the Gaussian dust sampler for 1000 iterations and shows the trace plots, the log-posterior over time, the acceptance rates against the 44 % target, autocorrelation functions and integrated autocorrelation times, and KDE posteriors of the burnt-in, thinned samples with each parameter's mean, standard deviation (from `cmb_lensing.chain_analysis.product_moments`) and z-score.
- **2026-11-17** The sampler writes a trace plot per sampled parameter (`<out_dir>/<name>_trace.png`, with the true value marked when known), refreshed every `log_every` iterations while the chain runs. Plots are written to a temporary file and renamed into place, so an image viewer watching them does not flicker.
- **2026-11-17** `cmb_scattering/sampler.py`: `sample_gaussian_posterior`, a Gibbs sampler of the Gaussian dust model. Each iteration draws the CMB field and the dust field with the Hoffman-Ribak algorithm (exact mode-by-mode Wiener filter) and updates the two dust amplitudes, the two dust indices, r and the frequency exponent by sequential 1D random-walk Metropolis. Runs from the terminal with `python -m cmb_scattering.sampler`, writing the sample histories and a log-posterior plot to `gaussian_sampler_output/`. The default proposal widths are hand-tuned to accept close to 44 % on the default configuration.
- **2026-11-17** `CHANGELOG.md`, `CONTRIBUTING.md`, `PROJECT_SUMMARY.md` (a briefing for AI agents and new contributors) and `CLAUDE.md`.
- **2026-11-17** `docs/01_simulating_data.ipynb`: an executed guide that simulates a data set on a 42.7 degree patch with 20' pixels and plots the B-mode spectrum of each component and the CMB, dust and data maps.
- **2026-11-17** The Gaussian prior on the dust frequency exponent (BK18's beta_d = 1.49 +/- 0.13) is part of the log-posterior.
- **2026-11-17** Separate amplitude and power-law index for the EE and BB dust spectra (`dust_amplitude_ee`, `dust_knee_ee`, `dust_amplitude_bb`, `dust_knee_bb`), with Planck 2018 XI's measured slopes and BB/EE ratio as reference constants.
- **2026-11-17** Links to the source paper next to every physical constant.
- **2026-11-17** `cmb_scattering/likelihoods.py`: `gaussian_posterior`, the joint log-density of the data, the CMB field and the dust field, and `frequency_exponent_logprior`.
- **2026-11-17** `cmb_scattering/simulate_dust.py`: the three-frequency, E + B data model `d = M B (f + a g) + n` with a frequency-independent Gaussian CMB of covariance `C_scalar + r * C_tensor`, a Gaussian dust field scaled between channels by a modified blackbody in uK_CMB, per-channel white noise and identity mask and beam (`simulate_data`, `cmb_covariances`, `dust_frequency_scaling`, `white_noise_covariance`, `data_model`).
- **2026-11-17** Python packaging: `pyproject.toml` (setuptools), the `cmb_scattering` package, import smoke tests and `.gitignore`. CMBLensing.py is a dependency, installed from GitHub.

### Changed

- **2026-11-17** The guide notebooks are renamed and numbered in reading order: `docs/data_generation.ipynb` is now `docs/01_simulating_data.ipynb`, `docs/sampler_gaussian_dust.ipynb` (briefly `gaussian_dust_gaussian_sampler.ipynb`) is now `docs/02_sampling_gaussian_dust.ipynb`, and `non_gaussian_dust_gaussian_sampler.ipynb` is now `docs/03_sampling_pysm_dust.ipynb`. The other entries here use the new names.

- **2026-11-17** `simulate_data` takes `frequency_exponent` as its fifth positional argument, and `dust_amplitude_ee`, `dust_knee_ee`, `dust_amplitude_bb`, `dust_knee_bb` are now optional keywords that apply to Gaussian dust only. Positional calls with the old order must be updated; `docs/01_simulating_data.ipynb` was.

- **2026-11-17** `simulate_dust._draw_field` / `_draw_noise` are public (`draw_field`, `draw_noise`), since the sampler shares them.
- **2026-11-17** The dust power law is defined in `D_ell = ell (ell + 1) C_ell / (2 pi)`, the BK18 convention, instead of in `C_ell`.
- **2026-11-17** `frequency_exponent_logprior` is `-(x - mean)^2 / (2 width^2)` with `width` the 1-sigma uncertainty; the first stub divided by `width` once.
