# Project summary

A briefing for AI agents (and new contributors) on what this repository is, how the code is laid out and what is easy to get wrong. Working rules and code style are in `CLAUDE.md`; the change history is in `CHANGELOG.md`. Update this file when the structure or the conventions change.

## Goal

Infer the tensor-to-scalar ratio r from flat-sky, multi-frequency CMB polarization maps in the presence of Galactic dust, eventually using scattering transform summary statistics to handle non-Gaussian dust. The package builds on [CMBLensing.py](https://github.com/zaneblood1/CMBLensing.py) (`cmb_lensing`, a JAX port of CMBLensing.jl) and reuses its field and operator types.

## Status (2026-11-17)

| Part | State |
|---|---|
| Gaussian dust simulation (`simulate_dust.py`) | implemented, runs end to end through `docs/data_generation.ipynb` |
| Gaussian log-posterior (`likelihoods.py`) | implemented, evaluated once on simulated data, not otherwise validated |
| Gaussian-case sampler (`sampler.py`) | implemented and runs (`docs/sampler_gaussian_dust.ipynb`); NOT validated: only short single-map chains so far |
| Scattering transform statistics, non-Gaussian dust | not started (see "Planned work") |
| Tests | two import smoke tests only (`tests/test_import.py`) |

## Planned work

From `#TODO` comments in the source (2026-11-17):

- `simulate_dust.simulate_data`: a flag to switch between the Gaussian dust and a more advanced (non-Gaussian) dust model, with every caller updated.
- `likelihoods.py`: a non-Gaussian log-posterior built on the L2 loss between the log scattering coefficients of a dust sample and those of the ground-truth dust field.

## Layout

```
cmb_scattering/
  __init__.py         empty
  simulate_dust.py    the data model: constants, covariances, dust frequency scaling, simulate_data
  likelihoods.py      gaussian_posterior and the beta_d prior
  sampler.py          sample_gaussian_posterior (Gibbs over f, g and the six parameters) and its CLI
docs/
  data_generation.ipynb   executed guide: simulate a data set and plot spectra and maps
  sampler_gaussian_dust.ipynb   executed guide: a 1000-iteration chain on one map, with traces,
                          acceptance rates, autocorrelation times and KDE posteriors
tests/
  test_import.py      import smoke tests
.claude/skills/update-docs/SKILL.md   the /update-docs command (the only tracked part of .claude/)
pyproject.toml        setuptools build; depends on cmb-lensing (from GitHub), jax, scipy
README.md, CONTRIBUTING.md, CHANGELOG.md, CLAUDE.md, PROJECT_SUMMARY.md
```

The sampler's default output directory, `gaussian_sampler_output/`, is gitignored.

## The data model

Three frequency channels (default 95 / 150 / 220 GHz), polarization only (E and B), maps in uK_CMB:

    d_freq = M_freq * B_freq * (f + a_freq * g) + n_freq

- **f**, the CMB, is the same in every channel. `f ~ N(0, C_f(r))` with `C_f(r) = scalar_covariance + r * tensor_covariance`. The scalar covariance is the LENSED scalar spectrum (lensing treated as Gaussian) and the tensor covariance is CAMB's tensor spectrum at r = 1. Both come from one CAMB run in `cmb_covariances`.
- **g**, the dust, is defined at 353 GHz. `g ~ N(0, C_g)` with a separate power law for EE and BB, in the BK18 convention `D_ell = ell (ell + 1) C_ell / (2 pi) = amplitude * (ell / 80)^knee`. Parameters: `dust_amplitude_ee`, `dust_knee_ee`, `dust_amplitude_bb`, `dust_knee_bb`.
- **a_freq** (`dust_frequency_scaling`) is a modified blackbody relative to 353 GHz, `(freq / 353)^beta_d * B_freq(T_d) / B_353(T_d)`, times the brightness to uK_CMB conversion `[dB/dT at 353] / [dB/dT at freq]` at T_CMB. `beta_d` is called `frequency_exponent` in the code.
- **n_freq** is white noise, the same level for E and B in every channel and independent between channels.
- **M, B**, the mask and beam, are identity operators for now.

`simulate_data(nside, theta_pix, master_seed, r, dust_amplitude_ee, dust_knee_ee, dust_amplitude_bb, dust_knee_bb, frequency_exponent, ...)` returns a dict with the fields `data`, `cmb_field`, `dust_field`, the operators `noise_covariance`, `scalar_covariance`, `tensor_covariance`, `field_covariance`, `dust_covariance`, `mask`, `beam`, and `frequencies` / `frequency_scaling`.

`gaussian_posterior(data_field, cmb_field, dust_field, mask, beam, dust_amplitude_ee, dust_knee_ee, dust_amplitude_bb, dust_knee_bb, frequency_exponent, r, noise_covariance, scalar_covariance, tensor_covariance, ...)` is the joint log-density of the data and the two fields, `-1/2 * [chi^2 terms + log-determinants]`, PLUS the Gaussian prior on `frequency_exponent` (BK18's beta_d = 1.49 +/- 0.13). There is no way to switch the prior off or change it through this function's arguments.

## The sampler

`sampler.sample_gaussian_posterior(data, noise_covariance, scalar_covariance, tensor_covariance, mask, beam, param_init, ...)` Gibbs-samples P(f, g, parameters | d). The parameters are named as everywhere else and updated in `PARAM_ORDER`: `dust_amplitude_ee`, `dust_amplitude_bb`, `dust_knee_ee`, `dust_knee_bb`, `r`, `frequency_exponent`. The dust temperature is fixed. It returns `(sample_histories, acceptance_rates, logposterior_values)`.

- **Field draws** (`gibbs_sample_field`, used for f and then g): Hoffman-Ribak. Simulate a field and noise from their priors, then add `wiener_filter` of (residual data - simulated data). The model has no lensing and the mask and beam are Fourier-diagonal, so the Wiener filter is solved exactly per mode; there is no conjugate gradient. A pixel-space mask would need one.
- **Parameter updates** (`metropolis_sample_parameter`): 1D random-walk Metropolis on `gaussian_posterior` at the current fields, with flat prior boxes `PARAM_BOUNDS` and widths `DEFAULT_PROPOSAL_WIDTHS` (hand-tuned on the default configuration, see below).
- **CLI:** `python -m cmb_scattering.sampler --nside 64 --num_iterations 2000 --r_proposal_width 0.01`. Every parameter has `--<name>` (truth of the simulated data, default `DEFAULT_TRUE_PARAMS`), `--<name>_init` and `--<name>_proposal_width`; the defaults are nside 128, 20' pixels and 5000 iterations. Output goes to `--out_dir` (default `gaussian_sampler_output/`): `sample_histories.npz`, the log-posterior plot `logpdf_over_time.png` and one trace plot per sampled parameter (`<name>_trace.png`). Every `--log_every` iterations (default 1, i.e. every iteration) the acceptance rates are printed and the plots rewritten; the plots are written to a temporary file and renamed into place so an image viewer never loads a partial file. Pass a larger `--log_every` for long chains, since each refresh redraws seven figures.
- **Mixing, measured 2026-11-17** (nside 128, 20', 3 uK-arcmin, default truth; about 36 iterations per second on the user's laptop CPU). r mixes fast (IAT about 6 iterations). The dust parameters and `frequency_exponent` mix slowly: a 6000-iteration chain gave IAT about 60 for the EE dust amplitude and knee, 130 to 220 for the BB ones and about 250 for `frequency_exponent`. A 500-iteration chain therefore underestimates their posterior widths (by a factor of four for `frequency_exponent`) and gives z-scores of 2 to 4 that a longer chain removes. Each of these parameters is updated with the dust field fixed, and the field with the parameters fixed; a parametrization that moves them together (as CMBLensing.py's mixing does) is the likely fix.
- **Proposal widths** near the 44 % target on that configuration: `dust_amplitude_ee` 0.2, `dust_amplitude_bb` 0.1, both knees 0.02, `r` 0.008, `frequency_exponent` 0.004. These are now `DEFAULT_PROPOSAL_WIDTHS`, which the notebook uses.
- **Not validated:** no many-map test that z-scores scatter with unit width has been run. One long chain on one map had `dust_amplitude_ee` 2.6 sigma low at about 90 effective samples; unexplained, possibly chance.

## Representation and conventions

- **Fields** are `cmb_lensing.fields.FlatS2` in EB parametrization and Fourier basis: E in `polar_matrix_1`, B in `polar_matrix_2`, each an `(nside, nside//2+1)` rfft2 array.
- **Operators** are `cmb_lensing.matrix_operators.DiagonalEB` (`matrix_EE`, `matrix_BB`), diagonal in Fourier space with no EB coupling.
- **Frequency axis.** Anything per channel (`data`, `noise_covariance`, and in the future a per-channel mask or beam) carries a leading axis: `(n_freq, nside, nside//2+1)`. `cmb_lensing`'s field and operator arithmetic, `dot`, `pinv` and `log_det` are element-wise sums, so they broadcast over it; the sums over channels happen inside `dot` and `log_det`. Noise correlated between channels cannot be represented.
- **Covariance units.** A covariance matrix entry is `C_ell / pix_width**2` (pix_width in radians), the convention of `cmb_lensing.simulate.covar_matrix_from_cls`. Multiply by `pix_width**2` to get C_ell. The (0, 0) entry (the map mean) is zero in every covariance, and `pinv` / `log_det` skip zero entries.
- **Maps.** `np.fft.irfft2(matrix)` gives the real-space map in uK.
- **Patch geometry.** `ell_min = 2 pi / (nside * pix_width)` and the Nyquist multipole is `pi / pix_width`. The tensor signal is at ell below about 150, so use coarse pixels: nside 128 at 20' gives a 42.7 degree patch covering ell 8 to 540, while 2.5' pixels start at ell 68.
- **Noise level.** `uk_arcmin` is the polarization noise level itself. `cmb_lensing.load_sim` instead takes a temperature level and makes polarization sqrt(2) higher.
- **Randomness.** `simulate_data` splits `PRNGKey(master_seed)` into five keys (CMB E, CMB B, dust E, dust B, noise); the noise key is split again per component and channel.

## Things that are easy to get wrong

- **The local CMBLensing.py checkout differs from GitHub.** `~/CMBLensing.py` has uncommitted changes, including a temporary change to the key argument of `simulate.field_from_covar`. This package therefore uses `field_from_covar_single_key`, which is the same in both. The declared dependency is the GitHub `master`; `pip install -e ../CMBLensing.py` switches an environment to the local checkout.
- **Importing `cmb_lensing.chain_analysis` switches matplotlib to Agg.** It calls `matplotlib.use("Agg")` at import time, so figures stop appearing in a notebook. `docs/sampler_gaussian_dust.ipynb` imports `product_moments` / `product_pdf` from it and runs `%matplotlib inline` afterwards to switch back. The clean fix is in CMBLensing.py (move the call into its command-line entry point), which this project does not edit.
- **Which `cmb_lensing` is installed.** As of 2026-11-17 `~/venvs/myenv` has the local `~/CMBLensing.py` checkout installed in editable mode, not the GitHub copy the dependency declares.
- **Private imports.** `simulate_dust.py` imports `_run_camb` and `_CAMB_COLS` from `cmb_lensing.simulate`. A rename there breaks `cmb_covariances`.
- **Scalar times operator.** Write `operator * r`, not `r * operator`, so the operator's own `__mul__` handles a JAX scalar.
- **Default-sized objects.** A bare `FlatS2()` or `DiagonalEB()` has nside 256 arrays and weights. Always pass `nside`, `theta_pix`, `pix_width` and `fourier_weights = get_fourier_weights((nside, nside//2+1))`, or build from an existing object with `.replace(...)`.
- **Log interpolation.** `covar_matrix_from_cls` interpolates in log(C_ell) and returns NaN for zeros. CAMB's tensor spectra are zero above `max_l_tensor = 600`, so `_covariance_from_cls` restricts to the positive multipoles and zeroes the matrix beyond them. It uses boolean indexing and so cannot run under `jit`.
- **CAMB is eager.** `cmb_covariances` and `simulate_data` call CAMB directly and are not jittable. `gaussian_posterior` only does array operations.
- **Negative r** makes the BB covariance negative at low ell. Nothing guards against it.
- **Flat sky.** A patch of tens of degrees stretches the flat-sky approximation; it is a toy geometry.

## Reference numbers for sanity checks

From a run with the notebook's settings (nside 128, 20' pixels, r = 0.03, BB dust 4.4 uK^2 with knee -0.66, beta_d = 1.49, 3 uK-arcmin) at `master_seed = 0`; the rms values depend on the seed:

- `frequency_scaling` = 0.0194, 0.0494, 0.1413 at 95, 150, 220 GHz
- tensor `D_ell^BB` at ell near 80 for r = 1: about 0.08 uK^2; lensing `D_ell^BB` at ell near 80: about 0.003 uK^2
- B-map rms: CMB about 0.26 uK, dust at 353 GHz about 5.9 uK

## Sources of the physical constants

Each constant has a comment with its source next to it in the code.

- BK18: BICEP/Keck XIII, <https://arxiv.org/abs/2110.00483> (bands, 353 GHz and ell = 80 pivots, T_d = 19.6 K, beta_d = 1.49 +0.13 / -0.12, A_d = 4.4 uK^2, alpha_d = -0.66)
- Planck 2018 XI, polarized dust foregrounds, <https://arxiv.org/abs/1801.04945> (EE and BB dust slopes, BB/EE = 0.53)
- Fixsen 2009, <https://arxiv.org/abs/0911.1955> (T_CMB = 2.72548 K)
