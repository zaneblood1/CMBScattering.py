# Project summary

A briefing for AI agents (and new contributors) on what this repository is, how the code is laid out and what is easy to get wrong. Working rules and code style are in `CLAUDE.md`; the change history is in `CHANGELOG.md`. Update this file when the structure or the conventions change.

## Goal

Infer the tensor-to-scalar ratio r from flat-sky, multi-frequency CMB polarization maps in the presence of Galactic dust, eventually using scattering transform summary statistics to handle non-Gaussian dust. The package builds on [CMBLensing.py](https://github.com/zaneblood1/CMBLensing.py) (`cmb_lensing`, a JAX port of CMBLensing.jl) and reuses its field and operator types.

## Status (2026-11-17)

| Part | State |
|---|---|
| Gaussian dust simulation (`simulate_dust.py`) | implemented, runs end to end through `docs/data_generation.ipynb` |
| Gaussian log-posterior (`likelihoods.py`) | implemented, evaluated once on simulated data, not otherwise validated |
| Sampler (`sampler.py`) | empty file, deliberately not started |
| Scattering transform statistics, non-Gaussian dust | not started |
| Tests | two import smoke tests only (`tests/test_import.py`) |

## Layout

```
cmb_scattering/
  __init__.py         empty
  simulate_dust.py    the data model: constants, covariances, dust frequency scaling, simulate_data
  likelihoods.py      gaussian_posterior and the beta_d prior
  sampler.py          empty
docs/
  data_generation.ipynb   executed guide: simulate a data set and plot spectra and maps
tests/
  test_import.py      import smoke tests
pyproject.toml        setuptools build; depends on cmb-lensing (from GitHub), jax, scipy
```

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
