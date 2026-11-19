# Project summary

A briefing for AI agents (and new contributors) on what this repository is, how the code is laid out and what is easy to get wrong. Working rules and code style are in `CLAUDE.md`; the change history is in `CHANGELOG.md`. Update this file when the structure or the conventions change.

## Goal

Infer the tensor-to-scalar ratio r from flat-sky, multi-frequency CMB polarization maps in the presence of Galactic dust, eventually using scattering transform summary statistics to handle non-Gaussian dust. The package builds on [CMBLensing.py](https://github.com/zaneblood1/CMBLensing.py) (`cmb_lensing`, a JAX port of CMBLensing.jl) and reuses its field and operator types.

## Status (2026-11-18)

| Part | State |
|---|---|
| Gaussian dust simulation (`simulate_dust.py`) | implemented, runs end to end through `docs/01_simulating_data.ipynb` |
| Gaussian log-posterior (`likelihoods.py`) | implemented, evaluated once on simulated data, not otherwise validated |
| Gaussian-case sampler (`sampler.py`) | implemented and runs (`docs/02_sampling_gaussian_dust.ipynb`); NOT validated: only short single-map chains so far |
| PySM (non-Gaussian) dust simulation (`simulate_data(dust_model = "pysm")`) | implemented; one map checked by eye and by its spectrum, and sampled once (`docs/03_sampling_pysm_dust.ipynb`) |
| Scattering transform posterior (`likelihoods.scattering_posterior`) and its sampler (`sampler.sample_scattering_posterior`) | implemented 2026-11-18 and run on one map (`docs/04_scattering_sampler_pysm_dust.ipynb`, and `docs/05_scattering_with_gaussian_prior.ipynb` for the `gaussian_dust_prior` flag); exploratory and NOT validated: the target coefficients are those of the true dust and the loss width is hand-chosen |
| Mask and beam (`instrument.py`, `mask` / `beam` arguments) | implemented 2026-11-18; identity results unchanged bit for bit; a pixel mask + BICEP3 beam run once (`docs/06_mask_and_beam.ipynb`), NOT validated |
| Tests | two import smoke tests only (`tests/test_import.py`) |

## Planned work

### Open points of the scattering posterior (2026-11-18)

The source has no `#TODO` comments left. What the first version of `scattering_posterior` leaves open:

- The loss is not a normalized density, so `frequency_exponent` cannot be sampled with it (see the sampler section). The fix, `gaussian_dust_prior = True` (keep the Gaussian dust prior next to the loss), stopped the runaway in a 300-iteration test; a chain of several thousand iterations, long enough for a beta_d posterior, has not been run. With the flag on, the integral of (Gaussian prior x loss) over g depends on the dust amplitudes and knees and is not divided out, so the loss acts as an implicit extra prior on them.
- The scattering term had almost no effect on r in the first test. A configuration where the dust prior matters (higher noise, fewer channels, a frequency scaling the sampler does not know) is needed to test it.
- The target coefficients come from the true dust field of the simulation, including that realization's own fluctuations (the Gaussian-sampler equivalent would be handing it the true band powers). A usable method needs a target from somewhere else (a dust model, other sky, or the synthesis below).
- **Proposed more agnostic dust term (discussed 2026-11-18, not implemented, not tested):** keep the Gaussian prior with its fitted amplitudes and knees, and add a scattering loss on amplitude-normalized coefficients (ratios such as S2 / S1, unchanged when the dust is rescaled), with the target mean and per-coefficient scatter taken from an ensemble of dust maps that excludes the analysed patch. This would also address the two points above and the width below. First check, not yet done: whether these shape statistics are similar across `d9` patches. Other options considered: a 353 GHz channel in the data vector as the dust tracer, and free target parameters (an overall log amplitude and a tilt), which again need a normalized density.
- One hand-chosen `scattering_width` for every coefficient. It should follow how much each coefficient varies between dust realizations (large-scale coefficients vary more).
- E and B are transformed separately; no coefficient couples them.
- `sample_scattering_posterior` has no command-line entry point (`python -m cmb_scattering.sampler` still runs the Gaussian sampler only) and does not return the sampled dust field.

### Open points of the mask and beam support (2026-11-18)

- The command line (`python -m cmb_scattering.sampler`) has no mask or beam flags; it runs with the identity.
- One beam for all channels in the notebook. A beam per channel (leading frequency axis) should broadcast through the code but has not been run.
- A `PixelMask` holds one map for all channels.
- The scattering loss acts on the whole dust patch, including the border the mask hides (decided 2026-11-18; a version that multiplied the dust maps by the mask before the transform was removed as unnecessary machinery). With a target computed from the true dust, the target therefore carries information about unobserved sky; with a target from a dust model it would not.
- The chain needs 200 to 300 iterations of burn-in under the mask, probably (not tested) because the dust starts from a crude estimate (`least_squares_dust_field` with the mask replaced by its rms). A Wiener-filter start would shorten it.
- The conjugate-gradient CMB draw converges slowly (about 0.8 s per draw at nside 128); a better preconditioner than the diagonal one would help.
- Not validated: one map, one 500-iteration chain. No many-map test under a mask.
- The Gaussian sampler (`sample_gaussian_posterior`) under a pixel mask has only been smoke-run: 20 iterations at nside 64 with Gaussian dust ran and gave finite samples. Its dust draw then also goes through conjugate gradient.

### Many non-Gaussian dust realizations by synthesis (exploratory, discussed 2026-11-18)

**Status: ideas only.** Nothing below is implemented or tested, the package choices are unverified, and the run-time and memorization statements are estimates. Treat it as a starting point: any part may need changing, and the whole route may be dropped for a different one (the alternatives are listed at the end).

**The problem.** PySM `d9` has no seed, so there is one dust realization. With `d9` fixed and many CMB + noise seeds, the mean of the per-map r estimates measures the bias *conditional on that one dust sky*, and a product of the per-map posteriors is not marginalized over dust. The target plot (scatter of per-map mean r over 100s of maps, Gaussian sampler against non-Gaussian sampler, with z-scores of unit width) needs 100s of statistically equivalent non-Gaussian dust maps.

**The idea.** Synthesize new flat-sky dust maps by gradient descent: start from a random map and minimize the distance between its wavelet statistics and those of a target dust patch. Generate with one statistic family (wavelet phase harmonics, WPH) and infer with the other (scattering coefficients) to reduce circularity. The two families carry much of the same information, so the real `d9` patch, which neither generated, stays the held-out test.

**Flat patch, not full sky.** Synthesize 128 x 128 periodic flat maps directly, instead of synthesizing full-sky maps and cutting patches from them. Reasons: full-sky statistics are dominated by bright low-latitude dust, only about 10 patches of 43 degrees fit per sky, patches at random positions differ in amplitude, and each patch would go through the projection, mask and taper again. FFT-based wavelets give periodic maps, so no taper is needed, and the realizations are identically distributed.

**Proposed loss**, minimized over one two-channel map (Q, U):

    loss = |wavelet stats(synthesis) - wavelet stats(target)|^2      texture
         + |cross stats between the polarization channels|^2         joint E / B
         + |binned EE, BB power(synthesis) - target band powers|^2   low-ell power

- **Variables.** Optimize Q and U, where filaments are local, and derive E and B inside the loss with the exact, differentiable Fourier rotation of a periodic flat grid. The texture term acts on the complex field Q + iU.
- **Joint E / B.** E and B must come from the same descent, since real dust puts correlated structure in both and the Gaussian sampler assumes they are independent. For parity-even dust the EB cross power spectrum is zero, so the cross term has to correlate wavelet *moduli* between channels (|E * psi| with |B * psi| at the same and neighbouring scales). Check: the correlation between smoothed E^2 and B^2 maps should match the `d9` patch (about zero for a Gaussian draw).
- **Low-ell power.** Wavelet bands are an octave wide and do not fix the spectrum's shape over ell 8 to 150, where the tensor signal is. Add EE and BB band powers in bins about one fundamental wide (delta ell about 8), with the rfft2 Fourier weights, at least up to ell 150, each bin weighted by its variance 2 / (number of modes).
- **Band-power targets.** Use the smooth power-law fit (the one `docs/03_sampling_pysm_dust.ipynb` makes), multiplied per realization and per bin by a chi-squared draw with that bin's number of modes. The patch's own band powers would copy its particular fluctuations into every map; the smooth fit alone would remove the dust sample variance at low ell and make the scatter of r too small.
- **Initialization.** Start each descent from a Gaussian draw with the fitted power law (`draw_field`).

**Known risks.**

- *Memorization.* A 128 x 128 map has about 16,000 pixels and a full WPH set can have thousands of coefficients; too many constraints from one target give near-copies of it. Use a reduced statistic set and check that syntheses from different seeds are not strongly correlated with each other or the target.
- *One target is a noisy estimate.* Averaging the target statistics over several `d9` patches, each normalized to a common amplitude, would help most.
- *Target edges.* The tapered patch is not a good target (the taper is a strong non-stationary feature), and the untapered patch is discontinuous at its periodic boundary. Some edge handling is needed.
- *Stationarity.* Synthesized maps are statistically uniform and lose the brightness gradient real dust has across the patch.

**Candidate tools (unverified):** [PyWPH](https://arxiv.org/abs/2102.03160) (Regaldo-Saint Blancard et al. 2021, WPH with cross moments, applied to Planck dust polarization), Kymatio for scattering transforms.

**Alternatives if this does not work out.** (1) Fixed `d9` with many CMB + noise seeds, paired with matched Gaussian dust: needs no new code and gives the conditional bias; worth doing first regardless. (2) More patches of the `d9` sky: about 10 at 43 degrees or 40 to 50 at 20 degrees, with very different amplitudes. (3) Projections of periodic MHD boxes ([Kritsuk, Flauger & Ustyugov 2018](https://arxiv.org/abs/1711.11108)): natively periodic flat maps, tens of realizations, need the simulation data. (4) A toy model, e.g. a Gaussian field times a smooth positive modulation, with the non-Gaussianity as a dial. PySM `d11` and [DustFilaments](https://arxiv.org/abs/2107.08317) are seedable only on small scales, with fixed large scales, so they do not help at the multipoles that constrain r.

## Layout

```
cmb_scattering/
  __init__.py         empty
  simulate_dust.py    the data model: constants, covariances, dust frequency scaling, simulate_data
  instrument.py       masks and beams: PixelMask, apodized_border_mask, gaussian_beam
  likelihoods.py      gaussian_posterior, scattering_posterior and the beta_d prior
  sampler.py          sample_gaussian_posterior (Gibbs over f, g and the six parameters) and its CLI;
                      sample_scattering_posterior (exact f draw, HMC for g, Metropolis for r and beta_d)
docs/
  01_simulating_data.ipynb   executed guide: simulate a data set and plot spectra and maps; its last
                          section shows the PySM dust step by step (needs the pysm extra)
  02_sampling_gaussian_dust.ipynb   executed guide: a 1000-iteration chain on one map, with traces,
                          acceptance rates, autocorrelation times and KDE posteriors
  03_sampling_pysm_dust.ipynb   executed comparison: the Gaussian sampler on PySM dust
                          against matched Gaussian dust, same CMB and noise
  04_scattering_sampler_pysm_dust.ipynb   executed comparison: the scattering sampler against the
                          Gaussian sampler on one PySM-dust data set
  05_scattering_with_gaussian_prior.ipynb   executed test: the scattering sampler with the Gaussian
                          dust prior kept (gaussian_dust_prior = True) and beta_d sampled
  06_mask_and_beam.ipynb   executed example: an apodized border mask and a BICEP3 beam, the data
                          built up component by component, and a 500-iteration scattering chain
tests/
  test_import.py      import smoke tests
.claude/skills/update-docs/SKILL.md   the /update-docs command (the only tracked part of .claude/)
pyproject.toml        setuptools build; depends on cmb-lensing (from GitHub), jax, scipy, kymatio;
                      optional extras: dev (pytest, jupyter), pysm (pysm3, healpy)
README.md, CONTRIBUTING.md, CHANGELOG.md, CLAUDE.md, PROJECT_SUMMARY.md
```

The sampler's default output directory, `gaussian_sampler_output/`, is gitignored.

## The data model

Three frequency channels (default 95 / 150 / 220 GHz), polarization only (E and B), maps in uK_CMB:

    d_freq = M_freq * B_freq * (f + a_freq * g) + n_freq

- **f**, the CMB, is the same in every channel. `f ~ N(0, C_f(r))` with `C_f(r) = scalar_covariance + r * tensor_covariance`. The scalar covariance is the LENSED scalar spectrum (lensing treated as Gaussian) and the tensor covariance is CAMB's tensor spectrum at r = 1. Both come from one CAMB run in `cmb_covariances`.
- **g**, the dust, is defined at 353 GHz. `g ~ N(0, C_g)` with a separate power law for EE and BB, in the BK18 convention `D_ell = ell (ell + 1) C_ell / (2 pi) = amplitude * (ell / 80)^knee`. Parameters: `dust_amplitude_ee`, `dust_knee_ee`, `dust_amplitude_bb`, `dust_knee_bb`.
- **a_freq** (`dust_frequency_scaling`) is a modified blackbody relative to 353 GHz, `(freq / 353)^beta_d * B_freq(T_d) / B_353(T_d)`, times the brightness to uK_CMB conversion `[dB/dT at 353] / [dB/dT at freq]` at T_CMB. `beta_d` is called `frequency_exponent` in the code.
- **PySM dust** (`dust_model = "pysm"`, needs the `pysm` extra) replaces the Gaussian draw of g with real-sky, non-Gaussian dust; everything else, including the `a_freq` scaling, is unchanged. `pysm_dust_field` builds it in five steps: (1) the PySM model (default `d9`: Planck GNILC templates, one spectral index and temperature) at 353 GHz in uK_CMB; (2) the Galactic plane masked (zero below |b| = 15 deg, cosine ramp to one at 30 deg); (3) E and B alms from `healpy.map2alm`, keeping only the multipoles between the patch's fundamental and its Nyquist, turned into two scalar maps; (4) the maps read at the patch's pixels, the patch being a longitude / latitude grid centred on the BICEP/Keck field (RA 0h, Dec -57.5 deg); (5) the patch mean removed and a Tukey taper (20 % of the width) applied so the field is periodic. The dust is the same on every call; `master_seed` then changes only the CMB and the noise, which are identical to the Gaussian run at the same seed.
- **n_freq** is white noise, the same level for E and B in every channel and independent between channels.
- **M, B**, the mask and beam, default to identity `DiagonalEB` operators. `simulate_data(..., mask = ..., beam = ...)` and both samplers take others, built in `instrument.py`:
  - a beam is a `DiagonalEB`: `gaussian_beam(nside, theta_pix, fwhm_arcmin)`, with `BICEP3_BEAM_FWHM_ARCMIN` (0.379 deg = 22.74') as the default width;
  - a mask is a `DiagonalEB` (Fourier-diagonal) or a `PixelMask(pixel_map)`, which multiplies the Q and U maps pixel by pixel and so mixes E and B; `apodized_border_mask(nside, taper_fraction = 0.2)` is a `PixelMask` of the Tukey window `edge_taper`.
  - The model applies the beam, then the mask, to `f + a_freq * g` (`data_model`: `mask * (beam * sky)`); the noise is not masked, so `noise_covariance` stays Fourier-diagonal and masked pixels of the data are pure noise.
  - `pysm_taper_fraction = None` (the default) tapers the PySM dust by `PYSM_TAPER_FRACTION = 0.2` under a `DiagonalEB` mask (including the identity) and not at all under any other mask (a `PixelMask`), so the dust is not tapered a second time; its mean subtraction and the Galactic-plane mask on the sphere stay. An explicit value overrides this. A pixel mask that does not go to zero at the edges leaves the dust discontinuous at the periodic boundary, so pass a taper fraction by hand then.

`simulate_data(nside, theta_pix, master_seed, r, frequency_exponent, dust_amplitude_ee = None, dust_knee_ee = None, dust_amplitude_bb = None, dust_knee_bb = None, ..., dust_model = "gaussian", pysm_preset = "d9", mask = None, beam = None, pysm_taper_fraction = None)` returns a dict with the fields `data`, `cmb_field`, `dust_field`, the operators `noise_covariance`, `scalar_covariance`, `tensor_covariance`, `field_covariance`, `dust_covariance` (`None` for PySM dust), `mask`, `beam`, `frequencies` / `frequency_scaling`, and `dust_model`. The four dust amplitude / knee arguments are needed for Gaussian dust only.

`gaussian_posterior(data_field, cmb_field, dust_field, mask, beam, dust_amplitude_ee, dust_knee_ee, dust_amplitude_bb, dust_knee_bb, frequency_exponent, r, noise_covariance, scalar_covariance, tensor_covariance, ...)` is the joint log-density of the data and the two fields, `-1/2 * [chi^2 terms + log-determinants]`, PLUS the Gaussian prior on `frequency_exponent` (BK18's beta_d = 1.49 +/- 0.13). There is no way to switch the prior off or change it through this function's arguments.

## The sampler

`sampler.sample_gaussian_posterior(data, noise_covariance, scalar_covariance, tensor_covariance, mask, beam, param_init, ...)` Gibbs-samples P(f, g, parameters | d). The parameters are named as everywhere else and updated in `PARAM_ORDER`: `dust_amplitude_ee`, `dust_amplitude_bb`, `dust_knee_ee`, `dust_knee_bb`, `r`, `frequency_exponent`. The dust temperature is fixed. It returns `(sample_histories, acceptance_rates, logposterior_values)`.

- **Field draws** (`gibbs_sample_field`, used for f and then g): Hoffman-Ribak. Simulate a field and noise from their priors, then add `wiener_filter` of (residual data - simulated data). The model has no lensing, so with a Fourier-diagonal mask and beam (`DiagonalEB`, including the default identity) the Wiener filter is solved exactly per mode. With a `PixelMask`, `wiener_filter` hands over to `conjugate_gradient_wiener_filter`: `jax.scipy.sparse.linalg.cg` on the real-space maps of `(1 + C^1/2 R^T N^-1 R C^1/2) y = C^1/2 R^T N^-1 d`, with `R^T` from `jax.linear_transpose` and the exact diagonal solution (mask replaced by its rms, `diagonal_response`) as preconditioner; `CONJUGATE_GRADIENT_TOLERANCE = 1e-6`, at most 500 iterations. The same rms stand-in is used for the HMC mass matrix and the starting dust field, where an approximation does not change the target distribution. Fields are pushed to data space with `apply_response(field, mask, beam, frequency_weights)`.
- **Parameter updates** (`metropolis_sample_parameter`): 1D random-walk Metropolis on `gaussian_posterior` at the current fields, with flat prior boxes `PARAM_BOUNDS` and widths `DEFAULT_PROPOSAL_WIDTHS` (hand-tuned on the default configuration, see below).
- **CLI:** `python -m cmb_scattering.sampler --nside 64 --num_iterations 2000 --r_proposal_width 0.01`. Every parameter has `--<name>` (truth of the simulated data, default `DEFAULT_TRUE_PARAMS`), `--<name>_init` and `--<name>_proposal_width`; the defaults are nside 128, 20' pixels and 5000 iterations. Output goes to `--out_dir` (default `gaussian_sampler_output/`): `sample_histories.npz`, the log-posterior plot `logpdf_over_time.png` and one trace plot per sampled parameter (`<name>_trace.png`). Every `--log_every` iterations (default 1, i.e. every iteration) the acceptance rates are printed and the plots rewritten; the plots are written to a temporary file and renamed into place so an image viewer never loads a partial file. Pass a larger `--log_every` for long chains, since each refresh redraws seven figures.
- **Mixing, measured 2026-11-17** (nside 128, 20', 3 uK-arcmin, default truth; about 36 iterations per second on the user's laptop CPU). r mixes fast (IAT about 6 iterations). The dust parameters and `frequency_exponent` mix slowly: a 6000-iteration chain gave IAT about 60 for the EE dust amplitude and knee, 130 to 220 for the BB ones and about 250 for `frequency_exponent`. A 500-iteration chain therefore underestimates their posterior widths (by a factor of four for `frequency_exponent`) and gives z-scores of 2 to 4 that a longer chain removes. Each of these parameters is updated with the dust field fixed, and the field with the parameters fixed; a parametrization that moves them together (as CMBLensing.py's mixing does) is the likely fix.
- **Proposal widths** near the 44 % target on that configuration: `dust_amplitude_ee` 0.2, `dust_amplitude_bb` 0.1, both knees 0.02, `r` 0.008, `frequency_exponent` 0.004. These are now `DEFAULT_PROPOSAL_WIDTHS`, which the notebook uses.
- **Non-Gaussian dust, first result (2026-11-17, one map, one 2000-iteration chain per data set, 3 uK-arcmin, nside 128 at 20').** With PySM d9 dust in the data and the Gaussian sampler unchanged, r = 0.0324 +/- 0.0045; with Gaussian dust of matched power laws and the same CMB and noise, r = 0.0323 +/- 0.0043 (truth 0.03). The shift is 0.03 sigma, so no effect on r is visible at this level. The dust is scaled between channels with the sampler's own `a_freq`, so this is the easy case: no frequency-scaling mismatch. A bias below the posterior width needs many seeds, which has not been done.
- **The Gaussian model fits PySM dust badly, and steeply.** The sampler's dust knees settle near -1.0 on PySM dust, against -0.70 (EE) and -0.63 (BB) from a binned power-law fit, and the default knee bounds `(-1, 0)` are hit; the notebook widens them to `(-2, 0)` through `param_bounds`. Part of this is the projection: the PySM dust has no power above the grid's axis Nyquist (ell 540) while the grid's corner modes reach 764, and the bilinear lookup smooths small scales. Proposal widths that accept near 44 % at this dust level: amplitudes 0.8 (EE) and 0.4 (BB), knees 0.02, r 0.008, frequency exponent 0.002.
- **Not validated:** no many-map test that z-scores scatter with unit width has been run. One long chain on one map had `dust_amplitude_ee` 2.6 sigma low at about 90 effective samples; unexplained, possibly chance.

## The scattering posterior and its sampler (exploratory, 2026-11-18)

`likelihoods.scattering_posterior(data_field, cmb_field, dust_field, mask, beam, frequency_exponent, r, noise_covariance, scalar_covariance, tensor_covariance, target_log_scattering_coefficients, scattering_width = 0.1, ...)` is `gaussian_posterior` with the Gaussian dust prior (the `g^2 / C_g` term and its log-determinant) replaced by an L2 loss, `-sum (log S(g) - log S_target)^2 / (2 scattering_width^2)`. It has no dust amplitudes or knees; its parameters are `r` and `frequency_exponent`. With the keyword `gaussian_dust_prior = True` the Gaussian dust prior is kept as well (both terms are added), and `dust_amplitude_ee`, `dust_knee_ee`, `dust_amplitude_bb`, `dust_knee_bb` are keyword arguments and parameters again.

- **Coefficients** (`log_scattering_coefficients(dust_field)`): Kymatio's `Scattering2D` (JAX frontend, Morlet wavelets, `SCATTERING_NUM_SCALES = 5`, `SCATTERING_NUM_ORIENTATIONS = 4`, orders 1 and 2) of the real-space E and B maps separately, averaged over the patch, then the log. Shape `(2, 180)`. The zeroth-order coefficient is dropped. Kymatio pads by reflection, so the transform does not use the maps' periodicity. The mask and beam are not applied: the loss acts on the sky dust on the whole patch.
- **Sampler** (`sampler.sample_scattering_posterior(data, noise_covariance, scalar_covariance, tensor_covariance, mask, beam, param_init, target_log_scattering_coefficients, scattering_width, proposal_widths, ..., hmc_num_steps = 10, hmc_step_size = 0.15, ...)`): each iteration draws f exactly (`gibbs_sample_field`), updates g with one HMC step (`hmc_sample_dust`: position = the two real-space dust maps, gradient by `jax.grad`, mass matrix = the data's precision on the dust, `inverse_noise_power`), then 1D Metropolis on the parameters in `proposal_widths`. Returns the same triple as the Gaussian sampler, for `r` and `frequency_exponent`, with the HMC acceptance rate under `"dust_field"`. About 0.47 s per iteration at nside 128 on the user's laptop CPU; HMC accepts about 59 % at the default step.
- **Start.** g starts from a draw about `least_squares_dust_field` (the joint least-squares fit of f and g to the data, with its covariance), not from zero (log S of a zero map is -inf) and not from the fit itself (see the pitfalls).
- **Sampling `frequency_exponent` makes the chain run away.** Measured on PySM d9 data (truth beta_d = 1.49, r = 0.03): over 2000 iterations beta_d climbed to 2.4 and r to 0.19 at `scattering_width = 0.1`, and to 2.8 and 0.25 at 0.5, with the log-posterior falling throughout. Cause (reasoned, consistent with the runs, not otherwise proven): the loss constrains 360 numbers of a dust field with 32,768 pixels and is not a normalized density, so it has no counterpart of the Gaussian prior's log-determinant. The data fix `a_freq * g`; lowering `a_freq` (raising beta_d) widens the dust field's conditional by a factor per pixel, and that volume gain outweighs the loss. Either hold `frequency_exponent` fixed by leaving it out of `proposal_widths`, or pass `gaussian_dust_prior = True`.
- **`gaussian_dust_prior = True` (2026-11-18, `docs/05_scattering_with_gaussian_prior.ipynb`: same map, four 300-iteration chains with beta_d sampled, burn-in 50).** `sample_scattering_posterior` then samples all six parameters of `PARAM_ORDER` (`param_init` needs all six; `proposal_widths = None` takes `DEFAULT_PROPOSAL_WIDTHS`) and adds the prior's precision to the HMC mass matrix; HMC accepts 60 % and an iteration takes about 0.55 s. Results: the scattering-only chain ran away (beta_d 1.49 to 1.72, r to about 0.05); with the Gaussian prior kept beta_d stayed within 0.005 of the truth (last 25 iterations: 1.493), as it did in the Gaussian sampler (1.500) and in a control with the prior but the loss switched off (1.480). r: 0.0317 +/- 0.0050 with the loss, 0.0322 +/- 0.0048 for the control, 0.0329 +/- 0.0049 for the Gaussian sampler, from 42 to 63 thinned samples each; the loss and the control chain share a seed and their r samples correlate at 0.93. So the flag stops the runaway and the scattering loss still has almost no effect on r on this data. NOT shown: a beta_d posterior (beta_d decorrelates over about 250 iterations, so 300 give about one sample) or converged dust parameters (the log-posteriors were still rising at iteration 300).
- **First result (2026-11-18, `docs/04_scattering_sampler_pysm_dust.ipynb`: one PySM d9 map at seed 0, 3 uK-arcmin, 1500 iterations per chain, burn-in 200, truth r = 0.03).** With beta_d fixed at the truth: scattering sampler r = 0.0316 +/- 0.0042, Gaussian sampler 0.0326 +/- 0.0042, and a control with the scattering term switched off (`scattering_width = 1e6`, i.e. no dust prior) 0.0321 +/- 0.0047. The Gaussian sampler with beta_d sampled gave 0.0331 +/- 0.0043. All agree within 0.35 sigma. The scattering and the control chain, which share a seed, have r samples correlated at 0.92: **the scattering term has almost no effect on r here**, because the data alone measure the dust well at this noise level. The comparison therefore does not show whether the scattering sampler is less biased; that needs a configuration where the dust prior matters, a target that is not the truth, and many maps.
- **The coefficients do see the non-Gaussianity.** Against a field with the PySM dust's exact power spectrum and random phases, the first-order log coefficients differ by +0.24 on average and the second-order ones by -0.13 with an rms of 0.42.
- **The runaway in the notebook:** 700 iterations with beta_d sampled took beta_d to 2.0 and r to 0.11.
- **The beta_d prior is in `scattering_posterior` but cannot stop the runaway.** Hand estimate (2026-11-18, order of magnitude, not measured): the volume gain is about 32,768 x |ln(220 / 353)| = 15,000 nats per unit of beta_d, against a prior pull of (beta_d - 1.49) / 0.13^2, about 8 nats per unit at 1 sigma. At beta_d = 2.4 the volume gain is about 14,000 nats, the scattering loss costs about 3,300 (360 coefficients, width 0.1) and the prior about 25.
- **Do not read structure into the r densities of that notebook.** The scattering chain's KDE shows two bumps (r near 0.0295 and 0.034, dip about 5 % of the peak). It is drawn from 163 thinned samples, where the density at the peak is uncertain by roughly 10 % (estimate), and the no-prior chain, which shares the seed, shows the same shoulder. A longer chain or other sampler seeds on the same map would settle it; neither has been run.
- **The Gaussian chains may not be converged at 1500 iterations:** by eye, the log-posterior of the chain with beta_d sampled is still rising slowly after iteration 800, in line with the slow dust-parameter mixing measured above.
## Mask and beam: first run (2026-11-18)

`docs/06_mask_and_beam.ipynb`: the PySM d9 map of notebooks 04 and 05 (seed 0, nside 128, 20', 3 uK-arcmin) with `apodized_border_mask(128)` (one over 64 % of the pixels, mean of its square 0.75) and `gaussian_beam` at the BICEP3 width (amplitude 0.975 at ell 80, 0.915 at ell 150, 0.32 at ell 540, 0.10 at the grid corners), dust untapered, one 500-iteration chain of the scattering posterior with `gaussian_dust_prior = True`, burn-in 200.

- **Checks made when implementing:** with the identity mask and beam, short chains of both samplers are identical bit for bit to those before the change. An all-ones `PixelMask` through the conjugate-gradient path matches the exact Wiener filter to a relative 2e-15. The pixel mask is self-adjoint under `dot` to rounding.
- **Conjugate gradient:** for the masked CMB Wiener filter, a relative residual of 1e-6 gives a relative error of 2.5e-4 in the solution and takes about 0.8 s; 1e-3 gives an error of 13 %, so the residual tolerance understates the error by a large factor. An iteration of the scattering sampler takes about 1.4 s under the mask (0.55 s without).
- **E to B leakage:** at 150 GHz the mask turns the E modes alone into a B map of 0.27 uK rms, against 0.14 uK for the masked CMB B modes. Nothing is done about it; the mask is in the data model.
- **Result:** r = 0.0336 +/- 0.0057 (z = +0.63, 17 thinned samples, IAT 18), against 0.0317 +/- 0.0050 in notebook 05 and 0.0316 +/- 0.0042 in notebook 04, both without mask and beam. The burn-in of 200 still includes the end of r's slow descent, which raises the mean, the width and the IAT. HMC accepts 61 % at the unchanged step size 0.15; the proposal widths of the earlier notebooks still accept 36 to 62 %.
- **Slow start:** r rose to 0.13 in the first 50 iterations, was near 0.04 by iteration 150 and scattered about 0.03 from about iteration 300. Suspected cause, not tested: the crude starting dust field (see the open points).
- **beta_d not settled:** it fell from 1.49 to about 1.450 by iteration 250 to 300 and was near 1.454 at iteration 500. Its KDE (5 thinned samples, z = -9.6) is not a posterior. Whether this is the slow start or a shift from the mask or beam is unknown; a longer chain is needed.
- **Widening of r from the mask** (naive expectation +15 % from 75 % of the patch): the width is 14 % above notebook 05's, but from 17 samples and with burn-in contamination, so the chains cannot tell.
- **An earlier run with the scattering loss windowed by the mask** (same seeds, since removed from the code) gave r = 0.0322 +/- 0.0049 (50 thinned samples) and the same slow start and beta_d fall (to 1.455). The two runs do not differ beyond what single 500-iteration chains allow.

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
- **Importing `cmb_lensing.chain_analysis` switches matplotlib to Agg.** It calls `matplotlib.use("Agg")` at import time, so figures stop appearing in a notebook. `docs/02_sampling_gaussian_dust.ipynb` imports `product_moments` / `product_pdf` from it and runs `%matplotlib inline` afterwards to switch back. The clean fix is in CMBLensing.py (move the call into its command-line entry point), which this project does not edit.
- **Which `cmb_lensing` is installed.** As of 2026-11-17 `~/venvs/myenv` has the local `~/CMBLensing.py` checkout installed in editable mode, not the GitHub copy the dependency declares.
- **Scalar E / B maps on the sphere are non-local.** Without the Galactic-plane mask, the plane (polarized intensity up to 3e4 uK at 353 GHz, against a median of 23 uK in the patch) dominated the patch's E map: the EE power at ell 20 to 40 was ten times higher than with the plane masked. Without removing the patch mean, the taper turns a constant offset into spurious large-scale power. Keep both steps when changing `pysm_dust_field`.
- **PySM dust has no dust-parameter truth and is not stationary.** The four dust amplitude / knee parameters are not inputs of a PySM simulation; the sampler still fits them, from `DEFAULT_TRUE_PARAMS` as a starting point, and only `r` and `frequency_exponent` have a true value. The dust is several times brighter toward the low-latitude side of the patch, and the taper fades it at the edges.
- **PySM geometry is approximate.** The longitude / latitude layout distorts a 43 degree patch by up to about 7 % at its edges, and its corners reach Galactic latitude -33.5 deg, just above the mask's ramp. A different patch centre or a larger patch needs the mask latitudes checked. healpy's B sign convention may differ from cmb_lensing's.
- **First PySM call downloads templates** (the nside 2048 GNILC dust map) into the astropy cache and took about a minute; later calls in the same process reuse a cached result and take about a second.
- **Averaging over rfft2 modes needs the Fourier weights.** A band power or any other average over the Fourier plane must weight each rfft2 entry by `get_fourier_weights` (2 in the bulk, 1 in the two self-conjugate columns). An unweighted mean double-counts the kx = 0 column, which for the PySM dust (brighter toward one edge) is not a harmless choice: it moved the fitted BB dust knee from -0.63 to -0.68.
- **`import kymatio.jax` fails** on current scipy: kymatio 0.3.0's top-level JAX module also imports its 3D transform, which needs `scipy.special.sph_harm` (removed). `likelihoods.py` imports `ScatteringJax2D` from `kymatio.scattering2d.frontend.jax_frontend` instead. kymatio 0.3.0 is the newest release (2022).
- **A mask must act on a field, not on another operator.** A `PixelMask` only defines `mask * field`. Write `mask * (beam * field)` (or `apply_response`), never `mask * beam * field`, which Python evaluates as `(mask * beam) * field`.
- **HMC must not start at the likelihood peak.** Started exactly at the least-squares dust, every HMC proposal was rejected: each mode starts with less potential energy than a typical sample, the trajectory always gains potential energy, and the leapfrog energy error has one sign (about -31 at step 0.15). The sampler therefore starts from a draw about the fit.
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

Measured on the PySM d9 dust of that patch (353 GHz, after masking, mean removal and taper, 2026-11-17): `D_ell` in uK^2 averaged over ell 8-20, 20-40, 60-100, 150-250, 350-500 is 131, 67, 26, 15.5, 8.7 for EE and 63, 30, 14, 8.9, 4.6 for BB. That is about three times BK18's BB amplitude of 4.4 uK^2 at ell = 80 (the patch is larger than the BICEP/Keck field and includes brighter sky), with EE / BB near 2. Map rms: 17 uK (E), 12 uK (B).

## Sources of the physical constants

Each constant has a comment with its source next to it in the code.

- BK18: BICEP/Keck XIII, <https://arxiv.org/abs/2110.00483> (bands, 353 GHz and ell = 80 pivots, T_d = 19.6 K, beta_d = 1.49 +0.13 / -0.12, A_d = 4.4 uK^2, alpha_d = -0.66)
- Planck 2018 XI, polarized dust foregrounds, <https://arxiv.org/abs/1801.04945> (EE and BB dust slopes, BB/EE = 0.53)
- Fixsen 2009, <https://arxiv.org/abs/0911.1955> (T_CMB = 2.72548 K)
- BICEP/Keck XV, <https://arxiv.org/abs/2110.00482> (BICEP3 beam: median Gaussian width 0.161 deg, FWHM 0.379 deg)
- PySM 3 dust models, <https://pysm3.readthedocs.io/en/latest/models/> (d9), and BK18 for the patch centre
