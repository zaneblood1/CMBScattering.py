# CMBScattering.py
Inference of the tensor-to-scalar ratio in the flat sky using scattering transform summary statistics.

## Installation

Requires Python 3.10 or newer.

```bash
pip install git+https://github.com/zaneblood1/CMBScattering.py.git
```

This also installs [CMBLensing.py](https://github.com/zaneblood1/CMBLensing.py) from GitHub, along with its dependencies (JAX, CAMB, ...), and [Kymatio](https://www.kymat.io/) for the scattering transform.

```python
import cmb_scattering
import cmb_lensing
```

## Getting started

The package implements a simulator of three-frequency E + B maps with Gaussian or PySM dust, and two posteriors with a sampler each: a Gaussian dust model (a Gibbs sampler for r and the dust parameters), and a first, exploratory scattering transform model, in which the dust prior is a loss on the wavelet scattering coefficients of the dust map.

- [`docs/01_simulating_data.ipynb`](docs/01_simulating_data.ipynb) simulates a data set and plots its spectra and maps, then shows how the non-Gaussian PySM dust is made (full-sky map, mask, patch, taper). Its last section needs the `pysm` extra.
- [`docs/02_sampling_gaussian_dust.ipynb`](docs/02_sampling_gaussian_dust.ipynb) runs the sampler on one simulated map and analyses the chain.
- [`docs/03_sampling_pysm_dust.ipynb`](docs/03_sampling_pysm_dust.ipynb) runs the same Gaussian sampler on data with non-Gaussian PySM dust and on matched Gaussian dust, and compares the recovered r.
- [`docs/04_scattering_sampler_pysm_dust.ipynb`](docs/04_scattering_sampler_pysm_dust.ipynb) runs the scattering transform sampler and the Gaussian sampler on the same PySM-dust data and compares them.
- [`docs/05_scattering_with_gaussian_prior.ipynb`](docs/05_scattering_with_gaussian_prior.ipynb) tests the scattering sampler with the Gaussian dust prior kept next to the scattering loss (`gaussian_dust_prior = True`), with the dust frequency exponent sampled.

The sampler also runs from the terminal, simulating its own data. Trace plots and the sample histories go to `gaussian_sampler_output/`:

```bash
python -m cmb_scattering.sampler --num_iterations 2000 --log_every 100
```

To test the Gaussian sampler on non-Gaussian dust, the simulated data can use a PySM dust map of the real sky instead (`simulate_data(..., dust_model = "pysm")`, or `--dust_model pysm` on the sampler). This needs the optional PySM dependencies and downloads PySM's dust template on first use:

```bash
pip install -e ".[pysm]"
```

`python -m cmb_scattering.sampler --help` lists every option. The command line runs the Gaussian sampler only; the scattering sampler is called from Python (`cmb_scattering.sampler.sample_scattering_posterior`). Both samplers have only been run on short single-map chains and are not yet validated.

## Development

```bash
git clone https://github.com/zaneblood1/CMBScattering.py.git
cd CMBScattering.py
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

To develop against a local checkout of CMBLensing.py instead of the GitHub version, install it in editable mode afterwards:

```bash
pip install -e ../CMBLensing.py
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to contribute, [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) for the data model and code layout, and [`CHANGELOG.md`](CHANGELOG.md) for the history.
