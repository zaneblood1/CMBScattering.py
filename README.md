# CMBScattering.py
Inference of the tensor-to-scalar ratio in the flat sky using scattering transform summary statistics.

## Installation

Requires Python 3.10 or newer.

```bash
pip install git+https://github.com/zaneblood1/CMBScattering.py.git
```

This also installs [CMBLensing.py](https://github.com/zaneblood1/CMBLensing.py) from GitHub, along with its dependencies (JAX, CAMB, ...).

```python
import cmb_scattering
import cmb_lensing
```

## Getting started

The package currently implements a Gaussian dust model: a simulator of three-frequency E + B maps, its log-posterior, and a Gibbs sampler for r and the dust parameters. The scattering transform statistics are not implemented yet.

- [`docs/01_simulating_data.ipynb`](docs/01_simulating_data.ipynb) simulates a data set and plots its spectra and maps, then shows how the non-Gaussian PySM dust is made (full-sky map, mask, patch, taper). Its last section needs the `pysm` extra.
- [`docs/02_sampling_gaussian_dust.ipynb`](docs/02_sampling_gaussian_dust.ipynb) runs the sampler on one simulated map and analyses the chain.
- [`docs/03_sampling_pysm_dust.ipynb`](docs/03_sampling_pysm_dust.ipynb) runs the same Gaussian sampler on data with non-Gaussian PySM dust and on matched Gaussian dust, and compares the recovered r.

The sampler also runs from the terminal, simulating its own data. Trace plots and the sample histories go to `gaussian_sampler_output/`:

```bash
python -m cmb_scattering.sampler --num_iterations 2000 --log_every 100
```

To test the Gaussian sampler on non-Gaussian dust, the simulated data can use a PySM dust map of the real sky instead (`simulate_data(..., dust_model = "pysm")`, or `--dust_model pysm` on the sampler). This needs the optional PySM dependencies and downloads PySM's dust template on first use:

```bash
pip install -e ".[pysm]"
```

`python -m cmb_scattering.sampler --help` lists every option. The sampler has only been run on short single-map chains and is not yet validated.

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
