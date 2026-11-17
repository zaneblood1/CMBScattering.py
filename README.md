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

- [`docs/data_generation.ipynb`](docs/data_generation.ipynb) simulates a data set and plots its spectra and maps.
- [`docs/sampler_gaussian_dust.ipynb`](docs/sampler_gaussian_dust.ipynb) runs the sampler on one simulated map and analyses the chain.

The sampler also runs from the terminal, simulating its own data. Trace plots and the sample histories go to `gaussian_sampler_output/`:

```bash
python -m cmb_scattering.sampler --num_iterations 2000 --log_every 100
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
