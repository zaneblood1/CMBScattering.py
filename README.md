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
