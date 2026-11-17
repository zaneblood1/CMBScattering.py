# Contributing to CMBScattering.py

Thanks for your interest in CMBScattering.py. Bug reports, questions, documentation fixes and code contributions are all welcome. This page explains how to get help, how to report a problem, and how to propose a change.

CMBScattering.py infers the tensor-to-scalar ratio from flat-sky, multi-frequency CMB polarization maps with a dust foreground. It builds on [CMBLensing.py](https://github.com/zaneblood1/CMBLensing.py). `PROJECT_SUMMARY.md` describes the data model, the code layout and the conventions. Two executed notebooks are worked examples: `docs/data_generation.ipynb` simulates and plots a data set, and `docs/sampler_gaussian_dust.ipynb` runs the Gaussian dust sampler on one.

## Getting help

All support goes through [GitHub Issues](https://github.com/zaneblood1/CMBScattering.py/issues). If you have a question about using the package, open an issue and say so in the title (e.g. "Question: ..."). Please search the existing issues first in case it has already been answered.

## Reporting a bug

Open a [GitHub issue](https://github.com/zaneblood1/CMBScattering.py/issues) that includes:

- what you ran: ideally a short, self-contained script that reproduces the problem
- what you expected to happen and what happened instead, with the full error traceback if there is one
- the run configuration: `nside`, `theta_pix`, the frequencies, the noise level, the seed and the model parameters (r, the dust amplitudes and indices, the frequency exponent)
- your environment: operating system, Python version, the versions of `jax`, `jaxlib`, `camb` and `cmb-lensing` (and whether `cmb-lensing` is the GitHub version or a local checkout), and whether you ran on CPU or GPU

For numerical problems (a result that looks wrong rather than a crash), a plot or the numbers you compared against are very helpful.

## Suggesting a feature

Open an issue describing what you would like to do and why. For larger changes, please open the issue before writing code so we can agree on the approach first.

## Setting up a development environment

Python 3.10 or newer is required.

```bash
git clone https://github.com/zaneblood1/CMBScattering.py.git
cd CMBScattering.py
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

This installs CMBLensing.py from GitHub. To develop against a local checkout of it instead, install that in editable mode afterwards (and again after any later `pip install -e ".[dev]"`, which puts the GitHub copy back):

```bash
pip install -e ../CMBLensing.py
```

## Running the tests

```bash
pytest
```

The test suite currently only checks that the package and its CMBLensing.py dependency import. The simulation, the log-posterior and the sampler are not covered yet, and contributions there are especially welcome. The sampler in particular has only been run on short single-map chains; a test that its z-scores scatter with unit width over many maps does not exist.

## Submitting a pull request

1. Fork the repository and create a branch off `main`.
2. Keep each pull request focused on one change.
3. If your change affects numerical results, say how you checked it in the pull request description.
4. If you change code that a notebook in `docs/` uses, re-execute it so its outputs match, and check that its commentary still describes them:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace docs/data_generation.ipynb
   jupyter nbconvert --to notebook --execute --inplace docs/sampler_gaussian_dust.ipynb
   ```

5. Add an entry to `CHANGELOG.md`, and update `PROJECT_SUMMARY.md` if the structure, a function signature or a convention changed. If you use Claude Code, `/update-docs` does this.
6. Open the pull request against `main` and describe what it changes and why.

## Code style

There is no automatic formatter, so please follow these conventions:

- Put spaces around equal signs and after commas (`a = method(b, c, d)`, not `a=method(b,c,d)`), and start comments without a space (`#like this`).
- We prefer plain, readable code, and variable names that follow the physics (`dust_amplitude_bb`, `frequency_exponent`, `noise_covariance`, ...). Try to strike a balance between readability and efficiency. Avoid the use of obscure APIs or algorithms which might obfuscate the logic to a novice.
- Do not try to reinvent the wheel. If numpy, scipy, jax or `cmb_lensing` already has the functionality, use it in your code rather than defining all methods from scratch.
- Absolutely no magic numbers. Every constant gets a name at the top of its module.
- Any constant taken from a paper gets a comment next to it with a link to that paper, so the value can be checked.
- Keep it lean: avoid adding defensive input checks or extra machinery that the computation does not need.

## License

CMBScattering.py is released under the [MIT License](LICENSE). By contributing, you agree that your contributions will be licensed under the same terms.
