# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Read the project summary first; it describes the data model, the layout and the pitfalls:

@PROJECT_SUMMARY.md

## Commands

```bash
#Install (editable), then switch cmb_lensing to the local checkout if developing against it
pip install -e ".[dev]"
pip install -e ../CMBLensing.py

#Tests (the user runs these; do not write or run tests unless asked)
pytest

#Re-execute the guide notebook in place after changing code it uses
jupyter nbconvert --to notebook --execute --inplace docs/data_generation.ipynb
```

The working environment on the user's machine is `~/venvs/myenv` (no venv is active in a fresh shell, and the system `python3` has neither package installed). `pip install -e ".[dev]"` replaces an editable `cmb_lensing` with the GitHub copy, so re-run the second command after it.

## Working in this repository

- **Tests:** do not write or run tests unless the user asks. Running a script or the notebook to check that new code executes is fine when the task is to produce working output.
- **Literature values:** look constants up with a web search (read the paper itself when a search summary does not give the number) instead of quoting them from memory. Put a link to the source in a comment next to every constant taken from a paper.
- **Notebooks:** edit `.ipynb` files through `nbformat` (or the notebook tools), not by hand-editing JSON, and re-execute so the stored outputs match the code. The user often has the notebook open; say when a file was changed on disk.
- **Docs to keep current:** add an entry to `CHANGELOG.md` for every user-visible change, and update `PROJECT_SUMMARY.md` when the structure, a signature or a convention changes.
- **File edits:** use the dedicated `Edit` / `Write` tools as the default way to change files, not `sed` or heredocs through `Bash`.
- **The user renames things.** Function names in `cmb_scattering/` change between sessions (e.g. `gaussian_likelihood` became `gaussian_posterior`). Read the current source before calling or importing anything.

## Numeric precision

JAX runs in float64: `jax.config.update("jax_enable_x64", True)` is repeated at the top of each module, as in `cmb_lensing`. Fourier arrays are complex128 and maps float64.

## Code style

Spaces are used between equal signs and after commas when calling methods or setting variables. For example
"a = b" is preferred style over "a=b" and "a = method(b, c, d)" is preferred style over "a=method(b,c,d)".

Comments should not start with a space. For example "#this is a preferred comment" is preferred style over "# this is NOT a preferred comment".

**Readability without giving up performance.** Aim for readability: intuitively-named quantities that follow the physics (`dust_amplitude_bb`, `frequency_exponent`, `noise_covariance`), one clear step per line, structure that mirrors the maths. Readability never justifies a slower implementation, though: vectorize, `jit` and `vmap` where it matters.

**Use existing APIs.** If numpy, scipy, jax, the Python standard library or `cmb_lensing` already does the job (an FFT, a Gaussian field draw, a covariance on the rfft2 grid, `pinv`, `log_det`, `dot`), use it rather than writing a hand-rolled version.

**No magic numbers.** Every numerical value with meaning (a frequency, a temperature, a pivot multipole, a prior width, a noise level, a seed) gets a named constant at the top of its module, with its source if it comes from a paper. Literal 0, 1, 2 in plain arithmetic (`x**2`, `n // 2 + 1`) are fine.

**Comments and docstrings** say what a block does and any non-obvious why, briefly. Skip restating the code, long derivations (point to the module docstring or a reference instead) and change history.

**Keep defensive machinery light.** No `raise ValueError` guard rails for misuse, no consistency checks between inputs, no checksums. Trust the caller; a wrong input should just give a wrong (or crashing) result.
