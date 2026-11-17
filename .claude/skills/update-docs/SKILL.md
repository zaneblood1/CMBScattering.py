---
name: update-docs
description: Bring this repository's documentation (CHANGELOG.md, PROJECT_SUMMARY.md, CLAUDE.md, CONTRIBUTING.md, README.md) in line with the current code. Use before a commit, after any change to code, notebooks, commands or conventions, or when the user asks to update the docs.
argument-hint: "[optional: what changed, or a git range]"
---

# Update the documentation

Make the markdown documentation of this repository match the code as it is now. Edit the files directly; do not ask for confirmation first and do not commit.

$ARGUMENTS

## 1. Find out what changed

Run these and read the results before writing anything:

- `git status --short` and `git diff HEAD` for uncommitted work (include untracked files; read new files in full).
- `git log --oneline -15`, and `git diff <last documented commit>..HEAD` if commits landed since the newest `CHANGELOG.md` entry.
- The current signatures and constants, straight from the source: `grep -n "^def \|^[A-Z_]* = " cmb_scattering/*.py`. Function and parameter names in this repository are renamed often, so never write a name from memory.
- The executed notebooks in `docs/` if they changed: read their cells and stored outputs with `nbformat`, since the docs quote numbers from them.

## 2. Update each file

Only touch a file when something in it is wrong or missing. Keep the existing structure and tone.

- **`CHANGELOG.md`** (Keep a Changelog format, everything under `## [Unreleased]`, newest first). Add one entry per user-visible change, starting with the date it landed in bold (`- **YYYY-MM-DD** ...`), under `### Added`, `### Changed`, `### Fixed` or `### Removed`. Say what changed and where, naming the functions and files. Correct an existing entry of the same unreleased work when it has become false (a renamed function, a changed default) instead of stacking a contradiction on top. Do not log pure refactors nobody would notice, or edits to these docs themselves beyond a new file.
- **`PROJECT_SUMMARY.md`** (the briefing for AI agents and new contributors). Check every section against the code: the status table and its date, the layout tree, the data model, the function signatures quoted in prose, the sampler section (defaults, CLI flags, output file names, measured mixing numbers), the conventions, the pitfalls list, the reference numbers and the sources. Add a pitfall when the session uncovered a non-obvious trap. Record `#TODO` comments that describe planned work under the planned-work section.
- **`CLAUDE.md`** (working rules for Claude Code). Update the commands, the environment notes and the working rules when a new standing preference or workflow appeared. It imports `PROJECT_SUMMARY.md`, so do not duplicate that content here.
- **`CONTRIBUTING.md`** and **`README.md`**. Update install steps, commands, the description of the tests, and the list of notebooks and entry points when they change.

## 3. Rules

- Every number, name, default and file path written into the docs must be checked against the code or a stored notebook output in this session. If something could not be checked, leave it out or say it is unverified.
- State what has not been validated plainly (for example "never run", "one map only"). Do not describe code as working because it exists.
- Follow the repository's writing conventions: dates in absolute form, constants taken from papers linked to the paper, no change history inside `PROJECT_SUMMARY.md` beyond dated measurements.
- Do not edit source code, notebooks or `CMBLensing.py` as part of this command. If the docs cannot be made true without a code change (stale notebook commentary, a broken example), report it instead.

## 4. Report

Finish with a short summary: which files changed and the main points of each, anything found that is out of date but outside these files, and anything left unverified.
