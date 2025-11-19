# Continuous Integration (CI) – Options and Quick Start

This repository ships without an active CI pipeline. Use this guide to enable CI when you need it, or to run the same checks locally.

## What to run locally (Windows PowerShell)
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt

# Optional dev tools
pip install pytest mypy flake8

# Unit tests
pytest -q

# Type check (optional)
mypy --ignore-missing-imports src/agmlib

# Lint (optional)
flake8 --ignore=E501 .
```

## Enable GitHub Actions (optional)
Create `.github/workflows/ci.yml` with a minimal workflow:
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.10' }
      - run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest
      - run: pytest -q
```

Matrix + type/lint example:
```yaml
jobs:
  matrix:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.10']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: ${{ matrix.python-version }} }
      - run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest mypy flake8
      - run: flake8 --ignore=E501 .
      - run: mypy --ignore-missing-imports src/agmlib
      - run: pytest -q
```

## Other CI providers (sketches)
- GitLab CI: define `.gitlab-ci.yml` with a Python image and the same commands as above.
- Azure Pipelines: create `azure-pipelines.yml` with a Python job; run the same install/test steps.

## Self‑hosted runners
- For private or offline environments, configure a self‑hosted runner and point the workflow to it via `runs-on: self-hosted`.

## Secrets & logs
- Never commit secrets. `.gitignore` is set to exclude `logs/`, common caches, env files, and key material patterns.
- For CI secrets (if used), store them in the provider’s secret store (not in the repo).


