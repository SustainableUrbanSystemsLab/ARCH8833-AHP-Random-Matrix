# AHP-Random-Matrix

## Environment setup (uv)

This project uses [uv](https://github.com/astral-sh/uv) for dependency and environment management.

### 1. Sync dependencies

```bash
uv sync --dev
```

### 2. Run Jupyter

```bash
uv run jupyter lab
```

### 3. (Optional) Register a project kernel for notebooks

```bash
uv run python -m ipykernel install --user --name ahp-random-matrix --display-name "Python (ahp-random-matrix)"
```
