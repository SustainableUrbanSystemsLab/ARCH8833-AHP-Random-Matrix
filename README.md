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

## Generate RI convergence GIF

Create an animation that shows how the RI approximation changes as the number of simulated matrices grows from 5 to 5000:

```bash
uv run python scripts/generate_ri_convergence_gif.py --min-steps 5 --max-steps 5000 --output artifacts/ri_convergence_5_to_5000.gif
```
