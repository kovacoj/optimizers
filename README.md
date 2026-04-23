# optimizers

[![Tests](https://github.com/kovacoj/optimizers/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/kovacoj/optimizers/actions/workflows/tests.yml)

Non-standard optimizers for `PyTorch`.

## Install with uv

Add the package from GitHub:

```bash
uv add git+https://github.com/kovacoj/optimizers.git
```

Add the package as a git submodule if you want to pull upstream updates into your repo explicitly:

```bash
git submodule add https://github.com/kovacoj/optimizers.git optimizers
uv add ./optimizers
```

Sync the submodule to the latest upstream commit later with:

```bash
git submodule update --remote --merge optimizers
uv lock
```

For local development after cloning the repo:

```bash
uv venv
uv pip install -e .
```

Because this repository uses a `src` layout, importing directly from the repo checkout without installing requires `PYTHONPATH=src`.

## Use

```python
from optimizers import KalmanFilter
```

`KalmanFilter` expects `closure()` to return `(errors, H)`, while `ExtendedKalmanFilter` computes the Jacobian from a residual-vector closure.

## Public API

| Optimizer | Closure contract | `step()` return |
| --- | --- | --- |
| `Newton` | scalar loss tensor | `None` |
| `Annealing` | scalar loss tensor | scalar loss tensor |
| `Metropolis` | scalar loss tensor | scalar loss tensor |
| `Genetic` | scalar loss tensor | scalar loss tensor |
| `LevenbergMarquardt` | residual vector tensor | Python `float` |
| `ExtendedKalmanFilter` | residual vector tensor | scalar loss tensor |
| `KalmanFilter` | `(errors, H)` | scalar loss tensor |

`KalmanFilter` is the linear-residual variant. `ExtendedKalmanFilter` computes the residual Jacobian internally.
