# optimizers

[![Tests](https://github.com/kovacoj/optimizers/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/kovacoj/optimizers/actions/workflows/tests.yml)

A set of mostly quasi-Newton optimizers for PyTorch.

This project started as an academic experiment. The quasi-Newton methods can speed up convergence, but they do so at the cost of higher memory usage, so they are best suited to relatively small systems.

The genetic and sampling-based optimizers are included for completeness, but in most practical settings they are not a good choice. If you find yourself reaching for them because you want more variance or exploration in the weight updates, consider using `ExtendedKalmanFilter` instead: increasing its process noise parameter `q` (or decreasing the forgetting factor `tau`) inflates the state covariance `P`, which increases the Kalman gain and produces larger, more exploratory steps — a gradient-based alternative to stochastic search.

## Install with uv

This package currently targets Python 3.13+ and is documented for Git/source installs rather than PyPI publication.

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
