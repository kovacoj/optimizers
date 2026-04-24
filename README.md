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
from optimizers import LevenbergMarquardt
from optimizers import Newton
from optimizers import line_search
```

`Newton`, `Annealing`, `Metropolis`, and `Genetic` expect `closure()` to return a scalar loss tensor. `LevenbergMarquardt` and `ExtendedKalmanFilter` expect a residual-vector closure. `KalmanFilter` expects `closure()` to return `(errors, H)`.

## Line search

The public `optimizers.line_search` submodule exposes pure-PyTorch callback-based helpers:

- `line_search.armijo_backtracking(phi, phi0, dphi0, ...)`
- `line_search.strong_wolfe(phi, dphi, phi0, dphi0, ...)`

`Newton(..., line_search_method="armijo" | "wolfe")` uses these helpers to scale the full Newton direction. `LevenbergMarquardt(..., strategy="line search", line_search_method="armijo" | "wolfe")` uses the same line-search methods for residual-vector problems, while `strategy="trust region"` switches to a trust-region LM update.

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

`Newton` defaults to the full Newton step when `line_search_method=None`. `LevenbergMarquardt` defaults to `strategy="line search"` and also accepts the compatibility aliases `"line_search"`, `"trust_region"`, and `"heuristic"`.
