Repository of not very useful optimizers in `PyTorch`.

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

`KalmanFilter` is an alias for `ExtendedKalmanFilter`.
