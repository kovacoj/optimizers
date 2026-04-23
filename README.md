Repository of not very useful optimizers in `PyTorch`.

## Install with uv

Add the package from GitHub:

```bash
uv add git+https://github.com/kovacoj/optimizers.git
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
