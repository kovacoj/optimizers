import torch

from optimizers import ExtendedKalmanFilter
from optimizers import KalmanFilter
from optimizers import LevenbergMarquardt
from optimizers import Newton


TARGET = torch.tensor([0.25, -0.75])
A = torch.tensor([[2.0, 0.5], [-1.0, 1.5]])
B = torch.tensor([1.0, -0.5])


def _vector_param(values=(1.5, -1.0)):
    return torch.nn.Parameter(torch.tensor(values))


def _quadratic_loss(x):
    return ((x - TARGET) ** 2).sum()


def _residuals(x):
    return A @ x - B


def _residual_loss(x):
    errors = _residuals(x)
    return errors @ errors


def test_newton_reduces_quadratic_loss():
    x = _vector_param()
    optimizer = Newton([x])

    with torch.no_grad():
        before = _quadratic_loss(x).item()

    optimizer.step(lambda: _quadratic_loss(x))

    with torch.no_grad():
        after = _quadratic_loss(x).item()

    assert after < before


def test_levenberg_marquardt_reduces_residual_loss():
    x = _vector_param()
    optimizer = LevenbergMarquardt([x])

    with torch.no_grad():
        before = _residual_loss(x).item()

    for _ in range(5):
        optimizer.step(lambda: _residuals(x))

    with torch.no_grad():
        after = _residual_loss(x).item()

    assert after < before


def test_extended_kalman_filter_reduces_residual_loss():
    x = _vector_param()
    optimizer = ExtendedKalmanFilter([x])

    with torch.no_grad():
        before = _residual_loss(x).item()

    optimizer.step(lambda: _residuals(x))

    with torch.no_grad():
        after = _residual_loss(x).item()

    assert after < before


def test_kalman_filter_reduces_residual_loss():
    x = _vector_param()
    optimizer = KalmanFilter([x])

    def closure():
        return _residuals(x), A

    with torch.no_grad():
        before = _residual_loss(x).item()

    optimizer.step(closure)

    with torch.no_grad():
        after = _residual_loss(x).item()

    assert after < before
