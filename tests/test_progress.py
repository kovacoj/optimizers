import torch

from optimizers import Annealing
from optimizers import ExtendedKalmanFilter
from optimizers import Genetic
from optimizers import KalmanFilter
from optimizers import LevenbergMarquardt
from optimizers import Metropolis
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


def test_newton_armijo_line_search_reduces_quadratic_loss():
    x = _vector_param()
    optimizer = Newton([x], line_search_method="armijo")

    with torch.no_grad():
        before = _quadratic_loss(x).item()

    optimizer.step(lambda: _quadratic_loss(x))

    with torch.no_grad():
        after = _quadratic_loss(x).item()

    assert after < before


def test_newton_wolfe_line_search_reduces_quadratic_loss():
    x = _vector_param()
    optimizer = Newton([x], line_search_method="wolfe")

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


def test_levenberg_marquardt_trust_region_reduces_residual_loss():
    x = _vector_param()
    optimizer = LevenbergMarquardt([x], strategy="trust region")

    with torch.no_grad():
        before = _residual_loss(x).item()

    for _ in range(5):
        optimizer.step(lambda: _residuals(x))

    with torch.no_grad():
        after = _residual_loss(x).item()

    assert after < before


def test_levenberg_marquardt_armijo_line_search_reduces_residual_loss():
    x = _vector_param()
    optimizer = LevenbergMarquardt([x], strategy="line search", line_search_method="armijo")

    with torch.no_grad():
        before = _residual_loss(x).item()

    for _ in range(5):
        optimizer.step(lambda: _residuals(x))

    with torch.no_grad():
        after = _residual_loss(x).item()

    assert after < before


def test_levenberg_marquardt_wolfe_line_search_reduces_residual_loss():
    x = _vector_param()
    optimizer = LevenbergMarquardt([x], strategy="line search", line_search_method="wolfe")

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


def test_annealing_reduces_quadratic_loss_over_many_steps():
    torch.manual_seed(42)
    x = _vector_param()
    optimizer = Annealing([x])

    with torch.no_grad():
        before = _quadratic_loss(x).item()

    for _ in range(200):
        optimizer.step(lambda: _quadratic_loss(x))

    with torch.no_grad():
        after = _quadratic_loss(x).item()

    assert after < before


def test_metropolis_reduces_quadratic_loss_over_many_steps():
    torch.manual_seed(123)
    x = _vector_param()
    optimizer = Metropolis([x])

    with torch.no_grad():
        before = _quadratic_loss(x).item()

    for _ in range(200):
        optimizer.step(lambda: _quadratic_loss(x))

    with torch.no_grad():
        after = _quadratic_loss(x).item()

    assert after < before


def test_genetic_reduces_quadratic_loss_over_many_steps():
    torch.manual_seed(0)
    x = _vector_param()
    optimizer = Genetic([x])

    with torch.no_grad():
        before = _quadratic_loss(x).item()

    for _ in range(10):
        optimizer.step(lambda: _quadratic_loss(x))

    with torch.no_grad():
        after = _quadratic_loss(x).item()

    assert after < before
