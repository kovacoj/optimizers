import torch
import pytest

from optimizers import Annealing
from optimizers import ExtendedKalmanFilter
from optimizers import Genetic
from optimizers import KalmanFilter
from optimizers import LevenbergMarquardt
from optimizers import Newton
from optimizers.Metropolis import Metropolis


def _scalar_param(value=1.0):
    return torch.nn.Parameter(torch.tensor([value]))


def _vector_param(values=(1.0, -1.0)):
    return torch.nn.Parameter(torch.tensor(values))


def test_newton_step_runs():
    x = _scalar_param()
    optimizer = Newton([x])

    optimizer.step(lambda: (x ** 2).sum())


def test_newton_step_runs_with_float64():
    x = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = Newton([x])

    optimizer.step(lambda: (x ** 2).sum())


def test_newton_rejects_multiple_param_groups():
    x = _scalar_param()
    y = _scalar_param(2.0)
    optimizer = Newton([
        {"params": [x]},
        {"params": [y]},
    ])

    with pytest.raises(AssertionError):
        optimizer.step(lambda: (x ** 2).sum() + (y ** 2).sum())


def test_annealing_step_runs():
    x = _scalar_param()
    optimizer = Annealing([x])

    loss = optimizer.step(lambda: (x ** 2).sum())

    assert isinstance(loss, torch.Tensor)


def test_metropolis_step_runs():
    x = _scalar_param()
    optimizer = Metropolis([x])

    loss = optimizer.step(lambda: (x ** 2).sum())

    assert isinstance(loss, torch.Tensor)


def test_genetic_step_runs_with_small_population():
    x = _vector_param()
    optimizer = Genetic([x])
    optimizer.pop_size = 4
    optimizer.population = optimizer.params.unsqueeze(0).repeat(optimizer.pop_size, 1)

    loss = optimizer.step(lambda: (x ** 2).sum())

    assert isinstance(loss, torch.Tensor)


def test_levenberg_marquardt_step_runs():
    x = _scalar_param()
    optimizer = LevenbergMarquardt([x])

    loss = optimizer.step(lambda: x.view(-1))

    assert isinstance(loss, float)


def test_levenberg_marquardt_step_runs_with_float64():
    x = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = LevenbergMarquardt([x])

    loss = optimizer.step(lambda: x.view(-1))

    assert isinstance(loss, float)


def test_extended_kalman_filter_step_runs():
    x = _scalar_param()
    optimizer = ExtendedKalmanFilter([x])

    loss = optimizer.step(lambda: x.view(-1))

    assert isinstance(loss, torch.Tensor)


def test_kalman_filter_step_runs():
    x = _scalar_param()
    optimizer = KalmanFilter([x])

    def closure():
        errors = x.view(-1)
        H = torch.eye(1, device=x.device, dtype=x.dtype)
        return errors, H

    loss = optimizer.step(closure)

    assert isinstance(loss, torch.Tensor)
