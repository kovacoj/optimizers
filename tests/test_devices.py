import pytest
import torch

from optimizers import Annealing
from optimizers import ExtendedKalmanFilter
from optimizers import Genetic
from optimizers import KalmanFilter
from optimizers import LevenbergMarquardt
from optimizers import Metropolis
from optimizers import Newton


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


def _vector_param(device):
    return torch.nn.Parameter(torch.tensor([1.5, -1.0], device=device))


def _target(device):
    return torch.tensor([0.25, -0.75], device=device)


def _residual_problem(device):
    A = torch.tensor([[2.0, 0.5], [-1.0, 1.5]], device=device)
    b = torch.tensor([1.0, -0.5], device=device)
    return A, b


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("line_search_method", [None, "armijo", "wolfe"])
def test_newton_runs_on_available_device(device, line_search_method):
    x = _vector_param(device)
    optimizer = Newton([x], line_search_method=line_search_method)

    def closure():
        return ((x - _target(device)) ** 2).sum()

    before = closure().item()
    optimizer.step(closure)
    after = closure().item()

    assert x.device.type == torch.device(device).type
    assert after < before


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("strategy", ["line search", "trust region"])
def test_levenberg_marquardt_runs_on_available_device(device, strategy):
    x = _vector_param(device)
    A, b = _residual_problem(device)
    optimizer = LevenbergMarquardt([x], strategy=strategy)

    def closure():
        return A @ x - b

    before = (closure() @ closure()).item()
    optimizer.step(closure)
    after = (closure() @ closure()).item()

    assert x.device.type == torch.device(device).type
    assert after < before


@pytest.mark.parametrize("device", DEVICES)
def test_extended_kalman_filter_runs_on_available_device(device):
    x = _vector_param(device)
    A, b = _residual_problem(device)
    optimizer = ExtendedKalmanFilter([x])

    def closure():
        return A @ x - b

    loss = optimizer.step(closure)

    assert x.device.type == torch.device(device).type
    assert loss.device.type == torch.device(device).type


@pytest.mark.parametrize("device", DEVICES)
def test_kalman_filter_runs_on_available_device(device):
    x = _vector_param(device)
    A, b = _residual_problem(device)
    optimizer = KalmanFilter([x])

    def closure():
        return A @ x - b, A

    loss = optimizer.step(closure)

    assert x.device.type == torch.device(device).type
    assert loss.device.type == torch.device(device).type


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("optimizer_cls", [Annealing, Metropolis])
def test_stochastic_search_runs_on_available_device(device, optimizer_cls):
    torch.manual_seed(0)
    x = _vector_param(device)
    optimizer = optimizer_cls([x])

    loss = optimizer.step(lambda: ((x - _target(device)) ** 2).sum())

    assert x.device.type == torch.device(device).type
    assert loss.device.type == torch.device(device).type


@pytest.mark.parametrize("device", DEVICES)
def test_genetic_runs_on_available_device(device):
    torch.manual_seed(0)
    x = _vector_param(device)
    optimizer = Genetic([x])
    optimizer.pop_size = 4
    optimizer.population = optimizer.params.unsqueeze(0).repeat(optimizer.pop_size, 1)

    loss = optimizer.step(lambda: ((x - _target(device)) ** 2).sum())

    assert x.device.type == torch.device(device).type
    assert loss.device.type == torch.device(device).type
