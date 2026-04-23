import torch
import pytest

from optimizers import Annealing
from optimizers import ExtendedKalmanFilter
from optimizers import Genetic
from optimizers import KalmanFilter
from optimizers import LevenbergMarquardt
from optimizers import Metropolis
from optimizers import Newton


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


def test_newton_step_ignores_unused_trainable_parameter():
    x = _scalar_param()
    y = _scalar_param(2.0)
    optimizer = Newton([x, y])

    optimizer.step(lambda: (x ** 2).sum())

    assert x.item() != 1.0
    assert y.item() == pytest.approx(2.0)


def test_newton_step_ignores_frozen_parameter():
    x = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
    y = _scalar_param()
    optimizer = Newton([x, y])

    optimizer.step(lambda: (y ** 2).sum())

    assert x.item() == pytest.approx(2.0)
    assert y.item() != 1.0


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


@pytest.mark.parametrize("optimizer_cls", [Annealing, Metropolis])
def test_stochastic_step_acceptance_uses_explicit_index(optimizer_cls, monkeypatch):
    x = _scalar_param()
    optimizer = optimizer_cls([x])

    def closure():
        return (x ** 2).sum()

    monkeypatch.setattr(optimizer, "mutate", lambda: torch.tensor([2.0]))

    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.0]))
    optimizer.step(closure)
    assert x.item() == pytest.approx(2.0)

    optimizer.update_weights(torch.tensor([1.0]))
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([1.0]))
    optimizer.step(closure)
    assert x.item() == pytest.approx(1.0)


def test_metropolis_mutate_changes_single_parameter():
    x = torch.nn.Parameter(torch.tensor([1.0, -1.0, 0.5]))
    optimizer = Metropolis([x])

    torch.manual_seed(0)
    before = optimizer.params.clone()
    after = optimizer.mutate()

    assert torch.count_nonzero(after - before).item() == 1


def test_annealing_step_ignores_frozen_parameter():
    x = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
    y = _scalar_param()
    optimizer = Annealing([x, y])

    optimizer.step(lambda: (y ** 2).sum())

    assert x.item() == pytest.approx(2.0)


def test_metropolis_step_ignores_frozen_parameter(monkeypatch):
    x = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
    y = _scalar_param()
    optimizer = Metropolis([x, y])

    monkeypatch.setattr(optimizer, "mutate", lambda: torch.tensor([0.0]))
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.0]))

    optimizer.step(lambda: (y ** 2).sum())

    assert x.item() == pytest.approx(2.0)
    assert y.item() == pytest.approx(0.0)


def test_genetic_step_ignores_frozen_parameter():
    x = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
    y = _scalar_param()
    optimizer = Genetic([x, y])
    optimizer.pop_size = 4
    optimizer.population = torch.tensor([[0.0]]).repeat(optimizer.pop_size, 1)

    optimizer.step(lambda: (y ** 2).sum())

    assert x.item() == pytest.approx(2.0)
    assert y.item() == pytest.approx(0.0)


def test_genetic_step_runs_with_small_population():
    x = _vector_param()
    optimizer = Genetic([x])
    optimizer.pop_size = 4
    optimizer.population = optimizer.params.unsqueeze(0).repeat(optimizer.pop_size, 1)

    loss = optimizer.step(lambda: (x ** 2).sum())

    assert isinstance(loss, torch.Tensor)


def test_genetic_step_runs_with_scalar_parameter_vector():
    x = _scalar_param()
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


def test_levenberg_marquardt_step_runs_with_zero_line_search_steps():
    x = _scalar_param()
    optimizer = LevenbergMarquardt([x], m_max=0)

    loss = optimizer.step(lambda: x.view(-1))

    assert isinstance(loss, float)


def test_levenberg_marquardt_step_ignores_frozen_parameter():
    x = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
    y = _scalar_param()
    optimizer = LevenbergMarquardt([x, y])

    loss = optimizer.step(lambda: y.view(-1))

    assert isinstance(loss, float)
    assert x.item() == pytest.approx(2.0)


def test_levenberg_marquardt_restores_weights_when_line_search_fails():
    x = torch.nn.Parameter(torch.tensor([-2.9], dtype=torch.float64))
    optimizer = LevenbergMarquardt([x], mu=1e-6, m_max=1)

    def closure():
        return torch.sin(5 * x).view(-1)

    before = (closure() @ closure()).item()
    optimizer.step(closure)
    after = (closure() @ closure()).item()

    assert after == pytest.approx(before)
    assert x.item() == pytest.approx(-2.9)


def test_levenberg_marquardt_state_dict_restores_mu():
    x = _scalar_param()
    optimizer = LevenbergMarquardt([x], mu=1.0, mu_factor=10, m_max=1)

    optimizer.step(lambda: torch.tensor([10.0]) * x.view(-1))
    state_dict = optimizer.state_dict()

    y = _scalar_param()
    restored = LevenbergMarquardt([y], mu=99.0, mu_factor=2, m_max=3)
    restored.load_state_dict(state_dict)

    assert state_dict['param_groups'][0]['mu'] == pytest.approx(optimizer.mu)
    assert restored.mu == pytest.approx(optimizer.mu)


def test_extended_kalman_filter_step_runs():
    x = _scalar_param()
    optimizer = ExtendedKalmanFilter([x])

    loss = optimizer.step(lambda: x.view(-1))

    assert isinstance(loss, torch.Tensor)


def test_extended_kalman_filter_step_ignores_frozen_parameter():
    x = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
    y = _scalar_param()
    optimizer = ExtendedKalmanFilter([x, y])

    loss = optimizer.step(lambda: y.view(-1))

    assert isinstance(loss, torch.Tensor)
    assert x.item() == pytest.approx(2.0)


def test_kalman_filter_step_runs():
    x = _scalar_param()
    optimizer = KalmanFilter([x])

    def closure():
        errors = x.view(-1)
        H = torch.eye(1, device=x.device, dtype=x.dtype)
        return errors, H

    loss = optimizer.step(closure)

    assert isinstance(loss, torch.Tensor)


def test_kalman_filter_step_ignores_frozen_parameter():
    x = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
    y = _scalar_param()
    optimizer = KalmanFilter([x, y])

    def closure():
        errors = y.view(-1)
        H = torch.eye(1, device=y.device, dtype=y.dtype)
        return errors, H

    loss = optimizer.step(closure)

    assert isinstance(loss, torch.Tensor)
    assert x.item() == pytest.approx(2.0)
