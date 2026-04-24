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


def test_newton_armijo_line_search_step_runs():
    x = _scalar_param()
    optimizer = Newton([x], line_search_method="armijo")

    optimizer.step(lambda: (x ** 2).sum())


def test_newton_wolfe_line_search_step_runs():
    x = _scalar_param()
    optimizer = Newton([x], line_search_method="wolfe")

    optimizer.step(lambda: (x ** 2).sum())


def test_newton_line_search_keeps_params_on_non_descent_direction(monkeypatch):
    x = _scalar_param()
    optimizer = Newton([x], line_search_method="armijo")

    def non_descent_direction(system, rhs):
        return -rhs

    monkeypatch.setattr(torch.linalg, "solve", non_descent_direction)

    optimizer.step(lambda: (x ** 2).sum())

    assert x.item() == pytest.approx(1.0)


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


def test_annealing_state_dict_restores_temperature():
    x = _scalar_param()
    optimizer = Annealing([x])
    optimizer.step(lambda: (x ** 2).sum())

    state_dict = optimizer.state_dict()

    y = _scalar_param()
    restored = Annealing([y])
    restored.load_state_dict(state_dict)

    assert state_dict['param_groups'][0]['temperature'] == pytest.approx(optimizer.temperature)
    assert restored.temperature == pytest.approx(optimizer.temperature)


def test_metropolis_step_ignores_frozen_parameter(monkeypatch):
    x = torch.nn.Parameter(torch.tensor([2.0]), requires_grad=False)
    y = _scalar_param()
    optimizer = Metropolis([x, y])

    monkeypatch.setattr(optimizer, "mutate", lambda: torch.tensor([0.0]))
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.0]))

    optimizer.step(lambda: (y ** 2).sum())

    assert x.item() == pytest.approx(2.0)
    assert y.item() == pytest.approx(0.0)


def test_metropolis_state_dict_restores_temperature():
    x = _scalar_param()
    optimizer = Metropolis([x])
    optimizer.step(lambda: (x ** 2).sum())

    state_dict = optimizer.state_dict()

    y = _scalar_param()
    restored = Metropolis([y])
    restored.load_state_dict(state_dict)

    assert state_dict['param_groups'][0]['temperature'] == pytest.approx(optimizer.temperature)
    assert restored.temperature == pytest.approx(optimizer.temperature)


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


def test_genetic_mutation_strength_scales_noise(monkeypatch):
    x = torch.nn.Parameter(torch.zeros(3))
    optimizer = Genetic([x])
    optimizer.mutation_rate = 1.0
    optimizer.mutation_strength = 0.1

    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.zeros_like(tensor))
    monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.ones_like(tensor))

    mutated = optimizer.mutate(torch.zeros_like(x))

    assert torch.equal(mutated, torch.full_like(x, 0.1))


def test_genetic_state_dict_restores_population_and_elite_state():
    x = _scalar_param()
    optimizer = Genetic([x])
    optimizer.pop_size = 4
    optimizer.population = optimizer.params.unsqueeze(0).repeat(optimizer.pop_size, 1)
    optimizer.step(lambda: (x ** 2).sum())

    state_dict = optimizer.state_dict()

    y = _scalar_param()
    restored = Genetic([y])
    restored.load_state_dict(state_dict)

    assert restored.pop_size == optimizer.pop_size
    assert torch.equal(restored.population, optimizer.population)
    assert torch.equal(restored.best_genome, optimizer.best_genome)
    assert restored.best_fitness == pytest.approx(optimizer.best_fitness)


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


def test_levenberg_marquardt_line_search_recovers_from_bad_full_step():
    x = torch.nn.Parameter(torch.tensor([-2.9], dtype=torch.float64))
    optimizer = LevenbergMarquardt([x], mu=1e-6, m_max=1)

    def closure():
        return torch.sin(5 * x).view(-1)

    before = (closure() @ closure()).item()
    optimizer.step(closure)
    after = (closure() @ closure()).item()

    assert after < before


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


def test_levenberg_marquardt_solve_epsilon_controls_diagonal_jitter():
    exact = _scalar_param()
    jittered = _scalar_param()

    LevenbergMarquardt([exact], mu=0.0, solve_epsilon=0.0).step(lambda: exact.view(-1))
    LevenbergMarquardt([jittered], mu=0.0, solve_epsilon=1.0).step(lambda: jittered.view(-1))

    assert exact.item() == pytest.approx(0.0)
    assert jittered.item() == pytest.approx(0.5)


def test_levenberg_marquardt_state_dict_restores_solve_epsilon():
    x = _scalar_param()
    optimizer = LevenbergMarquardt([x], solve_epsilon=0.25)
    state_dict = optimizer.state_dict()

    y = _scalar_param()
    restored = LevenbergMarquardt([y])
    restored.load_state_dict(state_dict)

    assert restored.solve_epsilon == pytest.approx(0.25)


def test_levenberg_marquardt_strategy_uses_line_search_name():
    optimizer = LevenbergMarquardt([_scalar_param()])

    assert optimizer.strategy == "line search"


def test_levenberg_marquardt_accepts_heuristic_alias():
    optimizer = LevenbergMarquardt([_scalar_param()], strategy="heuristic")

    assert optimizer.strategy == "line search"


def test_levenberg_marquardt_trust_region_step_runs():
    x = _scalar_param()
    optimizer = LevenbergMarquardt([x], strategy="trust region")

    loss = optimizer.step(lambda: x.view(-1))

    assert isinstance(loss, float)


def test_levenberg_marquardt_armijo_line_search_step_runs():
    x = _scalar_param()
    optimizer = LevenbergMarquardt([x], strategy="line search", line_search_method="armijo")

    loss = optimizer.step(lambda: x.view(-1))

    assert isinstance(loss, float)


def test_levenberg_marquardt_wolfe_line_search_step_runs():
    x = _scalar_param()
    optimizer = LevenbergMarquardt([x], strategy="line search", line_search_method="wolfe")

    loss = optimizer.step(lambda: x.view(-1))

    assert isinstance(loss, float)


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


def test_extended_kalman_filter_state_dict_restores_covariance():
    x = _scalar_param()
    optimizer = ExtendedKalmanFilter([x])
    optimizer.step(lambda: x.view(-1))

    state_dict = optimizer.state_dict()

    y = _scalar_param()
    restored = ExtendedKalmanFilter([y])
    restored.load_state_dict(state_dict)

    assert torch.equal(optimizer.P, restored.P)


def test_extended_kalman_filter_q_update_changes_Q():
    x = _scalar_param()
    optimizer = ExtendedKalmanFilter([x], q=1.0)

    optimizer.q = 5.0

    assert optimizer.Q[0, 0].item() == pytest.approx(5.0)


def _kalman_covariance_problem():
    dtype = torch.float64
    x = torch.nn.Parameter(torch.tensor([1.5, -1.0], dtype=dtype))
    A = torch.tensor([[2.0, 0.5], [-1.0, 1.5]], dtype=dtype)
    b = torch.tensor([1.0, -0.5], dtype=dtype)
    return x, A, b


def _assert_positive_definite_covariance(P):
    assert torch.allclose(P, P.T, atol=1e-10)
    assert torch.linalg.eigvalsh(P).min().item() > 0


def test_extended_kalman_filter_covariance_stays_positive_definite():
    x, A, b = _kalman_covariance_problem()
    optimizer = ExtendedKalmanFilter([x])

    for _ in range(20):
        optimizer.step(lambda: A @ x - b)

    _assert_positive_definite_covariance(optimizer.P)


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


def test_kalman_filter_state_dict_restores_covariance():
    x = _scalar_param()
    optimizer = KalmanFilter([x])

    def closure():
        return x.view(-1), torch.eye(1, device=x.device, dtype=x.dtype)

    optimizer.step(closure)
    state_dict = optimizer.state_dict()

    y = _scalar_param()
    restored = KalmanFilter([y])
    restored.load_state_dict(state_dict)

    assert torch.equal(optimizer.P, restored.P)


def test_kalman_filter_q_update_changes_Q():
    x = _scalar_param()
    optimizer = KalmanFilter([x], q=1.0)

    optimizer.q = 5.0

    assert optimizer.Q[0, 0].item() == pytest.approx(5.0)


def test_kalman_filter_covariance_stays_positive_definite():
    x, A, b = _kalman_covariance_problem()
    optimizer = KalmanFilter([x])

    for _ in range(20):
        optimizer.step(lambda: (A @ x - b, A))

    _assert_positive_definite_covariance(optimizer.P)
