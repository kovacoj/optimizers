import torch

from optimizers import line_search


def _quadratic_phi(alpha):
    return 0.5 * (1 - alpha) ** 2


def _quadratic_dphi(alpha):
    return alpha - 1


def test_line_search_submodule_exports_public_functions():
    assert callable(line_search.armijo_backtracking)
    assert callable(line_search.strong_wolfe)


def test_armijo_backtracking_satisfies_sufficient_decrease():
    phi0 = _quadratic_phi(torch.tensor(0.0))
    dphi0 = _quadratic_dphi(torch.tensor(0.0))

    alpha, phi_alpha = line_search.armijo_backtracking(
        phi=_quadratic_phi,
        phi0=phi0,
        dphi0=dphi0,
        alpha0=torch.tensor(2.0),
    )

    assert isinstance(alpha, torch.Tensor)
    assert isinstance(phi_alpha, torch.Tensor)
    assert bool(phi_alpha <= phi0 + 1e-4 * alpha * dphi0)


def test_strong_wolfe_satisfies_conditions():
    phi0 = _quadratic_phi(torch.tensor(0.0))
    dphi0 = _quadratic_dphi(torch.tensor(0.0))

    alpha, phi_alpha, dphi_alpha = line_search.strong_wolfe(
        phi=_quadratic_phi,
        dphi=_quadratic_dphi,
        phi0=phi0,
        dphi0=dphi0,
        alpha0=torch.tensor(2.0),
    )

    assert isinstance(alpha, torch.Tensor)
    assert isinstance(phi_alpha, torch.Tensor)
    assert isinstance(dphi_alpha, torch.Tensor)
    assert bool(phi_alpha <= phi0 + 1e-4 * alpha * dphi0)
    assert bool(torch.abs(dphi_alpha) <= 0.9 * torch.abs(dphi0))
