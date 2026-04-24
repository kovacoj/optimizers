from collections.abc import Callable

import torch

from ._utils import _ParamGroupDefault
from ._utils import add_flat_update_
from ._utils import flat_params
from ._utils import load_flat_params_
from ._utils import residual_jacobian
from ._utils import residual_sum_squares
from ._utils import trainable_params
from .line_search import _canonical_line_search_method
from .line_search import armijo_backtracking
from .line_search import strong_wolfe


class LevenbergMarquardt(torch.optim.Optimizer):
    """Levenberg-Marquardt optimizer for residual-vector closures.

    This optimizer assumes a single parameter group and expects `closure()` to
    return a 1D tensor of residuals. The `strategy` argument selects between a
    line-search style damping update and a trust-region gain-ratio update.
    `solve_epsilon` adds a small diagonal jitter to the linear solve.

    `step()` returns the final residual sum of squares as a Python float, so
    callers can keep any loss history externally.
    """

    mu = _ParamGroupDefault()
    mu_factor = _ParamGroupDefault()
    m_max = _ParamGroupDefault()
    solve_epsilon = _ParamGroupDefault()

    def __init__(self, params, mu = 10**3, mu_factor = 5, m_max = 10, strategy = "line search", line_search_method = "armijo", solve_epsilon = 1e-8):
        defaults = dict(mu = mu,
                        mu_factor = mu_factor,
                        m_max = m_max,
                        strategy = strategy,
                        line_search_method = line_search_method,
                        solve_epsilon = solve_epsilon,
                    )
        
        super().__init__(params, defaults)

        params = trainable_params(self.param_groups)
        self.numel = sum(param.numel() for param in params)

        if self.numel == 0:
            raise ValueError("LevenbergMarquardt requires at least one trainable parameter")

        self.strategy = strategy

        self.prototype = params[0]

    def _canonical_strategy(self, value):
        aliases = {
            "line search": "line search",
            "trust region": "trust region",
            "line_search": "line search",
            "trust_region": "trust region",
            "heuristic": "line search",
        }

        if value not in aliases:
            raise ValueError(
                "LevenbergMarquardt strategy must be 'line search' or 'trust region'"
            )

        return aliases[value]

    @property
    def strategy(self):
        return self._canonical_strategy(self.param_groups[0]['strategy'])

    @strategy.setter
    def strategy(self, value):
        self.param_groups[0]['strategy'] = self._canonical_strategy(value)

    @property
    def line_search_method(self):
        return _canonical_line_search_method(
            self.param_groups[0]['line_search_method'],
            optimizer_name="LevenbergMarquardt",
        )

    @line_search_method.setter
    def line_search_method(self, value):
        self.param_groups[0]['line_search_method'] = _canonical_line_search_method(
            value,
            optimizer_name="LevenbergMarquardt",
        )

    def jacobian(self, targets):
        return residual_jacobian(targets, trainable_params(self.param_groups))
    
    @torch.no_grad()
    def loss(self, errors):
        return residual_sum_squares(errors)
    
    def _lm_step(self, J, errors):
        damping = self.mu + self.solve_epsilon
        system = J.T @ J + damping * torch.eye(
            self.numel,
            device=self.prototype.device,
            dtype=self.prototype.dtype,
        )
        return -torch.linalg.solve(system, J.T @ errors)

    def _step_line_search(self, closure, J, errors, base_loss):
        params = trainable_params(self.param_groups)
        base_params = flat_params(params).clone()
        direction = self._lm_step(J, errors)

        def phi(alpha):
            load_flat_params_(params, base_params + alpha * direction)
            return 0.5 * self.loss(closure())

        def dphi(alpha):
            load_flat_params_(params, base_params + alpha * direction)
            trial_errors = closure()
            trial_jacobian = self.jacobian(trial_errors)
            return (trial_jacobian.T @ trial_errors) @ direction

        phi0 = 0.5 * base_loss
        dphi0 = (J.T @ errors) @ direction

        if self.line_search_method == "wolfe":
            alpha, phi_alpha, _ = strong_wolfe(
                phi=phi,
                dphi=dphi,
                phi0=phi0,
                dphi0=dphi0,
                alpha0=phi0.new_tensor(1.0),
                max_iters=self.m_max + 1,
                zoom_iters=self.m_max + 1,
            )
        else:
            alpha, phi_alpha = armijo_backtracking(
                phi=phi,
                phi0=phi0,
                dphi0=dphi0,
                alpha0=phi0.new_tensor(1.0),
                max_iters=self.m_max + 1,
            )

        load_flat_params_(params, base_params + alpha * direction)
        return (2.0 * phi_alpha).item()

    def _step_trust_region(self, closure, J, errors, base_loss):
        gradient = J.T @ errors
        nu = 2.0

        for _ in range(self.m_max + 1):
            updates = self._lm_step(J, errors)
            damping = self.mu + self.solve_epsilon
            predicted_reduction = updates @ (damping * updates - gradient)

            if predicted_reduction <= 0:
                self.mu *= nu
                nu *= 2.0
                continue

            add_flat_update_(trainable_params(self.param_groups), updates)
            new_loss = self.loss(closure())
            actual_reduction = base_loss - new_loss
            rho = actual_reduction / predicted_reduction

            if rho > 0:
                self.mu *= max(1 / 3, 1 - (2 * rho - 1) ** 3)
                return new_loss.item()

            add_flat_update_(trainable_params(self.param_groups), -updates)
            self.mu *= nu
            nu *= 2.0

        return base_loss.item()

    def step(self, closure: Callable):

        if len(self.param_groups) != 1:
            raise ValueError("LevenbergMarquardt requires exactly one parameter group")

        closure = torch.enable_grad()(closure)

        errors = closure()
        base_loss = self.loss(errors)
        
        J = self.jacobian(errors)

        if self.strategy == "trust region":
            return self._step_trust_region(closure, J, errors, base_loss)

        return self._step_line_search(closure, J, errors, base_loss)
