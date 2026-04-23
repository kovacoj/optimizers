import torch

from ._utils import add_flat_update_
from ._utils import residual_jacobian
from ._utils import residual_sum_squares
from ._utils import trainable_params
    

class LevenbergMarquardt(torch.optim.Optimizer):
    # only one parameter group can be used!
    # need to add comments
    # update defaults
    # add loss history; batches can be controled from closure()
    def __init__(self, params, mu = 10**3, mu_factor = 5, m_max = 10, strategy = "line search"):
        defaults = dict(mu = mu,
                        mu_factor = mu_factor,
                        m_max = m_max,
                        strategy = strategy,
                    )
        
        super(LevenbergMarquardt, self).__init__(params, defaults)

        params = trainable_params(self.param_groups)
        self.numel = sum(param.numel() for param in params)
        # self.numel = reduce(lambda total, p: total + p.numel(), self.param_groups, 0)

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
    def mu(self):
        return self.param_groups[0]['mu']

    @mu.setter
    def mu(self, value):
        self.param_groups[0]['mu'] = value

    @property
    def mu_factor(self):
        return self.param_groups[0]['mu_factor']

    @mu_factor.setter
    def mu_factor(self, value):
        self.param_groups[0]['mu_factor'] = value

    @property
    def m_max(self):
        return self.param_groups[0]['m_max']

    @m_max.setter
    def m_max(self, value):
        self.param_groups[0]['m_max'] = value

    @property
    def strategy(self):
        return self._canonical_strategy(self.param_groups[0]['strategy'])

    @strategy.setter
    def strategy(self, value):
        self.param_groups[0]['strategy'] = self._canonical_strategy(value)

    # @torch.compile ?
    def jacobian(self, targets):
        return residual_jacobian(targets, trainable_params(self.param_groups))
    
    @torch.no_grad()
    def loss(self, errors):
        return residual_sum_squares(errors)
    
    @torch.no_grad()
    def update_weights(self, update):
        add_flat_update_(trainable_params(self.param_groups), update)

    def _lm_step(self, J, errors):
        system = J.T @ J + (self.mu + 1e-8) * torch.eye(
            self.numel,
            device=self.prototype.device,
            dtype=self.prototype.dtype,
        )
        return -torch.linalg.solve(system, J.T @ errors)

    def _step_line_search(self, closure, J, errors, base_loss):
        updates = self._lm_step(J, errors)

        self.update_weights(updates)

        loss_decreased = self.loss(closure()) < base_loss
        for _ in range(self.m_max):
            if loss_decreased:
                break

            self.update_weights(update = -updates)

            self.mu *= self.mu_factor
            updates = self._lm_step(J, errors)
            self.update_weights(update = +updates)

            loss_decreased = self.loss(closure()) < base_loss

        if loss_decreased:
            self.mu /= self.mu_factor
        else:
            self.update_weights(update = -updates)

        return self.loss(closure()).item()

    def _step_trust_region(self, closure, J, errors, base_loss):
        gradient = J.T @ errors
        nu = 2.0

        for _ in range(self.m_max + 1):
            updates = self._lm_step(J, errors)
            predicted_reduction = updates @ ((self.mu + 1e-8) * updates - gradient)

            if predicted_reduction <= 0:
                self.mu *= nu
                nu *= 2.0
                continue

            self.update_weights(updates)
            new_loss = self.loss(closure())
            actual_reduction = base_loss - new_loss
            rho = actual_reduction / predicted_reduction

            if rho > 0:
                self.mu *= max(1 / 3, 1 - (2 * rho - 1) ** 3)
                return new_loss.item()

            self.update_weights(-updates)
            self.mu *= nu
            nu *= 2.0

        return base_loss.item()

    def step(self, closure = None):

        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        # errors need to be computed from closure
        # closure (callable) - reevaluates the model and returns the loss, in our case the errors
        errors = closure()
        base_loss = self.loss(errors)
        
        # compute Jacobian matrix
        J = self.jacobian(errors)

        if self.strategy == "trust region":
            return self._step_trust_region(closure, J, errors, base_loss)

        return self._step_line_search(closure, J, errors, base_loss)
