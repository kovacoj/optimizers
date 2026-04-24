from collections.abc import Callable

import torch

from ._utils import _ParamGroupDefault
from ._utils import add_flat_update_
from ._utils import kalman_update
from ._utils import residual_sum_squares
from ._utils import trainable_params


class KalmanFilter(torch.optim.Optimizer):
    """Kalman filter optimizer for linear residual closures.

    This optimizer assumes a single parameter group and expects `closure()` to
    return `(errors, H)`, where `errors` is a residual vector and `H` is the
    linear observation matrix. The covariance `P` is persistent optimizer
    state, while `eta`, `eps`, `q`, and `tau` are live param-group properties.
    """

    eta = _ParamGroupDefault()
    eps = _ParamGroupDefault()
    q = _ParamGroupDefault()
    tau = _ParamGroupDefault()

    def __init__(self, params, eta = 1e3, eps = 1e-3, q = 1e-6, tau = 1):
        defaults = dict(
                    eta = eta,
                    eps = eps,
                    q = q,
                    tau = tau
                )
        
        super().__init__(params, defaults)

        params = trainable_params(self.param_groups)
        self.numel = sum(param.numel() for param in params)

        if self.numel == 0:
            raise ValueError("KalmanFilter requires at least one trainable parameter")

        self.state[params[0]]['P'] = torch.eye(
            self.numel,
            device=params[0].device,
            dtype=params[0].dtype,
        )

    @property
    def P(self):
        return self.state[trainable_params(self.param_groups)[0]]['P']

    @P.setter
    def P(self, value):
        self.state[trainable_params(self.param_groups)[0]]['P'] = value

    @property
    def Q(self):
        prototype = trainable_params(self.param_groups)[0]
        return self.q * torch.eye(self.numel, device=prototype.device, dtype=prototype.dtype)

    def loss(self, errors):
        return residual_sum_squares(errors)
    
    @torch.no_grad()
    def update_weights(self, updates):
        add_flat_update_(trainable_params(self.param_groups), updates)

    def step(self, closure: Callable):

        if len(self.param_groups) != 1:
            raise ValueError("KalmanFilter requires exactly one parameter group")

        with torch.no_grad():
            errors, H = closure()

        updates, self.P, linearized_errors = kalman_update(
            errors=errors,
            H=H,
            P=self.P,
            Q=self.Q,
            eta=self.eta,
            eps=self.eps,
            tau=self.tau,
        )

        self.update_weights(updates)

        return self.loss(linearized_errors.view(-1))
