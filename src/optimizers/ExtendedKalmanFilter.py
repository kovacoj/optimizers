from collections.abc import Callable

import torch

from ._utils import _ParamGroupDefault
from ._utils import add_flat_update_
from ._utils import kalman_update
from ._utils import residual_jacobian
from ._utils import residual_sum_squares
from ._utils import trainable_params


class ExtendedKalmanFilter(torch.optim.Optimizer):
    """Extended Kalman filter optimizer for residual-vector closures.

    This optimizer assumes a single parameter group and expects `closure()` to
    return a 1D tensor of residuals. The Kalman covariance `P` is persistent
    optimizer state, while `eta`, `eps`, `q`, and `tau` are live properties on
    the single param group.
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
            raise ValueError("ExtendedKalmanFilter requires at least one trainable parameter")

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

    def jacobian(self, targets):
        return residual_jacobian(targets, trainable_params(self.param_groups))
    
    def loss(self, errors):
        return residual_sum_squares(errors)
    
    def step(self, closure: Callable):

        if len(self.param_groups) != 1:
            raise ValueError("ExtendedKalmanFilter requires exactly one parameter group")

        with torch.enable_grad():
            errors = closure()
            H = self.jacobian(errors)

        updates, self.P, _ = kalman_update(
            errors=errors,
            H=H,
            P=self.P,
            Q=self.Q,
            eta=self.eta,
            eps=self.eps,
            tau=self.tau,
        )

        add_flat_update_(trainable_params(self.param_groups), updates)

        with torch.no_grad():
            return self.loss(closure())
