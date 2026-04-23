import torch

from ._utils import add_flat_update_
from ._utils import residual_sum_squares
from ._utils import trainable_params


class KalmanFilter(torch.optim.Optimizer):
    # only one parameter group can be used!
    # closure() returns (errors, H)
    def __init__(self, params, eta = 1e3, eps = 1e-3, q = 1e-6, tau = 1):
        defaults = dict(
                    eta = eta,
                    eps = eps,
                    q = q,
                    tau = tau
                )
        
        super(KalmanFilter, self).__init__(params, defaults)

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
    def eta(self):
        return self.param_groups[0]['eta']

    @eta.setter
    def eta(self, value):
        self.param_groups[0]['eta'] = value

    @property
    def eps(self):
        return self.param_groups[0]['eps']

    @eps.setter
    def eps(self, value):
        self.param_groups[0]['eps'] = value

    @property
    def q(self):
        return self.param_groups[0]['q']

    @q.setter
    def q(self, value):
        self.param_groups[0]['q'] = value

    @property
    def tau(self):
        return self.param_groups[0]['tau']

    @tau.setter
    def tau(self, value):
        self.param_groups[0]['tau'] = value

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

    def step(self, closure = None):

        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        # closure (callable) - reevaluates the model and returns the residuals and linear observation matrix
        errors, H = closure()
        errors = errors.view(-1)

        P = self.P / self.tau + self.Q

        A = H @ P @ H.T + ((1 / self.eta) + self.eps) * torch.eye(errors.shape[0], device=errors.device, dtype=errors.dtype)

        K = torch.linalg.solve(A, H @ P).T

        updates = -(K @ errors).view(-1)

        self.update_weights(updates)

        self.P = P - K @ H @ P

        errors, _ = closure()

        return self.loss(errors.view(-1))
