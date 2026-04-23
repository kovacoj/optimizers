import torch

from ._utils import add_flat_update_
from ._utils import residual_jacobian
from ._utils import residual_sum_squares
from ._utils import trainable_params


class ExtendedKalmanFilter(torch.optim.Optimizer):
    # only one parameter group can be used!
    # add @property q(t)
    def __init__(self, params, eta = 1e3, eps = 1e-3, q = 1e-6, tau = 1):
        self.eta, self.eps, self.q, self.tau = eta, eps, q, tau

        defaults = dict(
                    eta = self.eta,
                    eps = self.eps,
                    q = self.q,
                    tau = self.tau
                )
        
        super(ExtendedKalmanFilter, self).__init__(params, defaults)

        params = trainable_params(self.param_groups)
        self.numel = sum(param.numel() for param in params)

        if self.numel == 0:
            raise ValueError("ExtendedKalmanFilter requires at least one trainable parameter")

        prototype = params[0]

        self.P = torch.eye(self.numel, device=prototype.device, dtype=prototype.dtype)
        self.Q = self.q * torch.eye(self.numel, device=prototype.device, dtype=prototype.dtype)

    def jacobian(self, targets):
        return residual_jacobian(targets, trainable_params(self.param_groups))
    
    def loss(self, errors):
        return residual_sum_squares(errors)
    
    @torch.no_grad()
    def update_weights(self, updates):
        add_flat_update_(trainable_params(self.param_groups), updates)

    def step(self, closure = None):

        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        # errors need to be computed from closure
        # closure (callable) - reevaluates the model and returns the loss, in our case the errors
        errors = closure()
        
        H = self.jacobian(errors)

        P = self.P / self.tau + self.Q

        A = H @ P @ H.T + ((1 / self.eta) + self.eps) * torch.eye(errors.shape[0], device=errors.device, dtype=errors.dtype)

        K = torch.linalg.solve(A, H @ P).T

        updates = -(K @ errors).view(-1)

        self.update_weights(updates)

        self.P = P - K @ H @ P

        return self.loss(closure())
