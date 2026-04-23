import torch


class KalmanFilter(torch.optim.Optimizer):
    # only one parameter group can be used!
    # closure() returns (errors, H)
    def __init__(self, params, eta = 1e3, eps = 1e-3, q = 1e-6, tau = 1):
        self.eta, self.eps, self.q, self.tau = eta, eps, q, tau

        defaults = dict(
                    eta = self.eta,
                    eps = self.eps,
                    q = self.q,
                    tau = self.tau
                )
        
        super(KalmanFilter, self).__init__(params, defaults)

        self.numel = sum(param.numel() for group in self.param_groups for param in group['params'] if param.requires_grad)

        if self.numel == 0:
            raise ValueError("KalmanFilter requires at least one trainable parameter")

        prototype = next(
            param for group in self.param_groups for param in group['params'] if param.requires_grad
        )

        self.P = torch.eye(self.numel, device=prototype.device, dtype=prototype.dtype)
        self.Q = self.q * torch.eye(self.numel, device=prototype.device, dtype=prototype.dtype)

    def loss(self, errors):
        # MSE loss, not divided by number of data, doesn't matter
        return errors @ errors
    
    @torch.no_grad()
    def update_weights(self, updates):
        # maybe could be optimzed using torch.chunk
        # Add the updates into the model
        start_idx = 0
        for group in self.param_groups:
            for param in group['params']:
                if not param.requires_grad:
                    continue

                param.data.add_(updates[start_idx:start_idx + param.numel()].view(param.size()))
                start_idx += param.numel()

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
