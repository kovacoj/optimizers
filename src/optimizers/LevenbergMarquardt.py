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
    def __init__(self, params, mu = 10**3, mu_factor = 5, m_max = 10):
        self.mu = mu
        self.mu_factor = mu_factor
        self.m_max = m_max

        defaults = dict(mu = self.mu,
                        mu_factor = self.mu_factor,
                        m_max = self.m_max
                    )
        
        super(LevenbergMarquardt, self).__init__(params, defaults)

        params = trainable_params(self.param_groups)
        self.numel = sum(param.numel() for param in params)
        # self.numel = reduce(lambda total, p: total + p.numel(), self.param_groups, 0)

        if self.numel == 0:
            raise ValueError("LevenbergMarquardt requires at least one trainable parameter")

        self.prototype = params[0]

    # @torch.compile ?
    def jacobian(self, targets):
        return residual_jacobian(targets, trainable_params(self.param_groups))
    
    @torch.no_grad()
    def loss(self, errors):
        return residual_sum_squares(errors)
    
    @torch.no_grad()
    def update_weights(self, update):
        add_flat_update_(trainable_params(self.param_groups), update)

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

        # compute updates %torch.diag(J.T @ J)%
        updates = -torch.inverse(J.T @ J + (self.mu+1e-8)*torch.eye(self.numel, device=self.prototype.device, dtype=self.prototype.dtype)) @ J.T @ errors

        self.update_weights(updates)

        # line search for mu
        loss_decreased = self.loss(closure()) < base_loss
        for _ in range(self.m_max):
            if loss_decreased:
                break

            # restore weights
            self.update_weights(update = -updates)

            self.mu *= self.mu_factor

            # compute new updates
            updates = -torch.inverse(J.T @ J + (self.mu+1e-8)*torch.eye(self.numel, device=self.prototype.device, dtype=self.prototype.dtype)) @ J.T @ errors

            # update weights
            self.update_weights(update = +updates)

            loss_decreased = self.loss(closure()) < base_loss

        if loss_decreased:
            self.mu /= self.mu_factor
        else:
            self.update_weights(update = -updates)

        # how to return break?, should I return loss?
        # -> returning loss, mu can be controled by
        #  optimizer.mu before running optimizer.step(...)
        return self.loss(closure()).item()
