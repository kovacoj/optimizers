import torch
from torch.autograd import grad

from ._utils import add_flat_update_
from ._utils import trainable_params


class Newton(torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, {})

        params = trainable_params(self.param_groups)
        self.numel = sum(
            param.numel() for param in params
        )

        if self.numel == 0:
            raise ValueError("Newton requires at least one trainable parameter")
    
    @property
    def params(self):
        return torch.cat([
            p.flatten() for group in self.param_groups for p in group['params']
        ])

    @torch.no_grad()
    def update_weights(self, update):
        add_flat_update_(trainable_params(self.param_groups), update)

    def step(self, closure: callable):
        assert len(self.param_groups) == 1

        params = trainable_params(self.param_groups)
        prototype = params[0]
        grads = grad(closure(), params, create_graph=True, allow_unused=True)

        g = torch.cat([
            g.reshape(-1) if g is not None else param.new_zeros(param.numel())
            for param, g in zip(params, grads)
        ])
        H = prototype.new_empty(self.numel, self.numel)

        for idx in range(g.shape[0]):
            if not g[idx].requires_grad:
                H[idx] = prototype.new_zeros(self.numel)
                continue

            H[idx] = torch.hstack([
                d.view(1, -1) if d is not None else param.new_zeros(1, param.numel())
                for param, d in zip(
                    params,
                    grad(g[idx], params, create_graph=True, retain_graph=True, allow_unused=True)
                )
            ])
        H += 1e-4 * torch.eye(self.numel, device=prototype.device, dtype=prototype.dtype) # damping for num. stability

        self.update_weights(
            torch.linalg.solve(H, -g)
        )
