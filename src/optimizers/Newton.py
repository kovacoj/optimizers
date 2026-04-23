import torch
from torch.autograd import grad


class Newton(torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, {})

        self.numel = sum(
            p.numel() for group in self.param_groups for p in group['params']
        )
    
    @property
    def params(self):
        return torch.cat([
            p.flatten() for group in self.param_groups for p in group['params']
        ])

    @torch.no_grad()
    def update_weights(self, update):
        update = update.flatten()

        offset = 0
        for group in self.param_groups:
            for param in group['params']:
                numel = param.numel()

                param.add_(update[offset: offset + numel].view_as(param))
                offset += numel

    def step(self, closure: callable):
        assert len(self.param_groups) == 1

        prototype = self.param_groups[0]['params'][0]

        params = self.param_groups[0]['params']
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
