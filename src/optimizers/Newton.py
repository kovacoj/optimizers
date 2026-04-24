import torch
from torch.autograd import grad

from ._utils import _FlatUpdateOptimizerMixin
from ._utils import flat_params
from ._utils import trainable_params
from .line_search import _canonical_line_search_method
from .line_search import armijo_backtracking
from .line_search import strong_wolfe


class Newton(_FlatUpdateOptimizerMixin, torch.optim.Optimizer):
    """Dense Newton optimizer for scalar-loss closures.

    This optimizer explicitly materializes the full Hessian of all trainable
    parameters. It is intended for small parameter vectors where dense
    second-order linear algebra is practical.
    """
    def __init__(self, params, line_search_method=None, damping=1e-4):
        super().__init__(params, dict(line_search_method=line_search_method, damping=damping))

        params = trainable_params(self.param_groups)
        self.numel = sum(
            param.numel() for param in params
        )

        if self.numel == 0:
            raise ValueError("Newton requires at least one trainable parameter")

        self.line_search_method = line_search_method
        self.damping = damping

    @property
    def line_search_method(self):
        return _canonical_line_search_method(
            self.param_groups[0]['line_search_method'],
            allow_none=True,
            optimizer_name="Newton",
        )

    @line_search_method.setter
    def line_search_method(self, value):
        self.param_groups[0]['line_search_method'] = _canonical_line_search_method(
            value,
            allow_none=True,
            optimizer_name="Newton",
        )

    @property
    def damping(self):
        return self.param_groups[0]['damping']

    @damping.setter
    def damping(self, value):
        self.param_groups[0]['damping'] = value
    
    def step(self, closure: callable):
        if len(self.param_groups) != 1:
            raise ValueError("Newton requires exactly one parameter group")
        closure = torch.enable_grad()(closure)

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
        H += self.damping * torch.eye(self.numel, device=prototype.device, dtype=prototype.dtype)

        direction = torch.linalg.solve(H, -g)

        if self.line_search_method is None:
            self.update_weights(direction)
            return

        base_params = flat_params(params).clone()
        loss0 = closure()
        dphi0 = g @ direction

        if not bool(dphi0 < 0):
            self._set_params(params, base_params)
            return

        def phi(alpha):
            self._set_params(params, base_params + alpha * direction)
            return closure()

        def dphi(alpha):
            self._set_params(params, base_params + alpha * direction)
            grads = grad(closure(), params, create_graph=True, allow_unused=True)
            gradient = torch.cat([
                grad_.reshape(-1) if grad_ is not None else param.new_zeros(param.numel())
                for param, grad_ in zip(params, grads)
            ])
            return gradient @ direction

        if self.line_search_method == "wolfe":
            alpha, _, _ = strong_wolfe(
                phi=phi,
                dphi=dphi,
                phi0=loss0,
                dphi0=dphi0,
                alpha0=loss0.new_tensor(1.0),
            )
        else:
            alpha, _ = armijo_backtracking(
                phi=phi,
                phi0=loss0,
                dphi0=dphi0,
                alpha0=loss0.new_tensor(1.0),
            )

        self._set_params(params, base_params + alpha * direction)
