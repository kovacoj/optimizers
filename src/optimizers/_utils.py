import torch
from torch.autograd import grad


def all_params(param_groups):
    return [
        param
        for group in param_groups
        for param in group['params']
    ]


def trainable_params(param_groups):
    return [
        param
        for group in param_groups
        for param in group['params']
        if param.requires_grad
    ]


def flat_params(params):
    return torch.cat([
        param.flatten()
        for param in params
    ])


@torch.no_grad()
def load_flat_params_(params, values):
    values = values.view(-1)

    offset = 0
    for param in params:
        numel = param.numel()

        param.copy_(values[offset: offset + numel].view_as(param))
        offset += numel


@torch.no_grad()
def add_flat_update_(params, update):
    update = update.view(-1)

    offset = 0
    for param in params:
        numel = param.numel()

        param.add_(update[offset: offset + numel].view_as(param))
        offset += numel


def residual_jacobian(targets, params):
    numel = sum(param.numel() for param in params)
    J = targets.new_empty(targets.shape[0], numel)

    for i in range(targets.shape[0]):
        J[i] = torch.hstack([
            d.view(1, -1) if d is not None else param.new_zeros(1, param.numel())
            for param, d in zip(
                params,
                grad(targets[i], params, create_graph=True, retain_graph=True, allow_unused=True)
            )
        ])

    return J


def residual_sum_squares(errors):
    return errors @ errors


def kalman_update(errors, H, P, Q, eta, eps, tau):
    errors = errors.view(-1).clone()
    P = P / tau + Q

    R = ((1 / eta) + eps) * torch.eye(errors.shape[0], device=errors.device, dtype=errors.dtype)
    A = H @ P @ H.T + R

    K = torch.linalg.solve(A, H @ P).T
    updates = -(K @ errors).view(-1)

    I = torch.eye(P.shape[0], device=P.device, dtype=P.dtype)
    IKH = I - K @ H
    next_P = IKH @ P @ IKH.T + K @ R @ K.T

    return updates, next_P, errors + H @ updates


class _FlatParamOptimizer:
    @property
    def params(self):
        return flat_params(trainable_params(self.param_groups))

    @torch.no_grad()
    def update_weights(self, update):
        load_flat_params_(trainable_params(self.param_groups), update)


class _FlatUpdateOptimizer:
    @torch.no_grad()
    def update_weights(self, update):
        add_flat_update_(trainable_params(self.param_groups), update)

    @torch.no_grad()
    def _set_params(self, params, values):
        load_flat_params_(params, values)
