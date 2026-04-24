from collections.abc import Callable

import torch

from ._utils import _FlatParamOptimizer
from ._utils import trainable_params


class Metropolis(_FlatParamOptimizer, torch.optim.Optimizer):
    def __init__(self, params, cooling_rate=1e-3):
        super().__init__(params, dict(temperature=10.0, cooling_rate=cooling_rate)) 

        params = trainable_params(self.param_groups)
        self.numel = sum(param.numel() for param in params)

        if self.numel == 0:
            raise ValueError("Metropolis requires at least one trainable parameter")

    @property
    def temperature(self):
        return self.param_groups[0]['temperature']

    @temperature.setter
    def temperature(self, value):
        self.param_groups[0]['temperature'] = value

    @property
    def cooling_rate(self):
        return self.param_groups[0]['cooling_rate']

    @cooling_rate.setter
    def cooling_rate(self, value):
        self.param_groups[0]['cooling_rate'] = value

    @torch.no_grad()
    def mutate(self, ):
        proposal = self.params.clone()
        idx = torch.randint(low=0, high=self.numel, size=()).item()
        proposal[idx] += torch.randn((), device=proposal.device, dtype=proposal.dtype) * self.temperature

        return proposal

    @torch.no_grad()
    def step(self, closure: Callable):
        variants = [self.params, self.mutate().clone()]
        Fs = torch.empty(2)

        for idx, params in enumerate(variants):
            self.update_weights(params)
            Fs[idx] = closure().item()

        idx = int((torch.rand(1) < torch.exp(
            (Fs[0] - Fs[1]) / self.temperature
        )).item())

        self.update_weights(variants[idx])
        self.temperature *= 1 - self.cooling_rate

        return closure()
