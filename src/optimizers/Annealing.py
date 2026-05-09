from collections.abc import Callable

import torch

from ._utils import _ParamGroupDefault
from ._utils import flat_params
from ._utils import load_flat_params_
from ._utils import trainable_params


class Annealing(torch.optim.Optimizer):
    temperature = _ParamGroupDefault()
    cooling_rate = _ParamGroupDefault()

    def __init__(self, params, cooling_rate=1e-3):
        super().__init__(params, dict(temperature=1.0, cooling_rate=cooling_rate))

        params = trainable_params(self.param_groups)
        self.numel = sum(param.numel() for param in params)

        if self.numel == 0:
            raise ValueError("Annealing requires at least one trainable parameter")

    @property
    def params(self):
        return flat_params(trainable_params(self.param_groups))

    @torch.no_grad()
    def mutate(self):
        return self.params + torch.randn_like(
            self.params
        )*self.temperature

    def step(self, closure: Callable):
        variants = [self.params, self.mutate().clone()]
        Fs = torch.empty(2)

        for idx, params in enumerate(variants):
            load_flat_params_(trainable_params(self.param_groups), params)
            Fs[idx] = closure().item()

        idx = int((torch.rand(1) < torch.exp(
            (Fs[0] - Fs[1]) / self.temperature
        )).item())

        load_flat_params_(trainable_params(self.param_groups), variants[idx])
        self.temperature *= 1 - self.cooling_rate

        with torch.enable_grad():
            return closure()
