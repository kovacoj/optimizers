import torch

from ._utils import _FlatParamOptimizerMixin
from ._utils import all_params


class Annealing(_FlatParamOptimizerMixin, torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, {}) 

        params = trainable_params(self.param_groups)
        self.numel = sum(
            param.numel() for param in all_params(self.param_groups)
        )

        if self.numel == 0:
            raise ValueError("Annealing requires at least one trainable parameter")

        self.temperature = 1

    @torch.no_grad
    def mutate(self):
        return self.params + torch.randn_like(
            self.params
        )*self.temperature

    @torch.no_grad
    def step(self, closure):
        variants = [self.params, self.mutate().clone()]
        Fs = torch.empty(2)

        for idx, params in enumerate(variants):
            self.update_weights(params)
            Fs[idx] = closure().item()

        idx = int((torch.rand(1) < torch.exp(
            (Fs[0] - Fs[1]) / self.temperature
        )).item())

        self.update_weights(variants[idx])
        self.temperature *= 1 - 1e-3

        return closure()
