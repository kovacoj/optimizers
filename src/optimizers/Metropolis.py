from collections.abc import Callable

import torch

from .Annealing import Annealing


class Metropolis(Annealing):
    def __init__(self, params, cooling_rate=1e-3):
        super().__init__(params, cooling_rate=cooling_rate)
        self.temperature = 10.0

    @torch.no_grad()
    def mutate(self):
        proposal = self.params.clone()
        idx = torch.randint(low=0, high=self.numel, size=()).item()
        proposal[idx] += torch.randn((), device=proposal.device, dtype=proposal.dtype) * self.temperature

        return proposal
