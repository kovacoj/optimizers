from collections.abc import Callable

import torch
from ._utils import _ParamGroupDefault
from ._utils import flat_params
from ._utils import load_flat_params_
from ._utils import trainable_params


class Genetic(torch.optim.Optimizer):
    mutation_rate = _ParamGroupDefault()
    mutation_strength = _ParamGroupDefault()
    elite_ratio = _ParamGroupDefault()
    pop_size = _ParamGroupDefault()
    noise_scale = _ParamGroupDefault()

    def __init__(self, params, noise_scale=1.0):
        super().__init__(params, dict(
            mutation_rate=0.1,
            mutation_strength=0.1,
            elite_ratio=0.2,
            pop_size=100,
            noise_scale=noise_scale,
        ))

        params = trainable_params(self.param_groups)
        self.numel = sum(param.numel() for param in params)

        if self.numel == 0:
            raise ValueError("Genetic requires at least one trainable parameter")

        self.best_genome = self.params
        self.best_fitness = float('inf')

        self.population = self.params.unsqueeze(0).repeat(self.pop_size, 1)
        self.population += torch.randn_like(self.population) * self.noise_scale

        self.helper = torch.optim.Adam(self.param_groups)

    @property
    def params(self):
        return flat_params(trainable_params(self.param_groups))

    def _state_param(self):
        return trainable_params(self.param_groups)[0]

    @property
    def best_genome(self):
        return self.state[self._state_param()]['best_genome']

    @best_genome.setter
    def best_genome(self, value):
        self.state[self._state_param()]['best_genome'] = value

    @property
    def best_fitness(self):
        return self.state[self._state_param()]['best_fitness']

    @best_fitness.setter
    def best_fitness(self, value):
        self.state[self._state_param()]['best_fitness'] = value

    @property
    def population(self):
        return self.state[self._state_param()]['population']

    @population.setter
    def population(self, value):
        self.state[self._state_param()]['population'] = value

    @torch.no_grad()
    def mutate(self, genome):
        mask = torch.rand_like(genome) < self.mutation_rate
        noise = torch.randn_like(genome)*self.mutation_strength

        genome[mask] += noise[mask]

        return genome

    @torch.no_grad()
    def crossover(self, parent1, parent2):
        if self.numel < 2:
            return parent1.clone(), parent2.clone()

        crossover_point = torch.randint(1, self.numel, (1,))[0]

        child1 = torch.cat((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = torch.cat((parent2[:crossover_point], parent1[crossover_point:]))

        return child1.clone(), child2.clone()

    @torch.no_grad()
    def directional(self, closure, steps=0):
        self.helper.state.clear()

        for _ in range(steps):
            self.helper.zero_grad()
            closure().backward()
            self.helper.step()

        return closure()

    @torch.no_grad()
    def step(self, closure: Callable):
        loss = torch.empty(self.pop_size)

        for idx in range(self.pop_size):
            load_flat_params_(trainable_params(self.param_groups), self.population[idx])
            loss[idx] = self.directional(closure)
            self.population[idx] = self.params.clone()

        best_idx = torch.argmin(loss)
        current_best_fitness = loss[best_idx].item()

        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_genome = self.population[best_idx].clone()

        num_parents = max(1, int(self.pop_size * self.elite_ratio))
        sorted_indices = torch.argsort(loss, descending=False)
        parent_indices = sorted_indices[:num_parents]
        parents = self.population[parent_indices]

        new_population = [self.best_genome.clone()]

        while len(new_population) < self.pop_size:
            p1_idx, p2_idx = torch.randint(0, num_parents, (2,)).unbind()

            p1 = parents[p1_idx]
            p2 = parents[p2_idx]

            child1, child2 = self.crossover(p1, p2)

            new_population.append(self.mutate(child1))
            if len(new_population) < self.pop_size:
                new_population.append(self.mutate(child2))

        self.population = torch.stack(new_population)

        load_flat_params_(trainable_params(self.param_groups), self.best_genome)

        return closure()
