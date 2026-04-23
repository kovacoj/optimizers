import torch
from .Newton import Newton
from ._utils import _FlatParamOptimizerMixin
from ._utils import trainable_params


class Genetic(_FlatParamOptimizerMixin, torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, dict(
            mutation_rate=0.1,
            mutation_strength=0.1,
            elite_ratio=0.2,
            pop_size=100,
        )) 

        params = trainable_params(self.param_groups)
        self.numel = sum(param.numel() for param in params)

        if self.numel == 0:
            raise ValueError("Genetic requires at least one trainable parameter")

        self.best_genome = self.params
        self.best_fitness = float('inf')

        self.population = self.params.unsqueeze(0).repeat(self.pop_size, 1)
        self.population += torch.randn_like(self.population) * 1e-0

        self.helper = torch.optim.Adam(self.param_groups)
        # self.helper = Newton(self.param_groups[0]['params'])

    def _state_param(self):
        return trainable_params(self.param_groups)[0]

    @property
    def mutation_rate(self):
        return self.param_groups[0]['mutation_rate']

    @mutation_rate.setter
    def mutation_rate(self, value):
        self.param_groups[0]['mutation_rate'] = value

    @property
    def mutation_strength(self):
        return self.param_groups[0]['mutation_strength']

    @mutation_strength.setter
    def mutation_strength(self, value):
        self.param_groups[0]['mutation_strength'] = value

    @property
    def elite_ratio(self):
        return self.param_groups[0]['elite_ratio']

    @elite_ratio.setter
    def elite_ratio(self, value):
        self.param_groups[0]['elite_ratio'] = value

    @property
    def pop_size(self):
        return self.param_groups[0]['pop_size']

    @pop_size.setter
    def pop_size(self, value):
        self.param_groups[0]['pop_size'] = value

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

    @torch.no_grad
    def mutate(self, genome):
        mask = torch.rand_like(genome) < self.mutation_rate
        noise = torch.randn_like(genome)*self.mutation_strength**2

        genome[mask] += noise[mask]

        return genome

    @torch.no_grad
    def crossover(self, parent1, parent2):
        if self.numel < 2:
            return parent1.clone(), parent2.clone()

        crossover_point = torch.randint(1, self.numel, (1,))[0]

        child1 = torch.cat((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = torch.cat((parent2[:crossover_point], parent1[crossover_point:]))

        return child1.clone(), child2.clone()

    # @torch.no_grad
    def directional(self, closure, steps=0):
        self.helper.state.clear()

        for _ in range(steps):
            self.helper.zero_grad()
            closure().backward()
            self.helper.step()
            # self.helper.step(closure)

        return closure()

    # @torch.no_grad
    def step(self, closure):
        loss = torch.empty(self.pop_size)

        for idx in range(self.pop_size):            
            self.update_weights(self.population[idx])
            loss[idx] = self.directional(closure)
            self.population[idx] = self.params.clone()

        # TRACK BEST INDIVIDUAL (ELITISM)
        best_idx = torch.argmin(loss)
        current_best_fitness = loss[best_idx].item()

        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_genome = self.population[best_idx].clone()

        # SELECTION (Truncation Selection)
        num_parents = max(1, int(self.pop_size * self.elite_ratio))
        sorted_indices = torch.argsort(loss, descending=False)
        parent_indices = sorted_indices[:num_parents]
        parents = self.population[parent_indices]

        # REPRODUCTION
        new_population = [self.best_genome.clone()]

        while len(new_population) < self.pop_size:
            p1_idx, p2_idx = torch.randint(0, num_parents, (2,)).unbind()
            
            p1 = parents[p1_idx]
            p2 = parents[p2_idx]

            # Crossover
            child1, child2 = self.crossover(p1, p2)

            # Mutation
            new_population.append(self.mutate(child1))
            if len(new_population) < self.pop_size:
                new_population.append(self.mutate(child2))

        self.population = torch.stack(new_population)

        # pdate best weights for external calls to model(x)
        self.update_weights(self.best_genome)

        return closure()
