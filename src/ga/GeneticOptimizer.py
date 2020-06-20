import matplotlib.pyplot as plt
import numpy as np
from typing import Callable


class GeneticOptimizer:
    def __init__(self, search_space: dict, population_size, crossover_rate, mutation_rate):
        self.search_space = search_space

        # GA params
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        self.hist_params = []
        self.hist_target = []

        self.population = Population(population_size, search_space)
        self.best_solution = None
        self.best_target = None

    def optimize(self, eval_function, iterations=None, metric_target=None, early_stop=None, objective="min", seed=None):
        assert iterations is not None or metric_target is not None, "No stop conditions were specified."
        if seed is not None:
            np.random.seed(seed)

        argbest: Callable = np.argmin if objective == "min" else np.argmax
        early_stop_counter = 0
        i = 0

        while True:
            self.population.fitness(eval_function, objective=objective)

            best_member_idx = argbest(self.population.normalized_fitness)
            self.hist_params.append(self.population.members[best_member_idx].params)
            self.hist_target.append(self.population.members[best_member_idx].fitness)

            try:
                if (objective == "min" and self.population.members[best_member_idx].fitness < self.best_solution)\
                        or (objective == "max" and self.population.members[best_member_idx].fitness > self.best_solution):
                    self.best_solution = self.population.members[best_member_idx].fitness
                    early_stop_counter = 0
            except TypeError:
                # Assign best_solution during first iteration
                self.best_solution = self.population.members[best_member_idx].fitness

            self.population.selection()
            self.population.crossover(self.crossover_rate)
            self.population.mutation(self.mutation_rate, i, iterations)

            if self.optimization_stop_conditions(i, iterations, early_stop, early_stop_counter, objective, metric_target):
                break
            else:
                early_stop_counter += 1
                i += 1

        best_idx = argbest(self.hist_target)
        self.best_solution = self.hist_params[best_idx]
        self.best_target = self.hist_target[best_idx]

    def optimization_stop_conditions(self, i, iterations, early_stop, early_stop_counter, objective, metric_target):
        if not len(self.population.members):
            print("Optimization stopped after {} iterations".format(i + 1))
            return True
        if i >= iterations:
            print("Optimization stopped reaching maximum number of iterations")
            return True
        if early_stop is not None and early_stop_counter >= early_stop:
            print("Early stop criteria met after {} iterations".format(i + 1))
            return True
        if (metric_target is not None and objective == "min" and self.hist_target[-1] <= metric_target) \
                or (metric_target is not None and objective == "max" and self.hist_target[-1] >= metric_target):
            print("Metric target reached after {} iterations".format(i + 1))
            return True
        return False

    def plot_solution_history(self):
        plt.scatter(x=range(len(self.hist_target)), y=self.hist_target, s=10)
        plt.scatter(x=np.where(np.array(self.hist_target) == self.best_target)[0][0], y=self.best_target, color="red")


class Population:
    def __init__(self, size, search_space):
        self.members = np.array([PopulationMember(search_space) for _ in range(size)])
        self.size = size
        self.nominal_fitness = None
        self.normalized_fitness = None

    def fitness(self, eval_function, objective="min"):
        assert objective in ("min", "max")
        [member.calculate_fitness(eval_function) for member in self.members]
        self.nominal_fitness = np.array([member.fitness for member in self.members])

        if objective == "max":
            total_fitness = np.sum(self.nominal_fitness)
            self.normalized_fitness = self.nominal_fitness / total_fitness
        elif objective == "min":
            if np.all(self.nominal_fitness == np.zeros_like(self.nominal_fitness)):
                scaled_fitness = np.ones_like(self.nominal_fitness)
            elif np.all(self.nominal_fitness == self.nominal_fitness[0]):
                scaled_fitness = np.ones_like(self.nominal_fitness)
            else:
                scaled_fitness = self.rescale(self.nominal_fitness, 1, 2)

            total_fitness = np.sum(1 / scaled_fitness)
            self.normalized_fitness = (1 / scaled_fitness) / total_fitness

    def selection(self):
        self.members = np.random.choice(self.members, size=self.size, replace=True, p=self.normalized_fitness)

    def crossover(self, crossover_rate):
        pairs = np.random.choice(self.members, self.size // 2 * 2, replace=False).reshape(self.size // 2, 2)
        for parents in pairs:
            if np.random.rand() < crossover_rate:
                parents[0].cross_members(parents[1])

    def mutation(self, mutation_rate, i, i_max):
        for member in self.members:
            if np.random.random() < mutation_rate:
                member.mutate(i, i_max)

    @staticmethod
    def rescale(x, min_value=0, max_value=1):
        return (x - x.min()) / (x.max() - x.min()) * (max_value - min_value) + min_value


class PopulationMember:
    def __init__(self, search_space):
        self.params = {name: param.get_value() for name, param in search_space.items()}
        self.search_space = search_space
        self.fitness = None
        self.normalized_fitness = None

    def calculate_fitness(self, eval_function):
        self.fitness = eval_function(**self.params)

    def mutate(self, i, i_max):
        param = np.random.choice(list(self.params))
        self.params[param] = self.search_space[param].mutate_param(self.params[param], i, i_max)

    def cross_members(self, other):
        for param in self.params.keys():
            self.search_space[param].cross_param(self.params[param], other.params[param])
