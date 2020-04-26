import matplotlib.pyplot as plt
import numpy as np
from typing import Callable


class GeneticOptimizer:
    def __init__(self, search_space: dict, population_size, selection_rate, crossover_rate, mutation_rate):
        self.search_space = search_space

        # GA params
        self.population_size = population_size
        self.selection_rate = selection_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        self.hist_params = []
        self.hist_target = []

        members = np.array([PopulationMember(search_space) for _ in range(population_size)])
        self.population = Population(members)
        self.best_solution = None
        self.best_target = None

    def optimize(self, eval_function, iterations, early_stop=None, target="min"):
        argbest: Callable = np.argmin if target == "min" else np.argmax
        early_stop_counter = 0
        for i in range(iterations):
            self.population.fitness(eval_function, target=target)

            best_member_idx = argbest(self.population.normalized_fitness)
            self.hist_params.append(self.population.members[best_member_idx].params)
            self.hist_target.append(self.population.members[best_member_idx].fitness)

            try:
                if (target == "min" and self.population.members[best_member_idx].fitness < self.best_solution)\
                        or (target == "max" and self.population.members[best_member_idx].fitness > self.best_solution):
                    self.best_solution = self.population.members[best_member_idx].fitness
                    early_stop_counter = 0
            except TypeError:
                self.best_solution = self.population.members[best_member_idx].fitness

            self.population.selection(self.selection_rate)
            self.population.crossover(self.crossover_rate)
            self.population.mutation(self.mutation_rate)

            if self.optimization_stop_conditions(i, early_stop, early_stop_counter):
                break

            early_stop_counter += 1

        best_idx = argbest(self.hist_target)
        self.best_solution = self.hist_params[best_idx]
        self.best_target = self.hist_target[best_idx]

    def optimization_stop_conditions(self, iteration, early_stop, early_stop_counter):
        if not len(self.population.members):
            print("Optimization stopped after {} iterations".format(iteration+1))
            return True
        if early_stop is not None and early_stop_counter >= early_stop:
            print("Early stop criteria met after {} iterations".format(iteration+1))
            return True
        return False

    def plot_solution_history(self):
        plt.scatter(x=range(len(self.hist_target)), y=self.hist_target)


class Population:
    def __init__(self, members=None):
        self.members = members
        self.size = len(members)
        self.nominal_fitness = None
        self.normalized_fitness = None

    def fitness(self, eval_function, target="min"):
        assert target in ("min", "max")
        [member.calculate_fitness(eval_function) for member in self.members]
        self.nominal_fitness = np.array([member.fitness for member in self.members])

        # Rescale fitness to [0, 1]
        dummy_min_value = self.nominal_fitness.min() - 0.00001  # Add false minimum to avoid division by zero
        scaled_fitness = self.rescale_min_max(np.append(self.nominal_fitness, dummy_min_value))
        scaled_fitness = scaled_fitness[scaled_fitness > 0]

        if target == "max":
            total_fitness = np.sum(scaled_fitness)
            self.normalized_fitness = scaled_fitness / total_fitness
        elif target == "min":
            total_fitness = np.sum(1 / scaled_fitness)
            self.normalized_fitness = (1 / scaled_fitness) / total_fitness

    def selection(self, selection_rate):
        self.members = np.random.choice(self.members, size=int(selection_rate * self.size), replace=False,
                                        p=self.normalized_fitness)
        self.size = len(self.members)

    def crossover(self, crossover_rate):
        pairs = np.random.choice(self.members, self.size // 2 * 2, replace=False).reshape(self.size // 2, 2)
        for parents in pairs:
            parents[0].cross_members(parents[1], crossover_rate)

    def mutation(self, mutation_rate, min_genoms=1):
        for member in self.members:
            member.mutate(mutation_rate, min_genoms)

    @staticmethod
    def rescale_min_max(x):
        return (x - x.min()) / (x.max() - x.min())


class PopulationMember:
    def __init__(self, search_space):
        self.params = {name: param.get_value() for name, param in search_space.items()}
        self.search_space = search_space
        self.fitness = None
        self.normalized_fitness = None

    def calculate_fitness(self, eval_function):
        self.fitness = eval_function(**self.params)

    def mutate(self, mutation_rate, min_genoms=1):
        size = max(min_genoms, int(mutation_rate * len(self.params)))
        chosen_params = np.random.choice([*self.params], size=size, replace=False)

        for param in chosen_params:
            self.params[param] = self.search_space[param].get_value()

    def cross_members(self, other, crossover_rate):
        # Uniform crossover
        if np.random.random() < crossover_rate:
            param_list = self.params.keys()
            points = np.random.randint(0, 2, size=len(param_list))

            self_params = {param: self.params[param] if add_to_self else other.params[param]
                           for add_to_self, param in zip(points, param_list)}
            other_params = {param: self.params[param] if not add_to_self else other.params[param]
                            for add_to_self, param in zip(points, param_list)}

            self.params = self_params
            other.params = other_params

