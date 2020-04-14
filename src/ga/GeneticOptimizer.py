import numpy as np


class GeneticOptimizer:
    def __init__(self, eval_function, search_space: dict, iterations,
                 population_size, selection_rate, crossover_rate, mutation_rate):
        self.iterations = iterations
        self.search_space = search_space
        self.eval_function = eval_function
        self.population = Population()

        # GA params
        self.population_size = population_size
        self.selection_rate = selection_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate


class Population:
    def __init__(self, members=None):
        self.members = members
        self.size = len(members)
        self.fitness = None
        self.normalized_fitness = None

    def population_fitness(self, eval_function, target="min"):
        assert target in ("min", "max")

        self.fitness = np.array([member.calculate_fitness(eval_function) for member in self.members])

        if target == "max":
            total_fitness = np.sum(self.fitness)
            self.normalized_fitness = self.fitness / total_fitness
        elif target == "min":
            total_fitness = np.sum(1 / self.fitness)
            self.normalized_fitness = (1 / self.fitness) / total_fitness

    def selection(self, selection_rate):
        self.members = np.choice(self.members, size=int(selection_rate * self.size), replace=False, p=self.fitness)
        self.size = len(self.members)

    def pair_parents(self):
        # parowanie wszystkich po dwa, może jakiś np.array,
        # co dla nieparzystych - jakiś losowy?
        # crossover dla każdej pary
        pass


class PopulationMember:
    def __init__(self, fitness, search_space):
        self.params = {name: param.get_value() for name, param in search_space.items()}
        self.fitness = None
        self.normalized_fitness = None

    def calculate_fitness(self, eval_function):
        fitness = eval_function(**self.params)
        return fitness

    def mutate(self, mutation_rate, min_genoms=1):
        size = max(min_genoms, int(mutation_rate * len(self.params)))
        chosen_params = np.random.choice(self.params.keys(), size=size, replace=False)

        for param in chosen_params:
            self.params[param] = self.params[param].get_value()

    def crossover(self, other, crossover_rate):
        # Uniform crossover
        if np.random.random() < crossover_rate:
            param_list = self.params.keys()
            points = np.random.randint(0, 1, size=len(param_list))

            self.params = {param: self.params[param] if add_to_self else other.params[param]
                           for add_to_self, param in zip(points, param_list)}
            other.params = {param: self.params[param] if not add_to_self else other.params[param]
                            for add_to_self, param in zip(points, param_list)}
