import numpy as np


class Uniform:
    def __init__(self, name, min_value, max_value):
        self.name = name
        self.min = min_value
        self.max = max_value

    def get_value(self):
        return np.random.uniform(self.min, self.max)

    def mutate_param(self, value):
        return max(min(value * (1 + np.random.uniform(-0.2, 0.2)), self.max), self.min)


class UniformInt:
    def __init__(self, name, min_value, max_value):
        self.name = name
        self.min = min_value
        self.max = max_value

    def get_value(self):
        return np.random.randint(self.min, self.max)

    def mutate_param(self, value):
        mutated_value = round(value * np.random.uniform(0, 0.2) * np.random.choice([-1, 1]))
        return max(min(mutated_value, self.max), self.min)


