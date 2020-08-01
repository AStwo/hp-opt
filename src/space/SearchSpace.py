import numpy as np


class Uniform:
    def __init__(self, name, min_value, max_value):
        self.name = name
        self.min = float(min_value)
        self.max = float(max_value)

    def get_value(self):
        return np.random.uniform(self.min, self.max)


class UniformInt:
    def __init__(self, name, min_value, max_value):
        self.name = name
        self.min = min_value
        self.max = max_value

    def get_value(self):
        return np.random.randint(self.min, self.max)


class Choice:
    def __init__(self, name, values):
        self.name = name
        self.values = np.array(values)

    def get_value(self):
        return np.random.choice(self.values)
