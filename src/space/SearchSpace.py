import numpy as np


class Uniform:
    def __init__(self, name, min_value, max_value):
        self.name = name
        self.min = min_value
        self.max = max_value

    def get_value(self):
        return np.random.uniform(self.min, self.max)

    def mutate_param(self, value, i, i_max):
        a = 1
        # return max(min(value * (1 + np.random.uniform(-a, a)), self.max), self.min)
        if np.random.choice([0, 1]):
            r = np.random.uniform(self.max - value)
        else:
            r = -np.random.uniform(value - self.min)

        # return_val = value + np.random.uniform(-0.5, 0.5) * (self.max - self.min)
        # return max(min(return_val, self.max), self.min)

        return value + r * (1-np.random.rand()**((1-i/i_max)**1.5))
        # return value + r * (1-i/i_max)**5
        # return value + np.random.uniform(self.min, self.max) # * (1 - i/i_max)**2


    # def mutate_param(self, value, i, i_max):
    #     a = 0.5
    #     mutated_value = value + ((1-i/i_max)**2) * np.random.uniform(-a, a) * (self.max - self.min)
    #     return max(min(mutated_value, self.max), self.min)

    def cross_param(self, self_value, other_value, alpha=0.5):
        interval_length = (1 + 2 * alpha) * abs(self_value - other_value)
        interval_midpoint = (self_value + other_value) / 2

        lower_bound = max(self.min, interval_midpoint - interval_length / 2)
        upper_bound = min(self.max, interval_midpoint + interval_length / 2)

        self_value = np.random.uniform(lower_bound, upper_bound)
        other_value = np.random.uniform(lower_bound, upper_bound)

        return self_value, other_value


class UniformInt:
    def __init__(self, name, min_value, max_value):
        self.name = name
        self.min = min_value
        self.max = max_value

    def get_value(self):
        return np.random.randint(self.min, self.max)

    def mutate_param(self, value, i, i_max):
        mutated_value = round(value * np.random.uniform(0, 0.2) * np.random.choice([-1, 1]))
        return max(min(mutated_value, self.max), self.min)

    def cross_param(self, self_value, other_value, alpha=0.5):
        interval_length = (1 + 2 * alpha) * abs(self_value - other_value)
        interval_midpoint = (self_value + other_value) / 2

        lower_bound = round(max(self.min, interval_midpoint - interval_length / 2))
        upper_bound = round(min(self.max, interval_midpoint + interval_length / 2)) + 1
        try:
            self_value = np.random.randint(lower_bound, upper_bound)
        except ValueError:
            print(lower_bound, upper_bound)
        other_value = np.random.randint(lower_bound, upper_bound)

        return self_value, other_value


class Choice:
    def __init__(self, name, values):
        self.name = name
        self.values = np.array(values)

    def get_value(self):
        return np.random.choice(self.values)

    def mutate_param(self, value, i, i_max):
        try:
            return np.random.choice(self.values[self.values != value])
        except ValueError:
            # When len(self.values) == 1
            return value

    def cross_param(self, self_value, other_value):
        return tuple(np.random.choice([self_value, other_value], 2))

