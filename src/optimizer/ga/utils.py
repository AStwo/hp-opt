import numpy as np


def mutation_uniform(param, value, i, i_max, round_values=False):
    if np.random.choice([0, 1]):
        r = np.random.uniform(high=param.max-value)
    else:
        r = -np.random.uniform(high=value-param.min)

    if round_values:
        return round(value + r * np.random.rand() * (1 - i / i_max) ** 1)
    else:
        return value + r * np.random.rand() * (1 - i / i_max) ** 1


def mutation_choice(param, value):
    try:
        return np.random.choice(param.values[param.values != value])
    except ValueError:
        # When len(self.values) == 1
        return value


def crossover_uniform(param, parent_1_value, parent_2_value, alpha=0.5, round_values=False):
    interval_length = (1 + 2 * alpha) * abs(parent_1_value - parent_2_value)
    interval_midpoint = (parent_1_value + parent_2_value) / 2

    lower_bound = max(param.min, interval_midpoint - interval_length / 2)
    upper_bound = min(param.max, interval_midpoint + interval_length / 2)

    parent_1_new_value = np.random.uniform(lower_bound, upper_bound)
    parent_2_new_value = np.random.uniform(lower_bound, upper_bound)

    if round_values:
        return round(parent_1_new_value), round(parent_2_new_value)
    else:
        return parent_1_new_value, parent_2_new_value


def crossover_choice(parent_1_value, parent_2_value):
    return tuple(np.random.choice([parent_1_value, parent_2_value], 2, replace=True))
