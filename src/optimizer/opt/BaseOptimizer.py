import matplotlib.pyplot as plt
import numpy as np


class BaseOptimizer:
    def __init__(self, search_space, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.search_space = search_space

        # Optimization
        self.hist_params = []
        self.hist_target = []
        self.best_solution = None
        self.best_target = None

    def optimize(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def optimization_stop_conditions(i, iterations, early_stop, early_stop_counter, hist_target, metric_target):
        if i >= iterations:
            print("Optimization stopped reaching maximum number of iterations")
            return True
        if early_stop is not None and early_stop_counter >= early_stop:
            print("Early stop criteria met after {} iterations".format(i + 1))
            return True
        if metric_target is not None and hist_target[-1] >= metric_target:
            print("Metric target reached after {} iterations".format(i + 1))
            return True
        return False

    def plot_solution_history(self):
        plt.scatter(x=range(len(self.hist_target)), y=self.hist_target, s=10)
        plt.scatter(x=np.where(np.array(self.hist_target) == self.best_target)[0][0], y=self.best_target, color="red")
