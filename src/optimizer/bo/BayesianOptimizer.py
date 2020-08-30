import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from scipy.optimize import minimize
from numpy.linalg import det, inv

import src.optimizer.bo.utils as utils
from src.optimizer.opt.BaseOptimizer import BaseOptimizer
from src.space.SearchSpace import Choice


class BayesianOptimizer(BaseOptimizer):
    def __init__(self, search_space, gpr_kernel="matern", seed=None):
        super().__init__(search_space, seed)

        self.acquisition_fun = utils.expected_improvement
        # Initialize gaussian process
        self.gpr = GaussianRegressor(search_space, kernel=gpr_kernel)

    def optimize(self, eval_function, iterations, metric_target=None, early_stop=None, objective="min", starting_points=3):
        assert objective in ("min", "max")

        # Prepare variables
        early_stop_counter = 0
        i = 0
        sign = 1 if objective == "max" else -1
        if metric_target is not None:
            metric_target *= sign

        # Initialize starting points
        X = np.array([self.get_random_point(self.search_space) for _ in range(starting_points)])
        y = np.array([sign * eval_function(**utils.reverse_transform(solution, self.search_space)) for solution in X])
        self.best_solution = utils.reverse_transform(X[np.argmax(y)], self.search_space)
        self.best_target = y[np.argmax(y)]

        self.hist_params.append(self.best_solution)
        self.hist_target.append(sign * self.best_target)

        # Prepare param grid
        param_bounds = self.get_param_bounds(self.search_space)

        while True:
            self.gpr.fit(X, y)

            proposed_solution = self.optimize_acquisition(self.acquisition_fun, self.gpr, self.best_target, self.search_space, param_bounds)
            proposed_target = sign * eval_function(**utils.reverse_transform(proposed_solution, self.search_space))

            self.hist_params.append(utils.reverse_transform(proposed_solution, self.search_space))
            self.hist_target.append(sign * proposed_target)
            X = np.vstack((X, proposed_solution))
            y = np.append(y, proposed_target)

            # Update best target
            if proposed_target >= self.best_target:
                self.best_target = proposed_target
                self.best_solution = utils.reverse_transform(proposed_solution, self.search_space)
                early_stop_counter = 0

            if self.optimization_stop_conditions(i, iterations, early_stop, early_stop_counter,
                                                 self.hist_target, metric_target):
                self.best_target *= sign
                break
            else:
                early_stop_counter += 1
                i += 1

    @staticmethod
    def get_random_point(search_space):
        return np.array([param.get_random_index() if isinstance(param, Choice)
                         else param.get_value() for param in search_space.values()])

    @staticmethod
    def get_param_bounds(search_space):
        return tuple((param.min, param.max) for param in search_space.values())

    @staticmethod
    def optimize_acquisition(acquisition, gpr, f_best, search_space, bounds, runs=20):
        def obj(x):
            x = x.reshape(1, -1)
            return -acquisition(x, gpr, f_best)

        def find_min():
            x0 = BayesianOptimizer.get_random_point(search_space)
            return minimize(obj, x0, method="L-BFGS-B", bounds=bounds)

        res = Parallel(n_jobs=-2)(delayed(find_min)() for i in range(runs))
        hist_params = [r.x for r in res]
        hist_target = [r.fun for r in res]

        best_params = utils.transform_input(hist_params[np.argmin(hist_target)].reshape(1, -1), search_space)[0]
        best_params = utils.reverse_one_hot_encoding(best_params, search_space)

        return best_params


class GaussianRegressor:
    def __init__(self, search_space, kernel="rbf", mean=utils.mean_const, noise=True):
        self.mean = mean
        self.noise = noise
        self.search_space = search_space

        # Kernel
        kernels = {"rbf": utils.kernel_rbf, "matern": utils.kernel_matern}
        self.kernel = utils.transform_kernel_input(kernels[kernel], self.search_space)
        self.kernel_params = {"length": [1.0], "sigma_f": 1.0, "sigma_y": 0.1}
        self.kernel_bounds = ((1e-5, 1e+3), (1e-5, 1e+2), (1e-5, None))

        # Args set during fit
        self.var = None
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y.reshape(-1, 1)

        if len(self.kernel_params["length"]) == 1:  # Reshape length parameters to number of features
            self.kernel_params["length"] *= self.X.shape[1]
        self.optimize()
        self.var = self.kernel(X, X, add_noise=self.noise, **self.kernel_params)

    def predict(self, X, conf=0.975):
        var_x_pred = self.kernel(X, X, **self.kernel_params)
        cov_x = self.kernel(X, self.X, **self.kernel_params)

        mu = cov_x.dot(inv(self.var)).dot(self.y - self.mean(self.X)) + self.mean(X)
        cov = var_x_pred - cov_x.dot(inv(self.var)).dot(cov_x.T)

        return mu.ravel(), norm.ppf(conf) * np.sqrt(np.diag(cov).clip(0))

    def optimize(self, runs=15):
        keys = list(self.kernel_params.keys())

        def log_likelihood(params):
            dict_params = self.prepare_param_dict(keys, params)

            var = self.kernel(self.X, self.X, **dict_params, add_noise=True)
            # Safeguard against singular matrix
            if not (det_var := det(var)):
                return np.array([[1e+6]])

            return 0.5 * (self.y.T.dot(inv(var)).dot(self.y) + np.log(det_var) + len(self.y)*np.log(2*np.pi))

        def find_min():
            x0 = [np.random.exponential(value) for value in self.kernel_params.values()]
            x0 = np.array([*x0[0], *x0[1:]])  # Flatten array
            if len(x0) != len(bounds):
                x0 = self.correct_lengths_for_one_hot_encoding(x0, self.search_space)
            return minimize(log_likelihood, x0, method="L-BFGS-B", bounds=bounds)

        # Reshape bounds for length param
        length_param_bounds = (self.kernel_bounds[0], ) * (utils.transform_input(self.X, self.search_space).shape[1])
        bounds = (*length_param_bounds, *self.kernel_bounds[1:])  # Flatten array

        res = Parallel(n_jobs=-2)(delayed(find_min)() for i in range(runs))

        hist_params = [r.x for r in res]
        hist_target = [r.fun for r in res]

        best_params = hist_params[np.argmin(hist_target)]
        dict_best_params = self.prepare_param_dict(keys, best_params)
        self.kernel_params.update(dict_best_params)

    @staticmethod
    def prepare_param_dict(keys, params):
        """Update params with kernel length."""
        length = params[:len(params)-len(keys)+1]
        dict_params = dict(zip(keys, [length, *params[-2:]]))

        return dict_params

    @staticmethod
    def correct_lengths_for_one_hot_encoding(arr, search_space):
        offset = 0
        for i, param in enumerate(search_space.values()):
            if isinstance(param, Choice):
                arr = np.insert(arr, i + offset, np.repeat(arr[i + offset], len(param.values) - 1))
                offset += len(param.values) - 1

        return arr

