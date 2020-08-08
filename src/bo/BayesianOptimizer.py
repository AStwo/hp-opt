import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from scipy.optimize import basinhopping
from numpy.linalg import det, inv

from src.bo.utils import kernel_rbf, kernel_matern, mean_const, expected_improvement, transform_kernel_input, \
    transform_input
from src.opt.BaseOptimizer import BaseOptimizer
from src.space.SearchSpace import Choice


class BayesianOptimizer(BaseOptimizer):
    def __init__(self, search_space, gpr_kernel="rbf", seed=None):
        super().__init__(search_space, seed)

        self.kernel = kernel_rbf
        self.mean = mean_const
        self.acquisition_fun = expected_improvement
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
        y = np.array([sign * eval_function(*self.transform_to_params(solution, self.search_space)) for solution in X])
        self.best_solution = X[np.argmax(y)]
        self.best_target = y[np.argmax(y)]

        self.hist_params.append(self.best_solution)
        self.hist_target.append(sign * self.best_target)

        # Prepare param grid
        param_bounds = self.get_param_bounds(self.search_space)

        while True:
            self.gpr.fit(X, y)

            proposed_solution = self.optimize_acquisition(self.acquisition_fun, self.gpr, self.best_target, self.search_space, param_bounds)
            proposed_target = sign * eval_function(*self.transform_to_params(proposed_solution, self.search_space))

            self.hist_params.append(proposed_solution)
            self.hist_target.append(sign * proposed_target)
            X = np.vstack((X, proposed_solution))
            y = np.append(y, proposed_target)

            # Update best target
            if proposed_target >= self.best_target:
                self.best_target = proposed_target
                self.best_solution = proposed_solution
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
    def optimize_acquisition(acquisition, gpr, f_best, search_space, bounds, runs=10):
        def obj(x):
            x = x.reshape(1, -1)
            return -acquisition(x, gpr, f_best)

        def find_min():
            x0 = BayesianOptimizer.get_random_point(search_space)
            return basinhopping(obj, x0, niter=runs, minimizer_kwargs=minimizer_kwargs)

        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
        res = Parallel(n_jobs=-2)(delayed(find_min)() for i in range(20))
        hist_params = [r.x for r in res]
        hist_target = [r.fun for r in res]

        best_params = transform_input(hist_params[np.argmin(hist_target)].reshape(1, -1), search_space)[0]
        for i, param in enumerate(search_space.values()):
            if isinstance(param, Choice):
                indexes = best_params[i:(i+len(param.values))]
                best_params[i] = np.argmax(indexes)
                best_params = np.delete(best_params, slice(i+1, i+len(param.values)))

        return best_params

    @staticmethod
    def transform_to_params(arr, search_space):
        x = [param.values[int(val)] if isinstance(param, Choice) else val for val, param in zip(arr, search_space.values())]
        return x


class GaussianRegressor:
    def __init__(self, search_space, kernel="rbf", mean=mean_const, noise=True):
        self.mean = mean
        self.noise = noise
        self.search_space = search_space

        # Kernel
        kernels = {"rbf": kernel_rbf, "matern": kernel_matern}
        self.kernel = transform_kernel_input(kernels[kernel], self.search_space)
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

    def optimize(self, runs=5):
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
                offset = 0
                for i, param in enumerate(self.search_space.values()):
                    if isinstance(param, Choice):
                        x0 = np.insert(x0, i + offset, np.repeat(x0[i + offset], len(param.values) - 1))
                        offset += len(param.values) - 1
            return basinhopping(log_likelihood, x0, niter=runs, minimizer_kwargs=minimizer_kwargs)

        # Reshape bounds for length param
        length_param_bounds = (self.kernel_bounds[0], ) * (transform_input(self.X, self.search_space).shape[1])
        bounds = (*length_param_bounds, *self.kernel_bounds[1:])  # Flatten array

        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
        res = Parallel(n_jobs=-2)(delayed(find_min)() for i in range(5))

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

