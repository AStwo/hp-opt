import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from numpy.linalg import det, inv


class BayesianOptimizer:
    def __init__(self, search_space, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.search_space = search_space

    def optimize(self):
        pass


def kernel_rbf(x1, x2, noise=False, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    # Validate input
    if len(x1.shape) == 1:
        x1 = x1.reshape(1, -1)
    if len(x2.shape) == 1:
        x2 = x2.reshape(1, -1)

    dist_2 = cdist(x1, x2) ** 2
    matrix = sigma_f**2 * np.exp(-0.5 / l**2 * dist_2)
    if noise:
        matrix += sigma_y**2 * np.eye(matrix.shape[0])

    return matrix


def mean_const(X):
    return np.zeros((X.shape[0], 1))


def expected_improvement(X, gpr, f_best):
    mu, std = gpr.predict(X)

    z = np.divide(mu - f_best, std, out=np.zeros_like(mu), where=(std != 0))
    ei = (mu - f_best) * norm.cdf(z) + std * norm.pdf(z)

    return ei


def optimize_acquisition(acquisition, gpr, f_best, bounds, runs=10):
    def obj(x):
        x = x.reshape(1, -1)
        return -acquisition(x, gpr, f_best)

    res_params = []
    res_target = []
    for i in range(runs):
        # x0 = np.array([[np.random.uniform(-3, 3, 2)]]) # todo: get random point
        # x0 = np.random.uniform(-3, 3, 2).reshape(-1, 2)
        x0 = np.array([[np.random.uniform(-3, 3)]])
        res = minimize(obj, x0=x0, method="L-BFGS-B", bounds=bounds)
        res_params.append(res.x)
        res_target.append(res.fun)

    best_params = res_params[np.argmin(res_target)]
    return best_params


class GaussianRegressor:
    def __init__(self, kernel, mean=mean_const, noise=True, **kwargs):
        self.kernel = kernel
        self.kernel_params = {"l": 1.0, "sigma_f": 1.0, "sigma_y": 0.1}
        self.kernel_params.update(kwargs)
        self.mean = mean
        self.noise = noise

        # Args set during fit
        self.var = None
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y.reshape(-1, 1)
        self.optimize()
        self.var = self.kernel(X, X, noise=self.noise, **self.kernel_params)

    def predict(self, X, conf=0.975):
        var_x_pred = self.kernel(X, X, **self.kernel_params)
        cov_x = self.kernel(X, self.X, **self.kernel_params)

        mu = cov_x.dot(inv(self.var)).dot(self.y - self.mean(self.X)) + self.mean(X)
        cov = var_x_pred - cov_x.dot(inv(self.var)).dot(cov_x.T)

        return mu.ravel(), norm.ppf(conf) * np.sqrt(np.diag(cov).clip(0))

    def optimize(self, runs=10):
        keys = list(self.kernel_params.keys())
        
        def log_likelihood(params):
            dict_params = dict(zip(keys, params))
            var = self.kernel(self.X, self.X, noise=True, **dict_params)

            # Safeguard against singular matrix
            if not (det_var := det(var)):
                return np.array([[1e+6]])

            return 0.5 * (self.y.T.dot(inv(var)).dot(self.y) + np.log(det_var) + len(self.y)*np.log(2*np.pi))

        start_params = np.fromiter(self.kernel_params.values(), dtype=float)

        bounds = ((1e-1, None), (1e-1, None), (1e-1, None))

        res_params = []
        res_target = []
        for i in range(runs):
            res = minimize(log_likelihood, start_params, bounds=bounds, method="L-BFGS-B")
            res_params.append(res.x)
            res_target.append(res.fun[0][0])

        best_params = res_params[np.argmin(res_target)]

        self.kernel_params.update(dict(zip(keys, best_params)))
