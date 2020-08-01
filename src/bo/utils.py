import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm


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