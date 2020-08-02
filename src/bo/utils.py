import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm


def kernel_rbf(x1, x2, add_noise=False, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    # Validate input
    if len(x1.shape) == 1:
        x1 = x1.reshape(1, -1)
    if len(x2.shape) == 1:
        x2 = x2.reshape(1, -1)

    dist_2 = cdist(x1, x2) ** 2
    k = sigma_f**2 * np.exp(-0.5 * dist_2 / l**2)
    if add_noise:
        k += sigma_y**2 * np.eye(k.shape[0])

    return k


def kernel_matern(x1, x2, add_noise=False, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    # Validate input
    if len(x1.shape) == 1:
        x1 = x1.reshape(1, -1)
    if len(x2.shape) == 1:
        x2 = x2.reshape(1, -1)

    dist = cdist(x1, x2)
    k = sigma_f**2 * (1 + np.sqrt(5)/l * dist + 5/(3*l)*dist**2) * np.exp(-np.sqrt(5)/l * dist)
    if add_noise:
        k += sigma_y**2 * np.eye(k.shape[0])

    return k


def mean_const(X):
    return np.zeros((X.shape[0], 1))


def expected_improvement(X, gpr, f_best):
    mu, std = gpr.predict(X)

    z = np.divide(mu - f_best, std, out=np.zeros_like(mu), where=(std != 0))
    ei = (mu - f_best) * norm.cdf(z) + std * norm.pdf(z)

    return ei