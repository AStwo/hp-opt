from functools import wraps
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm

from src.space.SearchSpace import UniformInt, Choice


def transform_input(arr, search_space):
    x = arr.copy()
    choice_encodings = {}

    for col_idx, param in enumerate(search_space.values()):
        if isinstance(param, UniformInt):
            x[:, col_idx] = x[:, col_idx].round()
        elif isinstance(param, Choice):
            x[:, col_idx] = x[:, col_idx].round()
            one_hot_encoding = np.zeros((x.shape[0], len(param.values)))
            one_hot_encoding[np.arange(x.shape[0]), x[:, col_idx].astype(int)] = 1

            choice_encodings[col_idx] = one_hot_encoding

    idx_offset = 0
    for col_idx in choice_encodings.keys():
        x = np.delete(x, col_idx + idx_offset, axis=1)
        x = np.insert(x, col_idx + idx_offset, choice_encodings[col_idx].T, axis=1)
        idx_offset += choice_encodings[col_idx].shape[1] - 1

    return x


def transform_kernel_input(kernel, search_space):
    @wraps(kernel)
    def wrapper(x1, x2, *args, **kwargs):
        tr_x1 = transform_input(x1, search_space)
        tr_x2 = transform_input(x2, search_space)
        return kernel(tr_x1, tr_x2, *args, **kwargs)

    return wrapper


def kernel_rbf(x1, x2, length=1.0, sigma_f=1.0, sigma_y=1e-8, add_noise=False):
    # Validate input
    if len(x1.shape) == 1:
        x1 = x1.reshape(1, -1)
    if len(x2.shape) == 1:
        x2 = x2.reshape(1, -1)

    dist = cdist(x1 / length, x2 / length, metric='sqeuclidean')

    k = sigma_f**2 * np.exp(-0.5 * dist)
    if add_noise:
        k += sigma_y**2 * np.eye(k.shape[0])

    return k


def kernel_matern(x1, x2, length=1.0, sigma_f=1.0, sigma_y=1e-8, add_noise=False):
    # Validate input
    if len(x1.shape) == 1:
        x1 = x1.reshape(1, -1)
    if len(x2.shape) == 1:
        x2 = x2.reshape(1, -1)

    dist = cdist(x1 / length, x2 / length, metric='sqeuclidean')
    k = sigma_f**2 * (1 + np.sqrt(5*dist) + 5/3 * dist) * np.exp(-np.sqrt(5*dist))
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
