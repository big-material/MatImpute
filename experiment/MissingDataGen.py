# stdlib
from typing import List

# third party
import numpy as np
from scipy import optimize
from scipy.special import expit


def pick_coeffs(
        X: np.ndarray,
        idxs_obs: List[int] = [],
        idxs_nas: List[int] = [],
        self_mask: bool = False,
) -> np.ndarray:
    n, d = X.shape
    if self_mask:
        coeffs = np.random.rand(d)
        Wx = X * coeffs
        coeffs /= np.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = np.random.rand(d_obs, d_na)
        Wx = X[:, idxs_obs] @ coeffs
        coeffs /= np.std(Wx, 0, keepdims=True)
    return coeffs


def fit_intercepts(
        X: np.ndarray, coeffs: np.ndarray, p: float, self_mask: bool = False
) -> np.ndarray:
    if self_mask:
        d = len(coeffs)
        intercepts = np.zeros(d)
        for j in range(d):
            def f(x: np.ndarray) -> np.ndarray:
                return expit(X * coeffs[j] + x).mean().item() - p

            intercepts[j] = optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = np.zeros(d_na)
        for j in range(d_na):
            def f(x: np.ndarray) -> np.ndarray:
                return expit(np.dot(X, coeffs[:, j]) + x).mean().item() - p

            intercepts[j] = optimize.bisect(f, -50, 50)
    return intercepts


def MAR_mask(
        X: np.ndarray,
        p: float,
        p_obs: float,
        sample_columns: bool = True,
) -> np.ndarray:
    """
    Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.

    Args:
        X : Data for which missing values will be simulated.
        p : Proportion of missing values to generate for variables which will have missing values.
        p_obs : Proportion of variables with *no* missing values that will be used for the logistic masking model.

    Returns:
        mask : Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    mask = np.zeros((n, d)).astype(bool)

    d_obs = max(
        int(p_obs * d), 1
    )  # number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs  # number of variables that will have missing values

    # Sample variables that will all be observed, and those with missing values:
    if sample_columns:
        idxs_obs = np.random.choice(d, d_obs, replace=False)
    else:
        idxs_obs = list(range(d_obs))

    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    # Other variables will have NA proportions that depend on those observed variables, through a logistic model
    # The parameters of this logistic model are random.

    # Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    print(coeffs.shape)
    # Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)
    print(intercepts)

    ps = expit(X[:, idxs_obs] @ coeffs + intercepts)
    print(ps)

    ber = np.random.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask


if __name__ == '__main__':
    X = np.random.rand(10, 10) * 100
    print(X)
    idxs_obs = [0, 1, 2]
    idxs_nas = [3, 4, 5, 6, 7, 8, 9]
    p = 0.3
    p_obs = 0.7
    mask = MAR_mask(X, p, p_obs)
    print(mask)
