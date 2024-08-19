import numpy as np


def get_ols_from_covariance(cov: np.array):
    ndims = cov.shape[0]
    XtX = cov[: ndims - 1, : ndims - 1]
    Xty = cov[: ndims - 1, ndims - 1]
    return np.linalg.solve(XtX, Xty)
