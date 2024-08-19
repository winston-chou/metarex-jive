from typing import List
import numpy as np


def get_ols_from_covariance(cov: np.array, metrics: List):
    XtX = cov[:len(metrics), :len(metrics)]
    Xty = cov[:len(metrics), len(metrics)]
    return np.linalg.solve(XtX, Xty)
