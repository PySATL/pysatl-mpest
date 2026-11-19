"""Objective functions shared by optimizer integration tests."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from numpy.typing import NDArray


def bounded_objective(params: NDArray[np.float64]) -> float:
    """Return a bounded objective with its minimum at ``mu=1`` and infinity for ``sigma <= 0``.

    Parameters
    ----------
    params : NDArray[np.float64]
        The two objective parameters, ``mu`` and ``sigma``.

    Returns
    -------
    float
        The objective value, or positive infinity when ``sigma`` is non-positive.
    """
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    return float((mu - 1) ** 2 + (sigma - 0.01) ** 2 + 1e-3 / sigma)
