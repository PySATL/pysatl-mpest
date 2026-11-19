"""Powell optimizer wrapper using SciPy."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

from .optimizer import Optimizer


class ScipyPowell(Optimizer):
    """Powell's conjugate direction optimizer from SciPy.

    Derivative-free search method suitable for functions without available gradients.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        minimize
    """

    def minimize(self, target: Callable, params: list[float]) -> list[float]:
        """Minimize target function using Powell's method.

        NumPy invalid-value warnings are suppressed during optimization.

        Parameters
        ----------
        target : Callable
            Objective function accepting parameter values and returning a scalar.
        params : list[float]
            Initial parameter guess.

        Returns
        -------
        list[float]
            Optimized parameter values.
        """

        with np.errstate(invalid="ignore"):
            result = minimize(target, params, method="Powell")
        return list(np.asarray(result.x, dtype=np.float64))
