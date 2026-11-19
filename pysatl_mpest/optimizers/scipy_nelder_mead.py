"""Nelder-Mead optimizer wrapper using SciPy."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

from .optimizer import Optimizer


class ScipyNelderMead(Optimizer):
    """Nelder-Mead simplex optimizer from SciPy.

    Derivative-free search method suitable for non-differentiable or noisy functions.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        minimize
    """

    def minimize(self, target: Callable, params: list[float]) -> list[float]:
        """Minimize target function using Nelder-Mead simplex algorithm.

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

        return list(np.asarray(minimize(target, params, method="Nelder-Mead").x, dtype=np.float64))
