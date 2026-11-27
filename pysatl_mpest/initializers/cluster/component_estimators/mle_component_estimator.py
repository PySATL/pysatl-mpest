"""Closed-form maximum likelihood estimation of a component from a cluster.

The formulas live in a private registry keyed by the pair ``(family name, parametrization name)``.
They are registered by calling the function returned by ``register`` instead of using it as a decorator,
because ``mutmut`` does not mutate decorated functions.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Mapping

import numpy as np
from pysatl_core.families import ParametricFamily
from pysatl_core.types import FamilyName

from ....core import _components
from ....typings import UnivariateFloatArray
from ._dispatch import FormulaDispatch
from .base import ComponentEstimator

type Formula = Callable[[UnivariateFloatArray, UnivariateFloatArray, Mapping[str, float]], dict[str, float]]

_formulas: FormulaDispatch[Formula] = FormulaDispatch()


def _normal_mean_std(X: UnivariateFloatArray, h: UnivariateFloatArray, fixed: Mapping[str, float]) -> dict[str, float]:
    """Weighted mean and standard deviation; the deviation is taken around the fixed ``mu`` when it is fixed.

    A degenerate cluster gives ``sigma = 0.0``; this is the correct answer of the formula, not its error.
    The cluster is not empty.
    """

    free = {}
    if "mu" not in fixed:
        free["mu"] = float(np.average(X, weights=h))
    if "sigma" not in fixed:
        m = fixed["mu"] if "mu" in fixed else free["mu"]
        free["sigma"] = float(np.sqrt(np.average((X - m) ** 2, weights=h)))
    return free


def _exponential_rate(X: UnivariateFloatArray, h: UnivariateFloatArray, fixed: Mapping[str, float]) -> dict[str, float]:
    """Inverse weighted mean of the points of the cluster.

    A zero mean gives ``inf`` because the division is performed on numpy numbers before the conversion
    to ``float``. ``fixed`` is unused: the single parameter of the family cannot be fixed.
    """

    return {"lambda_": float(1.0 / np.average(X, weights=h))}


def _continuous_uniform_standard(
    X: UnivariateFloatArray, h: UnivariateFloatArray, fixed: Mapping[str, float]
) -> dict[str, float]:
    """Smallest and largest values of the points of the cluster.

    The cluster is not empty.
    """

    free = {}
    if "lower_bound" not in fixed:
        free["lower_bound"] = float(X[h > 0].min())
    if "upper_bound" not in fixed:
        free["upper_bound"] = float(X[h > 0].max())
    return free


_formulas.register(FamilyName.NORMAL, "meanStd")(_normal_mean_std)
_formulas.register(FamilyName.EXPONENTIAL, "rate")(_exponential_rate)
_formulas.register(FamilyName.CONTINUOUS_UNIFORM, "standard")(_continuous_uniform_standard)


class MLEComponentEstimator(ComponentEstimator):
    """Estimates the free parameters of a component by closed-form maximum likelihood formulas, without optimization.

    The estimator supports ``Normal`` in the ``meanStd`` parametrization, ``Exponential`` in the ``rate``
    parametrization and ``ContinuousUniform`` in the ``standard`` parametrization. Parameters fixed through
    ``family.view(...)`` are taken into account and are not returned. The data are not checked: degenerate
    values are rejected by the calling code.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        supports
        estimate
    """

    def supports(self, family: ParametricFamily) -> bool:
        """Whether a closed-form formula is registered for the family and its parametrization.

        Parameters fixed via ``family.view(...)`` do not affect the answer.

        Parameters
        ----------
        family : ParametricFamily
            The family to check.

        Returns
        -------
        bool
            Whether :meth:`estimate` accepts the family.
        """

        return _formulas.supports(family)

    def estimate(self, X: UnivariateFloatArray, h: UnivariateFloatArray, family: ParametricFamily) -> dict[str, float]:
        """Estimate the free parameters of a component of the family by its closed-form formula.

        Parameters
        ----------
        X : UnivariateFloatArray
            The whole sample.
        h : UnivariateFloatArray
            Column of the partition matrix as ``float64``: 1.0 for the points of the cluster and 0.0 for
            the others; the points of the cluster are ``X[h > 0]``.
        family : ParametricFamily
            The family of the component.

        Returns
        -------
        dict[str, float]
            Values of the free parameters of the family only.

        Raises
        ------
        NotImplementedError
            If no formula is registered for the family and parametrization.

        Notes
        -----
        The estimator does not raise :class:`~.ComponentEstimationError`: its formulas need no more data than
        the common lower bound, as many distinct values in the cluster as free parameters, which the calling
        code checks.
        """

        return _formulas.dispatch(family)(X, h, _components.family_fixed_parameters(family))
