"""
Contract of a component estimator: how the parameters of one component are estimated from a cluster.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

from pysatl_core.families import ParametricFamily

from ....typings import UnivariateFloatArray


class ComponentEstimator(ABC):
    """Estimates the free parameters of one component from the points of a cluster.

    It is not an estimator of the whole mixture such as :class:`~pysatl_mpest.estimators.EM`,
    :class:`~pysatl_mpest.estimators.ECM` or :class:`~pysatl_mpest.estimators.ECME`: it only computes numbers.
    The calling code builds the component from them and checks it in the same way for every estimator (keys of
    the result, enough data, finiteness, constraints of the parametrization, numerical degeneracy), so an
    estimator does not repeat these checks.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        supports
        estimate
    """

    @abstractmethod
    def supports(self, family: ParametricFamily) -> bool:
        """Whether the estimator can estimate the family in its parametrization.

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

    @abstractmethod
    def estimate(self, X: UnivariateFloatArray, h: UnivariateFloatArray, family: ParametricFamily) -> dict[str, float]:
        """Estimate the free parameters of a component of the family from one cluster.

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
            Values of the free parameters of the family only. Parameters fixed via ``family.view(...)``
            are never returned.

        Raises
        ------
        NotImplementedError
            If the family is not supported.
        ComponentEstimationError
            If the cluster has too little data for the formulas of the estimator.
        """
