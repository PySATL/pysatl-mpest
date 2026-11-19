"""Abstract base class for mixture model parameter estimators."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

from numpy.typing import ArrayLike

from ..core import MixtureModel


class BaseEstimator(ABC):
    """Abstract base class for mixture model parameter estimators.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        fit
    """

    @abstractmethod
    def fit(self, X: ArrayLike, mixture: MixtureModel) -> MixtureModel:
        """Fit mixture model parameters to input data.

        Parameters
        ----------
        X : ArrayLike
            Input data sample.
        mixture : MixtureModel
            Initial mixture model to fit; not modified in-place.

        Returns
        -------
        MixtureModel
            Fitted mixture model with estimated parameters.
        """
