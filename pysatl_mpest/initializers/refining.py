"""Initializer wrapper that refines the starting mixture with an estimator."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from pysatl_core.families import ParametricFamily

from ..core import MixtureModel
from ..estimators import BaseEstimator
from ..typings import UnivariateFloatArray
from .exceptions import InitializationError
from .initializer import Initializer


class RefiningInitializer(Initializer):
    """Refines the start of another initializer with a mixture estimator.

    The start of ``initializer`` goes to ``estimator.fit(X, start, **fit_kwargs)``. The names in ``fit_kwargs`` are
    checked against the signature of ``fit`` at creation, their values are not. Only the nested initializer gets
    the generator.

    If the estimator returns fewer components than families (a pruner removed one), the run fails with
    :class:`~.InitializationError`, so a multistart skips it.

    Parameters
    ----------
    initializer : Initializer
        The initializer that builds the starting mixture.
    estimator : BaseEstimator
        The estimator that refines the start.
    fit_kwargs : Mapping[str, Any] | None, optional
        Keyword arguments of every ``fit`` call after the sample and the mixture.

    Attributes
    ----------
    initializer : Initializer
        The nested initializer, the same object that was passed to the constructor.
    estimator : BaseEstimator
        The estimator, the same object that was passed to the constructor.
    fit_kwargs : dict[str, Any]
        A copy of the passed mapping; the values are shared.

    Raises
    ------
    TypeError
        If a role has the wrong type, or ``fit_kwargs`` is not a mapping or does not fit the signature of ``fit``
        (the error of the signature is the ``__cause__``).

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        initialize
    """

    initializer: Initializer
    estimator: BaseEstimator
    fit_kwargs: dict[str, Any]

    def __init__(
        self,
        initializer: Initializer,
        estimator: BaseEstimator,
        *,
        fit_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(initializer, Initializer):
            raise TypeError(f"initializer must be an Initializer, got {type(initializer).__name__}.")
        if not isinstance(estimator, BaseEstimator):
            raise TypeError(f"estimator must be a BaseEstimator, got {type(estimator).__name__}.")
        if fit_kwargs is not None and not isinstance(fit_kwargs, Mapping):
            raise TypeError(f"fit_kwargs must be a mapping or None, got {type(fit_kwargs).__name__}.")

        self.initializer = initializer
        self.estimator = estimator
        self.fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)

        try:
            inspect.signature(estimator.fit).bind(None, None, **self.fit_kwargs)
        except TypeError as error:
            raise TypeError(
                f"fit_kwargs do not match the signature of {type(estimator).__name__}.fit: {error}."
            ) from error

    def _initialize(
        self, X: UnivariateFloatArray, families: Sequence[ParametricFamily], rng: np.random.Generator
    ) -> MixtureModel:
        """Build the start with the nested initializer and refine it with the estimator.

        Parameters
        ----------
        X : UnivariateFloatArray
            The checked sample.
        families : Sequence[ParametricFamily]
            One family per component of the mixture.
        rng : numpy.random.Generator
            The generator passed to the nested initializer.

        Returns
        -------
        MixtureModel
            The fitted mixture, the same object that ``fit`` returned.

        Raises
        ------
        InitializationError
            ``"The estimator returned {m} of {k} components."`` if the estimator removed a component.
            Errors of the nested initializer and the estimator pass through unchanged.
        """

        start = self.initializer.initialize(X, families, random_state=rng)
        fitted = self.estimator.fit(X, start, **self.fit_kwargs)

        k = len(families)
        if fitted.n_components < k:
            raise InitializationError(f"The estimator returned {fitted.n_components} of {k} components.")
        return fitted
