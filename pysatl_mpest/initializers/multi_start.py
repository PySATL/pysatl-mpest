"""Multistart wrapper of an initializer."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import warnings
from collections.abc import Sequence

import numpy as np
from pysatl_core.families import ParametricFamily

from ..core import MixtureModel
from ..typings import UnivariateFloatArray
from .exceptions import InitializationError
from .initializer import Initializer


class MultiStartInitializer(Initializer):
    """Runs another initializer several times and returns the start with the largest log-likelihood.

    Run ``i`` gets child ``i`` of ``rng.spawn(n_init)``: an integer seed reproduces the whole multistart, and
    ``numpy.random.default_rng(seed).spawn(i + 1)[i]`` reproduces one run. A passed generator advances only its
    counter of children, not its stream of numbers. Ties go to the earlier run.

    A run that raises :class:`~.InitializationError` is skipped. If some runs fail, a ``RuntimeWarning`` reports
    how many and the last reason; if all fail, :class:`~.InitializationError` is raised with the last reason as
    its cause. Other exceptions pass through at once.

    Parameters
    ----------
    initializer : Initializer
        The initializer to repeat; its checks of the input and the result apply in every run.
    n_init : int | numpy.integer
        The positive number of independent runs.

    Attributes
    ----------
    initializer : Initializer
        The nested initializer, the same object that was passed to the constructor.
    n_init : int
        The number of runs as ``int``.

    Raises
    ------
    TypeError
        If ``initializer`` is not an :class:`~.Initializer`.
    ValueError
        If ``n_init`` is not a positive integer.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        initialize
    """

    initializer: Initializer
    n_init: int

    def __init__(self, initializer: Initializer, n_init: int | np.integer) -> None:
        if not isinstance(initializer, Initializer):
            raise TypeError(f"initializer must be an Initializer, got {type(initializer).__name__}.")
        if isinstance(n_init, bool) or not isinstance(n_init, (int, np.integer)) or n_init < 1:
            raise ValueError(f"n_init must be a positive integer, got {n_init!r}.")

        self.initializer = initializer
        self.n_init = int(n_init)

    def _initialize(
        self, X: UnivariateFloatArray, families: Sequence[ParametricFamily], rng: np.random.Generator
    ) -> MixtureModel:
        """Run the nested initializer for every child generator and return the best mixture.

        Parameters
        ----------
        X : UnivariateFloatArray
            The checked sample.
        families : Sequence[ParametricFamily]
            One family per component of the mixture.
        rng : numpy.random.Generator
            The parent generator of the independent run generators.

        Returns
        -------
        MixtureModel
            The best successful mixture, the same object that its run returned.

        Raises
        ------
        InitializationError
            ``"All {n_init} runs failed; the last reason: {reason}"`` if every run failed; the last error is the
            ``__cause__``.

        Warns
        -----
        RuntimeWarning
            ``"{m} of {n_init} runs failed; the last reason: {reason}"`` if some runs failed.
        """

        children = rng.spawn(self.n_init)
        best: MixtureModel | None = None
        best_score = -np.inf
        failed = 0
        last: InitializationError | None = None

        for child in children:
            try:
                mixture = self.initializer.initialize(X, families, random_state=child)
            except InitializationError as error:
                failed += 1
                last = error
                continue

            score = mixture.loglikelihood(X)
            if score > best_score:
                best_score = score
                best = mixture

        if best is None:
            raise InitializationError(f"All {self.n_init} runs failed; the last reason: {last}") from last
        if last is not None:
            warnings.warn(f"{failed} of {self.n_init} runs failed; the last reason: {last}", RuntimeWarning)
        return best
