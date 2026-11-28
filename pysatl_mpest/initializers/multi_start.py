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
    """Runs a nested initializer several times and chooses the start with the largest log-likelihood.

    A random initializer can give a poor starting mixture or fail on an unlucky partition of the sample. The
    multistart makes ``n_init`` independent runs of the nested initializer, skips unlucky runs and returns the
    successful starting mixture with the largest log-likelihood on the sample. If several runs reach the same
    largest value, the earliest of them wins.

    The generator of run ``i`` is child ``i`` of the generator passed by :meth:`initialize`; all children are
    created by ``rng.spawn(n_init)``. Therefore an integer seed reproduces the whole multistart, and run ``i``
    can be reproduced separately by ``numpy.random.default_rng(seed).spawn(i + 1)[i]``. A passed generator is
    shifted only in its counter of spawned children, not in its random stream: numbers taken from it after the
    multistart are the same as without the multistart. Taking numbers from the generator before the multistart
    does not change its children either: such a generator gives the same multistart as a fresh generator with
    the same seed.

    Even with ``n_init = 1`` the result is not the same as a direct call of the nested initializer with the same
    seed: the run receives the first child of the generator rather than that generator itself.

    A run that raises :class:`~.InitializationError`, including its subclass
    :class:`~.ComponentEstimationError`, is unlucky and is skipped. If some runs succeed, one
    ``RuntimeWarning`` reports the number of failed runs and the last reason. If every run fails,
    :class:`~.InitializationError` is raised with the last reason as its cause. Any other exception is an error
    of the input or the implementation and passes through immediately: repeating such a run cannot help.

    Parameters
    ----------
    initializer : Initializer
        The nested initializer. Its public :meth:`initialize` method is called in every run, so its input and
        result checks apply every time.
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
            The successful starting mixture with the largest log-likelihood on the sample, returned as the same
            object that its run returned. If several runs reach the same largest value, the earliest of them wins.

        Raises
        ------
        InitializationError
            If every run failed. The message is ``"All {n_init} runs failed; the last reason: {reason}"``, and
            the last error of a run is available as ``__cause__``. Any exception that is not an initialization
            error passes through unchanged.

        Warns
        -----
        RuntimeWarning
            If some but not all runs failed. The message is
            ``"{m} of {n_init} runs failed; the last reason: {reason}"``.
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
