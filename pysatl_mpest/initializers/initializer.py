"""Base class of initializers with the template method ``initialize``."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import final

import numpy as np
from numpy.typing import ArrayLike
from pysatl_core.families import ParametricFamily

from ..core import MixtureModel, _components
from ..core._validation import sample_coverage
from ..typings import UnivariateFloatArray, UnivariateIntArray
from .exceptions import InitializationError


class Initializer(ABC):
    """Builds a starting :class:`~pysatl_mpest.core.MixtureModel` from a sample and component families.

    The initializer receives one family per component and returns the mixture from which EM starts. The common
    template method :meth:`initialize` gives every implementation, including a user-provided one, the same
    guarantees: the input is checked, ``random_state`` is handled in one way, and the returned mixture is checked
    to be usable as the starting point of EM.

    To write an initializer, inherit this class and implement ``_initialize``; do not override
    :meth:`initialize`.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        initialize
    """

    @final
    def initialize(
        self,
        X: ArrayLike,
        families: Sequence[ParametricFamily],
        random_state: int | np.random.Generator | None = None,
    ) -> MixtureModel:
        """Build a starting mixture from the sample.

        The method checks the input, creates the random generator, calls ``_initialize``, checks its result
        and returns that mixture unchanged. The method is final and cannot be overridden.

        Parameters
        ----------
        X : ArrayLike
            The sample; it is converted to a one-dimensional ``float64`` array before the initializer step.
        families : Sequence[ParametricFamily]
            One family per component of the mixture. A parametrization is chosen and parameters are fixed through
            ``family.view(...)``.
        random_state : int | numpy.random.Generator | None, optional
            Source of randomness. An integer seed makes the same arguments produce the same mixture independently
            of previous calls. A generator is used in place and advances, so two calls with the same generator
            produce different mixtures. ``None`` uses fresh entropy.

        Returns
        -------
        MixtureModel
            The starting mixture; component ``j`` belongs to ``families[j]``.

        Raises
        ------
        ValueError
            If the input is invalid, or if the initializer step returned a number of components different from
            the number of families or components not ordered as ``families``. The latter means an error in the
            initializer implementation.
        TypeError
            If an element of ``families`` is not a parametric family.
        InitializationError
            If the log-likelihood of the starting mixture on the sample is not finite; the message names example
            points that are not covered or where the density is undefined.
        """

        sample = _validate_input(X, families)
        rng = np.random.default_rng(random_state)
        mixture = self._initialize(sample, families, rng)
        _validate_result(mixture, sample, families)
        return mixture

    @abstractmethod
    def _initialize(
        self, X: UnivariateFloatArray, families: Sequence[ParametricFamily], rng: np.random.Generator
    ) -> MixtureModel:
        """Build the starting mixture from the checked input.

        The sample is a one-dimensional ``float64`` array of finite values with at least as many points as there
        are families, ``families`` is the same object that was passed to :meth:`initialize`, and ``rng`` is a
        ready random generator. All randomness of the implementation must come from ``rng``.

        Parameters
        ----------
        X : UnivariateFloatArray
            The checked sample.
        families : Sequence[ParametricFamily]
            One family per component of the mixture.
        rng : numpy.random.Generator
            The only source of randomness of the implementation.

        Returns
        -------
        MixtureModel
            A mixture with exactly ``len(families)`` components, component ``j`` belonging to ``families[j]``.

        Raises
        ------
        InitializationError
            If this run of the initializer fails and another run may succeed.
        """


def _validate_input(X: ArrayLike, families: Sequence[ParametricFamily]) -> UnivariateFloatArray:
    """Check the sample and families, returning the sample as a one-dimensional ``float64`` array."""

    if len(families) == 0:
        raise ValueError("families must not be empty.")

    for j, family in enumerate(families):
        if not isinstance(family, ParametricFamily):
            raise TypeError(f"families[{j}] must be a ParametricFamily, got {type(family).__name__}.")

    sample = np.asarray(X, dtype=np.float64)
    if sample.ndim != 1:
        raise ValueError(f"X must be one-dimensional, got an array of shape {sample.shape}.")
    if not np.isfinite(sample).all():
        raise ValueError("X must contain only finite values.")

    n = len(sample)
    k = len(families)
    if n < k:
        raise ValueError(f"X has {n} points, fewer than the {k} families.")

    return sample


def _validate_result(mixture: MixtureModel, X: UnivariateFloatArray, families: Sequence[ParametricFamily]) -> None:
    """Check the composition and sample coverage of the mixture returned by the initializer step."""

    k = len(families)
    m = mixture.n_components
    if m != k:
        raise ValueError(f"The initializer returned {m} components for {k} families.")

    for j in range(k):
        expected = _components.family_key(families[j])
        received = _components.component_family_key(mixture[j])
        if received != expected:
            raise ValueError(
                f"Component {j} of the result is not from families[{j}]: "
                f"expected {_format_family_key(expected)}, got {_format_family_key(received)}."
            )

    loglikelihood = mixture.loglikelihood(X)
    if np.isfinite(loglikelihood):
        return

    coverage = sample_coverage(mixture, X)
    message = f"The log-likelihood of the starting mixture on the sample is {loglikelihood}."
    if len(coverage.uncovered) != 0:
        message += (
            f" {len(coverage.uncovered)} of {len(X)} sample points are not covered by any component, "
            f"for example {_point_examples(X, coverage.uncovered)}."
        )
    if len(coverage.undefined) != 0:
        message += (
            f" The density of the mixture is undefined at {len(coverage.undefined)} of {len(X)} sample points, "
            f"for example {_point_examples(X, coverage.undefined)}."
        )
    raise InitializationError(message)


def _format_family_key(key: _components.FamilyKey) -> str:
    """Human-readable form of a key produced by ``family_key`` and ``component_family_key``."""

    name, parametrization, fixed = key
    return f"family {name} in parametrization {parametrization} with fixed parameters {dict(fixed)}"


def _point_examples(X: UnivariateFloatArray, indices: UnivariateIntArray) -> str:
    """Values of the sample in the first three points with the given indices."""

    return ", ".join(str(float(X[index])) for index in indices[:3])
