"""Lazy component table with shared records of identical families and assembly of the starting mixture."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence

import numpy as np
from pysatl_core.families import ParametricFamily, ParametricFamilyDistribution

from ....core import MixtureModel, _components
from ....typings import MultivariateFloatArray, UnivariateFloatArray, UnivariateIntArray
from ...exceptions import ComponentEstimationError
from ..component_estimators.base import ComponentEstimator
from ..table import ComponentTable
from ._pair import PairEvaluation, evaluate_pair


class LazyComponentTable(ComponentTable):
    """Component table that evaluates a pair when it is first requested and remembers the result.

    A record is keyed by the family key and the cluster, not by the family index: identical families, whose
    keys are equal, share records. With ``d`` distinct families and ``k`` clusters there are at most
    ``d · k`` evaluations. The input is not checked because the initializer creates the table from a sample,
    a partition and families that it has already checked.

    Parameters
    ----------
    X : UnivariateFloatArray
        The whole sample. It is stored as the same object.
    H : MultivariateFloatArray
        Hard partition matrix of shape ``(n, k)`` without empty columns. It is stored as the same object.
    families : Sequence[ParametricFamily]
        The ``k`` families of the components, stored as a tuple.
    component_estimator : ComponentEstimator
        Estimator passed to the evaluation of every pair.

    Attributes
    ----------
    X : UnivariateFloatArray
        The whole sample, the same object that was passed to the constructor.
    H : MultivariateFloatArray
        The partition matrix, the same object that was passed to the constructor.
    families : tuple[ParametricFamily, ...]
        The families of the components.
    k : int
        Number of families, equal to the number of clusters.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        groups
        cluster_means
        loglikelihood
        failure
        component
        mixture
    """

    X: UnivariateFloatArray
    H: MultivariateFloatArray
    families: tuple[ParametricFamily, ...]

    def __init__(
        self,
        X: UnivariateFloatArray,
        H: MultivariateFloatArray,
        families: Sequence[ParametricFamily],
        component_estimator: ComponentEstimator,
    ) -> None:
        self.X = X
        self.H = H
        self.families = tuple(families)
        self._component_estimator = component_estimator
        self._scale = float(np.std(X))
        self._evaluations: dict[tuple[_components.FamilyKey, int], PairEvaluation] = {}

    @property
    def k(self) -> int:
        """int: Number of families, equal to the number of clusters."""

        return len(self.families)

    def groups(self) -> tuple[tuple[int, ...], ...]:
        """Groups of identical families as tuples of family indices.

        The groups are computed on every call; no pair is evaluated.
        """

        groups: dict[_components.FamilyKey, list[int]] = {}
        for j, family in enumerate(self.families):
            groups.setdefault(_components.family_key(family), []).append(j)
        return tuple(tuple(group) for group in groups.values())

    def cluster_means(self) -> UnivariateFloatArray:
        """Read-only means of the clusters, computed on every call; no pair is evaluated."""

        means = self.X @ self.H / self.H.sum(axis=0)
        means.setflags(write=False)
        return means

    def loglikelihood(self, j: int, c: int) -> float:
        """Log-likelihood of the pair; a non-finite value of the record is returned as ``-inf``.

        In particular, ``NaN`` and ``+inf`` never leave the table.
        """

        value = self._pair(j, c).loglikelihood
        if np.isfinite(value):
            return value
        return -np.inf

    def failure(self, j: int, c: int) -> str | None:
        """Why the log-likelihood of the pair is not finite, with the numbers prepended when read."""

        reason = self._pair(j, c).reason
        if reason is None:
            return None
        return f"family {j} ({self.families[j].name}), cluster {c}: {reason}"

    def component(self, j: int, c: int) -> ParametricFamilyDistribution:
        """Component of the pair, the same object at every call.

        Returns
        -------
        ParametricFamilyDistribution
            Component of the pair in the states "estimated" and "does not cover the cluster".

        Raises
        ------
        ComponentEstimationError
            If the pair is not estimated. Every call raises a new object whose text is ``failure(j, c)``.
        """

        component = self._pair(j, c).component
        if component is None:
            raise ComponentEstimationError(self.failure(j, c))
        return component

    def mixture(self, assignment: UnivariateIntArray) -> MixtureModel:
        """Assemble a mixture from the pairs selected by a permutation of clusters.

        Parameters
        ----------
        assignment : UnivariateIntArray
            Permutation of the clusters: family ``j`` gets cluster ``assignment[j]``. It is not checked.

        Returns
        -------
        MixtureModel
            Components in the order of the families; the weights are the fractions of the assigned clusters.

        Raises
        ------
        ComponentEstimationError
            If a selected pair is not estimated.
        """

        components = [self.component(j, assignment[j]) for j in range(self.k)]
        weights = self.H.mean(axis=0)[assignment]
        return MixtureModel(components, weights)

    def _pair(self, j: int, c: int) -> PairEvaluation:
        """Record of the pair, evaluating it once if needed."""

        key = (_components.family_key(self.families[j]), c)
        if key not in self._evaluations:
            self._evaluations[key] = evaluate_pair(
                self.X, self.H[:, c], self.families[j], self._component_estimator, self._scale
            )
        return self._evaluations[key]
