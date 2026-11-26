"""
Contract of the component table: what a matcher may ask about the pairs of families and clusters.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Protocol

from ....typings import UnivariateFloatArray


class ComponentTable(Protocol):
    """Everything a matcher may ask about the components estimated for pairs of a family and a cluster.

    Matchers use only this protocol and see neither the sample, nor the partition, nor the component
    estimator.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        groups
        cluster_means
        loglikelihood
        failure
    """

    @property
    def k(self) -> int:
        """Number of families, equal to the number of clusters."""

    def groups(self) -> tuple[tuple[int, ...], ...]:
        """Groups of identical families as tuples of family indices.

        Groups follow the order of the first occurrence, indices inside a group are ascending, and
        every index belongs to exactly one group.
        """

    def cluster_means(self) -> UnivariateFloatArray:
        """Means of the clusters, indexed by the column of the partition."""

    def loglikelihood(self, j: int, c: int) -> float:
        """Log-likelihood of cluster ``c`` under the component of family ``j`` estimated on that cluster.

        It is a finite number or ``-inf``; ``NaN`` and ``+inf`` never occur.
        """

    def failure(self, j: int, c: int) -> str | None:
        """Why the log-likelihood of the pair is not finite, or ``None`` exactly when it is finite."""
