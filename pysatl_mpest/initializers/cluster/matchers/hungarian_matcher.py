"""Exact matcher that maximizes the total log-likelihood by the Hungarian algorithm."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np

from ....typings import UnivariateIntArray
from ...exceptions import InitializationError
from ..table import ComponentTable
from ._matching import solve_assignment
from .base import Matcher


class HungarianMatcher(Matcher):
    """Matches clusters to families by the maximum sum of the pair log-likelihoods.

    The matcher solves the assignment problem exactly, so it can choose a locally worse pair when this makes
    the total log-likelihood larger. A pair with log-likelihood ``-inf`` is impossible: it is never chosen
    while an assignment without impossible pairs exists. If every assignment contains an impossible pair, the
    error lists the reasons of all impossible pairs.

    When all families form a single group of identical families, all assignments are equivalent. In this case
    the pair log-likelihoods and failure reasons are not requested at all: the identity permutation is returned
    and ``match`` orders the clusters by their means. For identical families among different families, the order
    inside each group is also canonicalized by ``match`` after the exact assignment is found.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        match
    """

    def _match(self, table: ComponentTable) -> UnivariateIntArray:
        """Return an assignment with the maximum sum of pair log-likelihoods.

        Parameters
        ----------
        table : ComponentTable
            Source of the pair log-likelihoods and the reasons of impossible pairs.

        Returns
        -------
        UnivariateIntArray
            Array where element ``j`` is the cluster assigned to family ``j``.

        Raises
        ------
        InitializationError
            If every assignment contains an impossible pair. The message lists the reasons of all impossible
            pairs in the order of families and then clusters.
        """

        k = table.k
        if len(table.groups()) == 1:
            return np.arange(k)

        loglikelihoods = np.array([[table.loglikelihood(j, c) for c in range(k)] for j in range(k)])
        assignment = solve_assignment(-loglikelihoods)
        if assignment is None:
            reasons = "".join(
                f"\n{table.failure(j, c)}" for j in range(k) for c in range(k) if np.isneginf(loglikelihoods[j, c])
            )
            raise InitializationError(
                "No assignment gives every family a suitable cluster. Impossible pairs:" + reasons
            )

        return assignment
