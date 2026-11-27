"""Matcher that keeps the correspondence between family and cluster indices chosen by the user."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np

from ....typings import UnivariateIntArray
from ..table import ComponentTable
from .base import Matcher


class IdentityMatcher(Matcher):
    """Assigns every family the cluster with the same index.

    The matcher is appropriate when the user has already ordered the families consistently with the
    clusters: family ``j`` receives cluster ``j`` without comparing log-likelihoods. It neither reorders
    identical families nor avoids impossible pairs; an impossible pair is found later during the mixture
    assembly.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        match
    """

    def _match(self, table: ComponentTable) -> UnivariateIntArray:
        """Return the identity assignment.

        Parameters
        ----------
        table : ComponentTable
            Source of the number of families and clusters.

        Returns
        -------
        UnivariateIntArray
            Array ``[0, 1, ..., k - 1]``.
        """

        return np.arange(table.k)

    def _order(self, assignment: UnivariateIntArray, table: ComponentTable) -> UnivariateIntArray:
        """Keep the assignment unchanged because its order is chosen by the user.

        Parameters
        ----------
        assignment : UnivariateIntArray
            Identity assignment returned by ``_match``.
        table : ComponentTable
            Unused component table.

        Returns
        -------
        UnivariateIntArray
            The received assignment.
        """

        return assignment
