"""Abstract matcher of clusters and distribution families with the template method ``match``."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from typing import final

from ....typings import UnivariateIntArray
from ..table import ComponentTable
from ._matching import order_identical


class Matcher(ABC):
    """Decides which cluster is assigned to which distribution family.

    The result is a permutation of ``0…k-1``: family ``j`` receives cluster ``assignment[j]``.

    To write a matcher, implement ``_match`` and use only the component table: take the values and
    the reasons of impossible pairs from it instead of estimating anything again. The public
    :meth:`match` orders identical families after the matching step. Override the hook ``_order``
    only when this order is defined by someone else.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        match
    """

    @final
    def match(self, table: ComponentTable) -> UnivariateIntArray:
        """Match clusters to families and order identical families.

        The template method calls ``_match`` and then the hook ``_order``; it is closed to
        overriding.

        Parameters
        ----------
        table : ComponentTable
            Everything the matcher may ask about the pairs of families and clusters.

        Returns
        -------
        UnivariateIntArray
            Permutation where ``assignment[j]`` is the cluster of family ``j``; identical families are
            ordered by the hook.
        """

        return self._order(self._match(table), table)

    @abstractmethod
    def _match(self, table: ComponentTable) -> UnivariateIntArray:
        """Return the cluster assigned to every family.

        The distribution of clusters within a group of identical families may be arbitrary because
        :meth:`match` canonicalizes it afterwards.

        Parameters
        ----------
        table : ComponentTable
            Everything the matcher may ask about the pairs of families and clusters.

        Returns
        -------
        UnivariateIntArray
            Array where element ``j`` is the cluster assigned to family ``j``.
        """

    def _order(self, assignment: UnivariateIntArray, table: ComponentTable) -> UnivariateIntArray:
        """Order clusters of identical families.

        By default, within every group of identical families the clusters go by increasing mean, and
        equal means are ordered by the column number. The function ``order_identical`` returns a new
        array.

        Parameters
        ----------
        assignment : UnivariateIntArray
            Assignment returned by ``_match``.
        table : ComponentTable
            Source of the groups of identical families and the cluster means.

        Returns
        -------
        UnivariateIntArray
            New array with canonicalized order inside groups of identical families.
        """

        return order_identical(assignment, table.groups(), table.cluster_means())
