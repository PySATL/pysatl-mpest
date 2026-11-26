"""Unit tests for the template method of ``Matcher``."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import cast

import numpy as np
from pysatl_mpest.initializers.cluster.matchers.base import Matcher
from pysatl_mpest.initializers.cluster.table import ComponentTable
from pysatl_mpest.typings import UnivariateFloatArray, UnivariateIntArray


class GroupedTable:
    """Component table stub with fixed groups and cluster means."""

    def __init__(self, groups: tuple[tuple[int, ...], ...], means: list[float]) -> None:
        self._groups = groups
        self._means = np.array(means, dtype=np.float64)
        self.k = len(self._means)

    def groups(self) -> tuple[tuple[int, ...], ...]:
        return self._groups

    def cluster_means(self) -> UnivariateFloatArray:
        return self._means

    def loglikelihood(self, j: int, c: int) -> float:
        return 0.0

    def failure(self, j: int, c: int) -> str | None:
        return None


class FixedMatcher(Matcher):
    """Matcher stub that always returns the same assignment."""

    def __init__(self, assignment: list[int]) -> None:
        self.assignment: UnivariateIntArray = np.array(assignment)
        self.tables: list[ComponentTable] = []

    def _match(self, table: ComponentTable) -> UnivariateIntArray:
        self.tables.append(table)

        return self.assignment


class KeepingOrderMatcher(FixedMatcher):
    """Matcher stub that keeps the order chosen by ``_match``."""

    def __init__(self, assignment: list[int]) -> None:
        super().__init__(assignment)
        self.order_calls: list[tuple[UnivariateIntArray, ComponentTable]] = []

    def _order(self, assignment: UnivariateIntArray, table: ComponentTable) -> UnivariateIntArray:
        self.order_calls.append((assignment, table))

        return assignment


def test_first_identical_family_gets_cluster_with_smaller_mean() -> None:
    matcher = FixedMatcher([0, 1, 2])
    table = cast(ComponentTable, GroupedTable(((0, 2), (1,)), [7.0, 3.0, 1.0]))

    result = matcher.match(table)

    np.testing.assert_array_equal(result, [2, 1, 0])


def test_keeps_assignment_between_distinct_families() -> None:
    matcher = FixedMatcher([2, 0, 1])
    table = cast(ComponentTable, GroupedTable(((0,), (1,), (2,)), [5.0, 1.0, 3.0]))

    result = matcher.match(table)

    np.testing.assert_array_equal(result, [2, 0, 1])


def test_passes_the_same_table_to_matching_step() -> None:
    matcher = FixedMatcher([0, 1, 2])
    table = cast(ComponentTable, GroupedTable(((0, 2), (1,)), [7.0, 3.0, 1.0]))

    matcher.match(table)

    assert len(matcher.tables) == 1
    assert matcher.tables[0] is table


def test_overridden_ordering_hook_replaces_default_order() -> None:
    matcher = KeepingOrderMatcher([0, 1, 2])
    table = cast(ComponentTable, GroupedTable(((0, 2), (1,)), [7.0, 3.0, 1.0]))

    result = matcher.match(table)

    assert result is matcher.assignment
    np.testing.assert_array_equal(result, [0, 1, 2])
    assert len(matcher.order_calls) == 1
    ordered_assignment, ordered_table = matcher.order_calls[0]
    assert ordered_assignment is matcher.assignment
    assert ordered_table is table
