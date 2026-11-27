"""Unit tests of ``HungarianMatcher`` with a scripted component table."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import re
from typing import cast

import numpy as np
import pytest
from pysatl_mpest.initializers.cluster.matchers.hungarian_matcher import HungarianMatcher
from pysatl_mpest.initializers.cluster.table import ComponentTable
from pysatl_mpest.initializers.exceptions import InitializationError
from pysatl_mpest.typings import UnivariateFloatArray


class ScriptedTable:
    """Component table stub that returns the scripted values and records pair requests."""

    def __init__(
        self,
        loglikelihoods: list[list[float]],
        groups: tuple[tuple[int, ...], ...],
        means: list[float],
    ) -> None:
        self._loglikelihoods = loglikelihoods
        self._groups = groups
        self._means = np.array(means, dtype=np.float64)
        self.k = len(loglikelihoods)
        self.calls: list[tuple[str, int, int]] = []

    def groups(self) -> tuple[tuple[int, ...], ...]:
        return self._groups

    def cluster_means(self) -> UnivariateFloatArray:
        return self._means

    def loglikelihood(self, j: int, c: int) -> float:
        self.calls.append(("loglikelihood", j, c))
        return float(self._loglikelihoods[j][c])

    def failure(self, j: int, c: int) -> str | None:
        self.calls.append(("failure", j, c))
        if np.isneginf(self._loglikelihoods[j][c]):
            return f"reason of ({j}, {c})"
        return None


def make_table(
    loglikelihoods: list[list[float]],
    groups: tuple[tuple[int, ...], ...],
    means: list[float],
) -> ComponentTable:
    """Cast a scripted table to the component table protocol."""

    return cast(ComponentTable, ScriptedTable(loglikelihoods, groups, means))


class TestAssignment:
    """Choice of the assignment with the maximum sum of log-likelihoods."""

    def test_maximizes_the_sum_where_greedy_choice_fails(self) -> None:
        table = make_table([[-100.0, -110.0], [-105.0, -1000.0]], ((0,), (1,)), [0.0, 1.0])

        result = HungarianMatcher().match(table)

        np.testing.assert_array_equal(result, [1, 0])

    def test_finds_best_assignment_in_three_by_three_matrix(self) -> None:
        table = make_table(
            [[-4.0, -1.0, -3.0], [-2.0, 0.0, -5.0], [-3.0, -2.0, -2.0]],
            ((0,), (1,), (2,)),
            [0.0, 1.0, 2.0],
        )

        result = HungarianMatcher().match(table)

        np.testing.assert_array_equal(result, [1, 0, 2])

    @pytest.mark.parametrize(
        ("loglikelihoods", "expected"),
        [
            pytest.param([[-1.0, -np.inf], [-1.0, -1000.0]], [0, 1], id="first-family-keeps-finite-pair"),
            pytest.param([[-np.inf, -1000.0], [-1.0, -1.0]], [1, 0], id="second-family-keeps-finite-pair"),
        ],
    )
    def test_impossible_pair_is_not_chosen_while_assignment_without_it_exists(
        self, loglikelihoods: list[list[float]], expected: list[int]
    ) -> None:
        table = make_table(loglikelihoods, ((0,), (1,)), [0.0, 1.0])

        result = HungarianMatcher().match(table)

        np.testing.assert_array_equal(result, expected)

    def test_identical_families_are_ordered_by_cluster_mean_among_different_families(self) -> None:
        table = make_table(
            [[-1.0, -50.0, -2.0], [-60.0, -5.0, -70.0], [-1.0, -50.0, -2.0]],
            ((0, 2), (1,)),
            [7.0, 3.0, 1.0],
        )

        result = HungarianMatcher().match(table)

        np.testing.assert_array_equal(result, [2, 1, 0])

    def test_requests_every_pair_and_never_asks_for_failures(self) -> None:
        scripted = ScriptedTable(
            [[-4.0, -1.0, -3.0], [-2.0, 0.0, -5.0], [-3.0, -2.0, -2.0]],
            ((0,), (1,), (2,)),
            [0.0, 1.0, 2.0],
        )
        table = cast(ComponentTable, scripted)

        HungarianMatcher().match(table)

        assert {(kind, j, c) for kind, j, c in scripted.calls} == {
            ("loglikelihood", j, c) for j in range(3) for c in range(3)
        }
        assert all(kind == "loglikelihood" for kind, _, _ in scripted.calls)

    def test_returns_one_dimensional_integer_array(self) -> None:
        table = make_table(
            [[-4.0, -1.0, -3.0], [-2.0, 0.0, -5.0], [-3.0, -2.0, -2.0]],
            ((0,), (1,), (2,)),
            [0.0, 1.0, 2.0],
        )

        result = HungarianMatcher().match(table)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.dtype.kind == "i"
        assert len(result) == 3  # noqa: PLR2004


class TestSingleGroup:
    """Single group of identical families."""

    @pytest.mark.parametrize(
        ("loglikelihoods", "groups", "means", "expected"),
        [
            pytest.param(
                [[-np.inf, -np.inf, -np.inf], [-np.inf, -np.inf, -np.inf], [-np.inf, -np.inf, -np.inf]],
                ((0, 1, 2),),
                [7.0, 3.0, 1.0],
                [2, 1, 0],
                id="three-identical-families",
            ),
            pytest.param([[-np.inf]], ((0,),), [0.0], [0], id="single-family"),
        ],
    )
    def test_returns_clusters_ordered_by_mean_without_asking_table(
        self,
        loglikelihoods: list[list[float]],
        groups: tuple[tuple[int, ...], ...],
        means: list[float],
        expected: list[int],
    ) -> None:
        scripted = ScriptedTable(loglikelihoods, groups, means)
        table = cast(ComponentTable, scripted)

        result = HungarianMatcher().match(table)

        np.testing.assert_array_equal(result, expected)
        assert scripted.calls == []


class TestNoAssignment:
    """Impossible pairs when no assignment exists and protocol violations."""

    def test_reports_every_impossible_pair_when_no_assignment_exists(self) -> None:
        table = make_table([[-np.inf, -np.inf], [-1.0, -2.0]], ((0,), (1,)), [0.0, 1.0])
        expected = (
            "No assignment gives every family a suitable cluster. Impossible pairs:\nreason of (0, 0)\nreason of (0, 1)"
        )

        with pytest.raises(InitializationError, match=rf"^{re.escape(expected)}$") as exc_info:
            HungarianMatcher().match(table)

        assert type(exc_info.value) is InitializationError
        assert str(exc_info.value) == expected

    def test_lists_impossible_pairs_in_family_then_cluster_order(self) -> None:
        scripted = ScriptedTable(
            [[-np.inf, -np.inf, -1.0], [-np.inf, -np.inf, -2.0], [-3.0, -np.inf, -4.0]],
            ((0,), (1,), (2,)),
            [0.0, 1.0, 2.0],
        )
        table = cast(ComponentTable, scripted)
        expected = (
            "No assignment gives every family a suitable cluster. Impossible pairs:"
            "\nreason of (0, 0)"
            "\nreason of (0, 1)"
            "\nreason of (1, 0)"
            "\nreason of (1, 1)"
            "\nreason of (2, 1)"
        )

        with pytest.raises(InitializationError, match=rf"^{re.escape(expected)}$") as exc_info:
            HungarianMatcher().match(table)

        assert str(exc_info.value) == expected
        assert [call for call in scripted.calls if call[0] == "failure"] == [
            ("failure", 0, 0),
            ("failure", 0, 1),
            ("failure", 1, 0),
            ("failure", 1, 1),
            ("failure", 2, 1),
        ]

    @pytest.mark.parametrize(
        "loglikelihoods",
        [
            pytest.param([[np.nan, -1.0], [-1.0, -1.0]], id="nan"),
            pytest.param([[np.inf, -1.0], [-1.0, -1.0]], id="plus-infinity"),
        ],
    )
    def test_protocol_violation_passes_value_error_through(self, loglikelihoods: list[list[float]]) -> None:
        table = make_table(loglikelihoods, ((0,), (1,)), [0.0, 1.0])

        with pytest.raises(ValueError, match=r"^The cost matrix must not contain NaN or -inf"):
            HungarianMatcher().match(table)
