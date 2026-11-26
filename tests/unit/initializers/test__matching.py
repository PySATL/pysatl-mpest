"""Unit tests for ``solve_assignment`` and ``order_identical``."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pysatl_mpest.initializers._matching import order_identical, solve_assignment


def cost_matrix(values: list[list[float]]) -> np.ndarray:
    """Build a float64 cost matrix from nested lists."""

    return np.array(values, dtype=np.float64)


def test_finds_assignment_with_minimal_sum_where_greedy_fails() -> None:
    cost = cost_matrix([[100.0, 110.0], [105.0, 1000.0]])

    result = solve_assignment(cost)

    np.testing.assert_array_equal(result, [1, 0])


def test_solves_assignment_for_three_families() -> None:
    cost = cost_matrix([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])

    result = solve_assignment(cost)

    np.testing.assert_array_equal(result, [1, 0, 2])


@pytest.mark.parametrize(
    ("cost", "expected"),
    [
        pytest.param(cost_matrix([[1.0, np.inf], [np.inf, 2.0]]), [0, 1], id="diagonal-allowed"),
        pytest.param(cost_matrix([[np.inf, 3e16], [3e16, 0.0]]), [1, 0], id="huge-costs"),
    ],
)
def test_treats_infinite_cost_as_forbidden_pair(cost: np.ndarray, expected: list[int]) -> None:
    result = solve_assignment(cost)

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "cost",
    [
        pytest.param(cost_matrix([[1.0, np.inf], [2.0, np.inf]]), id="column-forbidden"),
        pytest.param(cost_matrix([[np.inf]]), id="single-forbidden"),
        pytest.param(cost_matrix([[np.inf, np.inf], [np.inf, np.inf]]), id="all-forbidden"),
    ],
)
def test_returns_none_when_no_assignment_avoids_forbidden_pairs(cost: np.ndarray) -> None:
    assert solve_assignment(cost) is None


@pytest.mark.parametrize(
    "cost",
    [
        pytest.param(cost_matrix([[1.0, np.nan], [2.0, 3.0]]), id="nan"),
        pytest.param(cost_matrix([[1.0, -np.inf], [2.0, 3.0]]), id="negative-infinity"),
    ],
)
def test_rejects_nan_and_negative_infinity(cost: np.ndarray) -> None:
    with pytest.raises(ValueError, match="^The cost matrix must not contain NaN or -inf"):
        solve_assignment(cost)


def test_assigns_single_cluster_to_single_family() -> None:
    result = solve_assignment(cost_matrix([[5.0]]))

    np.testing.assert_array_equal(result, [0])


def test_returns_one_dimensional_integer_array() -> None:
    result = solve_assignment(cost_matrix([[100.0, 110.0], [105.0, 1000.0]]))

    assert result is not None
    assert result.ndim == 1
    assert result.dtype.kind == "i"


@given(
    k=st.integers(min_value=1, max_value=5),
    data=st.data(),
)
def test_matches_exhaustive_search(k: int, data: st.DataObject) -> None:
    values = data.draw(
        st.lists(
            st.one_of(st.integers(min_value=-1000, max_value=1000).map(float), st.just(np.inf)),
            min_size=k * k,
            max_size=k * k,
        )
    )
    cost = np.array(values, dtype=np.float64).reshape(k, k)
    best = min(cost[np.arange(k), list(p)].sum() for p in itertools.permutations(range(k)))

    result = solve_assignment(cost)

    if np.isinf(best):
        assert result is None
        return

    assert result is not None
    assert sorted(result.tolist()) == list(range(k))
    assert cost[np.arange(k), result].sum() == best


def test_first_identical_family_gets_cluster_with_smaller_mean() -> None:
    assignment = np.array([0, 1, 2])
    groups = ((0, 2), (1,))
    means = np.array([7.0, 3.0, 1.0])

    result = order_identical(assignment, groups, means)

    np.testing.assert_array_equal(result, [2, 1, 0])


def test_leaves_already_ordered_assignment_unchanged() -> None:
    assignment = np.array([2, 1, 0])
    groups = ((0, 2), (1,))
    means = np.array([7.0, 3.0, 1.0])

    result = order_identical(assignment, groups, means)

    np.testing.assert_array_equal(result, [2, 1, 0])


def test_prefers_smaller_column_on_equal_means() -> None:
    assignment = np.array([1, 0])
    groups = ((0, 1),)
    means = np.array([5.0, 5.0])

    result = order_identical(assignment, groups, means)

    np.testing.assert_array_equal(result, [0, 1])


def test_keeps_assignment_between_distinct_families() -> None:
    assignment = np.array([2, 0, 1])
    groups = ((0,), (1,), (2,))
    means = np.array([5.0, 1.0, 3.0])

    result = order_identical(assignment, groups, means)

    np.testing.assert_array_equal(result, [2, 0, 1])


@pytest.mark.parametrize(
    ("assignment", "groups", "means", "expected"),
    [
        pytest.param(
            np.array([3, 1, 0, 2]),
            ((0, 1, 3), (2,)),
            np.array([4.0, 9.0, 1.0, 2.0]),
            [2, 3, 0, 1],
            id="three-and-one",
        ),
        pytest.param(
            np.array([3, 2, 1, 0]),
            ((0, 2), (1, 3)),
            np.array([1.0, 2.0, 3.0, 4.0]),
            [1, 0, 3, 2],
            id="two-and-two",
        ),
    ],
)
def test_orders_groups_independently(
    assignment: np.ndarray,
    groups: tuple[tuple[int, ...], ...],
    means: np.ndarray,
    expected: list[int],
) -> None:
    result = order_identical(assignment, groups, means)

    np.testing.assert_array_equal(result, expected)


def test_returns_new_array_without_modifying_input() -> None:
    assignment = np.array([0, 1, 2])
    groups = ((0, 2), (1,))
    means = np.array([7.0, 3.0, 1.0])

    result = order_identical(assignment, groups, means)

    np.testing.assert_array_equal(assignment, [0, 1, 2])
    assert result is not assignment


@given(
    k=st.integers(min_value=1, max_value=6),
    data=st.data(),
)
def test_preserves_permutation_and_group_clusters(k: int, data: st.DataObject) -> None:
    labels = data.draw(st.lists(st.integers(min_value=0, max_value=2), min_size=k, max_size=k))
    assignment = np.array(data.draw(st.permutations(list(range(k)))))
    means = np.array(data.draw(st.lists(st.sampled_from([-1.0, 0.0, 2.5]), min_size=k, max_size=k)), dtype=np.float64)

    groups_by_label: dict[int, list[int]] = {}
    for family, label in enumerate(labels):
        groups_by_label.setdefault(label, []).append(family)
    groups = tuple(tuple(members) for members in groups_by_label.values())

    result = order_identical(assignment, groups, means)

    assert sorted(result.tolist()) == list(range(k))
    for members in groups:
        assert set(result[list(members)].tolist()) == set(assignment[list(members)].tolist())
        pairs = [(means[c], c) for c in result[list(members)]]
        assert pairs == sorted(pairs)
