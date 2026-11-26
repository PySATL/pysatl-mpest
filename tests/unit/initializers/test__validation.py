"""Unit tests for the pure validation helpers of the ``initializers`` package."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import re

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import ArrayLike
from pysatl_mpest.core._numerics import NUMERICAL_TOLERANCE
from pysatl_mpest.initializers import _validation
from pysatl_mpest.initializers._validation import (
    cluster_is_large_enough,
    component_is_wide_enough,
    exceeds_m_step_tolerance,
    has_enough_distinct_values,
    validate_partition,
    validate_permutation,
)
from pysatl_mpest.initializers.exceptions import InitializationError


class TestValidatePartition:
    """Checks of the hard partition matrix returned by a partitioning method."""

    @pytest.mark.parametrize(
        "H",
        [
            pytest.param(np.array([[1, 0], [0, 1], [1, 0]], dtype=np.bool_), id="bool"),
            pytest.param(np.array([[1, 0], [0, 1], [1, 0]], dtype=np.int64), id="int"),
            pytest.param(np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float64), id="float"),
        ],
    )
    def test_returns_float64_matrix_with_same_values(self, H: ArrayLike) -> None:
        result = validate_partition(H, n=3, k=2)

        assert result.dtype == np.float64
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result, np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))

    @pytest.mark.parametrize(
        ("H", "n", "k"),
        [
            pytest.param(np.array([1, 0, 1]), 3, 2, id="one-dimensional"),
            pytest.param(np.array([[1, 0, 0], [0, 1, 0]]), 2, 2, id="wrong-number-of-columns"),
            pytest.param(np.array([[1, 0], [0, 1]]), 3, 2, id="wrong-number-of-rows"),
        ],
    )
    def test_rejects_array_of_wrong_shape(self, H: ArrayLike, n: int, k: int) -> None:
        with pytest.raises(ValueError, match=r"must have shape \(\d+, 2\)"):
            validate_partition(H, n=n, k=k)

    @pytest.mark.parametrize(
        "H",
        [
            pytest.param(np.array([[2, -1], [0, 1]]), id="other-integers"),
            pytest.param(np.array([[0.5, 0.5], [0.0, 1.0]]), id="fractional"),
            pytest.param(np.array([[np.nan, 1.0], [0.0, 1.0]]), id="nan"),
        ],
    )
    def test_rejects_values_other_than_zero_and_one(self, H: ArrayLike) -> None:
        with pytest.raises(ValueError, match="^A partition matrix must contain only zeros and ones"):
            validate_partition(H, n=2, k=2)

    @pytest.mark.parametrize(
        ("H", "n"),
        [
            pytest.param(np.array([[1, 1], [0, 1]]), 2, id="two-ones"),
            pytest.param(np.array([[0, 0], [1, 0], [0, 1]]), 3, id="no-ones"),
        ],
    )
    def test_rejects_row_without_exactly_one_one(self, H: ArrayLike, n: int) -> None:
        with pytest.raises(ValueError, match="^Every row of a partition matrix must contain exactly one 1"):
            validate_partition(H, n=n, k=2)

    @pytest.mark.parametrize(
        ("H", "n", "expected_indices"),
        [
            pytest.param(np.array([[1, 0, 0], [0, 0, 1], [1, 0, 0]]), 3, "[1]", id="one-empty"),
            pytest.param(np.array([[1, 0, 0], [1, 0, 0]]), 2, "[1, 2]", id="two-empty"),
        ],
    )
    def test_rejects_empty_clusters(self, H: ArrayLike, n: int, expected_indices: str) -> None:
        with pytest.raises(InitializationError, match=rf"empty clusters: {re.escape(expected_indices)}"):
            validate_partition(H, n=n, k=3)

    def test_structure_error_takes_precedence_over_empty_cluster(self) -> None:
        with pytest.raises(ValueError, match="^Every row of a partition matrix"):
            validate_partition(np.array([[1, 0], [0, 0]]), n=2, k=2)


@st.composite
def _hard_partition(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, int]:
    """A random hard partition of ``n`` points into ``k`` non-empty clusters, as ``(H, labels, k)``."""

    k = draw(st.integers(min_value=1, max_value=5))
    extras = draw(st.lists(st.integers(min_value=0, max_value=k - 1), max_size=20))
    labels = np.asarray(draw(st.permutations(list(range(k)) + extras)))
    dtype = draw(st.sampled_from([np.bool_, np.int64, np.float64]))

    return np.eye(k, dtype=dtype)[labels], labels, k


@given(_hard_partition())
def test_hard_partition_without_empty_clusters_passes(partition: tuple[np.ndarray, np.ndarray, int]) -> None:
    H, labels, k = partition

    result = validate_partition(H, n=len(labels), k=k)

    assert result.dtype == np.float64
    assert result.shape == (len(labels), k)
    np.testing.assert_array_equal(result.argmax(axis=1), labels)


class TestValidatePermutation:
    """Checks of the cluster-to-family assignment returned by a matcher."""

    @pytest.mark.parametrize(
        ("assignment", "k"),
        [
            pytest.param(np.array([0, 1, 2]), 3, id="identity"),
            pytest.param(np.array([2, 0, 1]), 3, id="shuffled"),
            pytest.param(np.array([0]), 1, id="single"),
        ],
    )
    def test_accepts_permutation(self, assignment: ArrayLike, k: int) -> None:
        assert validate_permutation(assignment, k=k) is None

    @pytest.mark.parametrize(
        "assignment",
        [
            pytest.param(np.array([0, 0, 1]), id="repeated"),
            pytest.param(np.array([0, 1]), id="too-short"),
            pytest.param(np.array([0, 1, 2, 3]), id="too-long"),
            pytest.param(np.array([0, 1, 3]), id="out-of-range"),
            pytest.param(np.array([-1, 0, 1]), id="negative"),
            pytest.param(np.array([0.0, 1.0, 2.0]), id="real-valued"),
            pytest.param(np.array([[0, 1, 2]]), id="two-dimensional"),
            pytest.param(np.array(1), id="scalar"),
        ],
    )
    def test_rejects_value_that_is_not_a_permutation(self, assignment: ArrayLike) -> None:
        with pytest.raises(ValueError, match="permutation of length 3"):
            validate_permutation(assignment, k=3)

    def test_error_message_contains_received_value(self) -> None:
        with pytest.raises(ValueError, match=r"got \[0, 0, 1\]"):
            validate_permutation(np.array([0, 0, 1]), k=3)


class TestThresholds:
    """Boundary predicates for degenerate clusters and M-step quantities."""

    @pytest.mark.parametrize(
        ("size", "min_cluster_size", "expected"),
        [
            pytest.param(4, 4, True, id="equal-to-minimum"),
            pytest.param(3, 4, False, id="below-minimum"),
            pytest.param(5, 4, True, id="above-minimum"),
            pytest.param(1, 1, True, id="single-point-minimum"),
            pytest.param(0, 1, False, id="empty-cluster"),
        ],
    )
    def test_cluster_passes_from_minimum_size(self, size: int, min_cluster_size: int, expected: bool) -> None:
        assert cluster_is_large_enough(size, min_cluster_size) == expected

    @pytest.mark.parametrize(
        ("n_distinct", "n_free_parameters", "expected"),
        [
            pytest.param(2, 2, True, id="equal-to-free-parameters"),
            pytest.param(1, 2, False, id="fewer-than-free-parameters"),
            pytest.param(3, 2, True, id="more-than-free-parameters"),
            pytest.param(1, 1, True, id="single-value-single-parameter"),
            pytest.param(0, 1, False, id="no-distinct-values"),
        ],
    )
    def test_passes_from_as_many_distinct_values_as_free_parameters(
        self, n_distinct: int, n_free_parameters: int, expected: bool
    ) -> None:
        assert has_enough_distinct_values(n_distinct, n_free_parameters) == expected

    @pytest.mark.parametrize(
        ("width", "scale", "expected"),
        [
            pytest.param(1e-6, 1.0, False, id="equal-to-relative-threshold"),
            pytest.param(np.nextafter(1e-6, 1.0), 1.0, True, id="just-above-relative-threshold"),
            pytest.param(2e-6, 2.0, False, id="scaled-equal-to-threshold"),
            pytest.param(np.nextafter(2e-6, 1.0), 2.0, True, id="scaled-just-above-threshold"),
            pytest.param(0.0, 0.03, False, id="zero-width"),
            pytest.param(2.70, 3.0, True, id="wide-component"),
            pytest.param(np.nan, 1.0, False, id="nan-width"),
        ],
    )
    def test_width_must_strictly_exceed_relative_threshold(self, width: float, scale: float, expected: bool) -> None:
        assert component_is_wide_enough(width, scale) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(NUMERICAL_TOLERANCE, False, id="equal-to-tolerance"),
            pytest.param(np.nextafter(NUMERICAL_TOLERANCE, 1.0), True, id="just-above-tolerance"),
            pytest.param(0.0, False, id="zero"),
            pytest.param(1.0, True, id="large-value"),
        ],
    )
    def test_value_must_strictly_exceed_m_step_tolerance(self, value: float, expected: bool) -> None:
        assert exceeds_m_step_tolerance(value) == expected

    def test_m_step_threshold_reads_module_tolerance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_validation, "NUMERICAL_TOLERANCE", 0.25)

        assert not exceeds_m_step_tolerance(0.25)
        assert exceeds_m_step_tolerance(0.5)
