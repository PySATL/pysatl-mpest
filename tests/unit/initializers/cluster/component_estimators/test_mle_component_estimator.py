"""Unit tests for the closed-form maximum likelihood estimation formulas."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Mapping

import numpy as np
import pytest
from hypothesis import given
from numpy.testing import assert_allclose
from pysatl_mpest.initializers.cluster.component_estimators.mle_component_estimator import (
    _continuous_uniform_standard,
    _exponential_rate,
    _normal_mean_std,
)
from pysatl_mpest.typings import UnivariateFloatArray

from tests.helpers.partitions import partition_inputs, random_partition

type Formula = Callable[[UnivariateFloatArray, UnivariateFloatArray, Mapping[str, float]], dict[str, float]]


class TestNormalMeanStd:
    """Weighted mean and standard deviation of a cluster."""

    def test_returns_mean_and_standard_deviation_without_fixed_parameters(self) -> None:
        X = np.array([4, 5, 6, 100], dtype=np.float64)
        h = np.array([1, 1, 1, 0], dtype=np.float64)

        result = _normal_mean_std(X, h, {})

        assert result == {"mu": 5.0, "sigma": 0.816496580927726}
        assert all(type(value) is float for value in result.values())

    def test_deviation_is_taken_around_fixed_mean(self) -> None:
        X = np.array([4, 5, 6, 100], dtype=np.float64)
        h = np.array([1, 1, 1, 0], dtype=np.float64)

        result = _normal_mean_std(X, h, {"mu": 0})

        assert result == {"sigma": 5.066228051190222}
        assert all(type(value) is float for value in result.values())

    def test_fixed_standard_deviation_is_not_returned(self) -> None:
        X = np.array([4, 5, 6, 100], dtype=np.float64)
        h = np.array([1, 1, 1, 0], dtype=np.float64)

        result = _normal_mean_std(X, h, {"sigma": 2.0})

        assert result == {"mu": 5.0}
        assert all(type(value) is float for value in result.values())

    def test_cluster_of_one_point_gives_zero_standard_deviation(self) -> None:
        X = np.array([5, 100], dtype=np.float64)
        h = np.array([1, 0], dtype=np.float64)

        result = _normal_mean_std(X, h, {})

        assert result == {"mu": 5.0, "sigma": 0.0}


@given(partition_inputs())
def test_unfixed_normal_parameters_match_cluster_statistics(inputs: tuple[UnivariateFloatArray, int, int]) -> None:
    X, k, seed = inputs
    h = random_partition(len(X), k, seed)[:, 0]
    points = X[h > 0]

    free = _normal_mean_std(X, h, {})

    atol = 1e-12 * (1 + np.abs(points).max())
    assert_allclose(free["mu"], points.mean(), rtol=0, atol=atol)
    assert_allclose(free["sigma"], points.std(), rtol=0, atol=atol)


@given(partition_inputs())
def test_fixed_mean_gives_root_mean_square_deviation_from_it(
    inputs: tuple[UnivariateFloatArray, int, int],
) -> None:
    X, k, seed = inputs
    h = random_partition(len(X), k, seed)[:, 0]
    points = X[h > 0]

    free = _normal_mean_std(X, h, {"mu": 1.5})

    atol = 1e-12 * (1 + np.abs(points).max())
    assert set(free) == {"sigma"}
    assert_allclose(free["sigma"], np.sqrt(np.mean((points - 1.5) ** 2)), rtol=0, atol=atol)


class TestExponentialRate:
    """Inverse weighted mean of a cluster."""

    def test_returns_inverse_cluster_mean(self) -> None:
        X = np.array([1, 2, 6, 100], dtype=np.float64)
        h = np.array([1, 1, 1, 0], dtype=np.float64)

        result = _exponential_rate(X, h, {})

        assert result == {"lambda_": 1 / 3}
        assert all(type(value) is float for value in result.values())

    @pytest.mark.parametrize(
        ("x_values", "expected"),
        [
            pytest.param([-1, 5, 100], 0.5, id="point-outside-support"),
            pytest.param([-5, 1, 100], -0.5, id="negative-mean"),
        ],
    )
    def test_returns_inadmissible_values_without_error(self, x_values: list[float], expected: float) -> None:
        X = np.array(x_values, dtype=np.float64)
        h = np.array([1, 1, 0], dtype=np.float64)

        result = _exponential_rate(X, h, {})

        assert result == {"lambda_": expected}


@given(partition_inputs())
def test_exponential_rate_times_cluster_mean_is_one(inputs: tuple[UnivariateFloatArray, int, int]) -> None:
    X, k, seed = inputs
    X = np.abs(X) + 1.0
    h = random_partition(len(X), k, seed)[:, 0]

    free = _exponential_rate(X, h, {})

    assert free["lambda_"] * X[h > 0].mean() == pytest.approx(1.0, rel=1e-12)


class TestContinuousUniformStandard:
    """Smallest and largest values of a cluster."""

    def test_returns_smallest_and_largest_cluster_values(self) -> None:
        X = np.array([2, 7, 3, 100, -50], dtype=np.float64)
        h = np.array([1, 1, 1, 0, 0], dtype=np.float64)

        result = _continuous_uniform_standard(X, h, {})

        assert result == {"lower_bound": 2.0, "upper_bound": 7.0}
        assert all(type(value) is float for value in result.values())

    @pytest.mark.parametrize(
        ("fixed", "expected"),
        [
            pytest.param({"lower_bound": 0.0}, {"upper_bound": 7.0}, id="fixed-lower"),
            pytest.param({"upper_bound": 9}, {"lower_bound": 2.0}, id="fixed-upper"),
        ],
    )
    def test_fixed_bound_is_not_returned(self, fixed: dict[str, float], expected: dict[str, float]) -> None:
        X = np.array([2, 7, 3, 100, -50], dtype=np.float64)
        h = np.array([1, 1, 1, 0, 0], dtype=np.float64)

        result = _continuous_uniform_standard(X, h, fixed)

        assert result == expected
        assert all(type(value) is float for value in result.values())

    def test_cluster_of_one_point_gives_equal_bounds(self) -> None:
        X = np.array([3, 100, -50], dtype=np.float64)
        h = np.array([1, 0, 0], dtype=np.float64)

        result = _continuous_uniform_standard(X, h, {})

        assert result == {"lower_bound": 3.0, "upper_bound": 3.0}


@given(partition_inputs())
def test_uniform_bounds_are_exactly_cluster_extremes(inputs: tuple[UnivariateFloatArray, int, int]) -> None:
    X, k, seed = inputs
    h = random_partition(len(X), k, seed)[:, 0]

    free = _continuous_uniform_standard(X, h, {})

    assert free == {"lower_bound": X[h > 0].min(), "upper_bound": X[h > 0].max()}


class TestInputs:
    """Treatment of the arrays and the mapping passed to a formula."""

    @pytest.mark.parametrize(
        ("formula", "fixed"),
        [
            pytest.param(_normal_mean_std, {"mu": 0.0}, id="normal"),
            pytest.param(_exponential_rate, {}, id="exponential"),
            pytest.param(_continuous_uniform_standard, {"lower_bound": 0.0}, id="uniform"),
        ],
    )
    def test_read_only_arrays_are_accepted_and_inputs_are_not_changed(
        self, formula: Formula, fixed: dict[str, float]
    ) -> None:
        X = np.array([2, 7, 3, 100, -50], dtype=np.float64)
        h = np.array([1, 1, 1, 0, 0], dtype=np.float64)
        X.flags.writeable = False
        h.flags.writeable = False
        X_before = X.copy()
        h_before = h.copy()
        fixed_before = fixed.copy()

        formula(X, h, fixed)

        np.testing.assert_array_equal(X, X_before)
        np.testing.assert_array_equal(h, h_before)
        assert fixed == fixed_before
