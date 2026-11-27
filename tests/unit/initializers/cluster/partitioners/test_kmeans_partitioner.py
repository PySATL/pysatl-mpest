"""Tests of the k-means partitioning method."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import re
import warnings

import numpy as np
import pytest
from hypothesis import given
from pysatl_mpest.initializers.cluster._table._validation import validate_partition
from pysatl_mpest.initializers.cluster.partitioners.kmeans_partitioner import KMeansPartitioner
from pysatl_mpest.initializers.exceptions import InitializationError
from pysatl_mpest.typings import UnivariateFloatArray
from scipy.cluster.vq import ClusterError, kmeans2

from tests.helpers.partitions import partition_inputs


def three_overlapping_modes() -> UnivariateFloatArray:
    """Sample of three overlapping normal modes for the tests of this file."""

    rng = np.random.default_rng(42)
    sample = np.concatenate([rng.normal(0.0, 1.0, 40), rng.normal(3.0, 1.0, 40), rng.normal(6.0, 1.0, 40)])
    sample.flags.writeable = False
    return sample


def test_stores_one_hundred_iterations_by_default() -> None:
    assert KMeansPartitioner().max_iter == 100  # noqa: PLR2004


def test_accepts_a_single_iteration() -> None:
    assert KMeansPartitioner(max_iter=1).max_iter == 1


def test_stores_a_numpy_integer_as_a_builtin_int() -> None:
    max_iter = KMeansPartitioner(max_iter=np.int64(5)).max_iter

    assert max_iter == 5  # noqa: PLR2004
    assert type(max_iter) is int


@pytest.mark.parametrize(
    "max_iter",
    [0, -1, 2.5, 3.0, "10", None, True],
    ids=["zero", "negative", "fractional", "whole-float", "string", "none", "bool"],
)
def test_rejects_max_iter_that_is_not_a_positive_integer(max_iter: object) -> None:
    escaped = re.escape(repr(max_iter))

    with pytest.raises(ValueError, match=rf"^max_iter must be a positive integer, got {escaped}\.$"):
        KMeansPartitioner(max_iter=max_iter)


@pytest.mark.parametrize("max_iter", [1, 100], ids=["one-iteration", "default"])
def test_matches_the_direct_kmeans2_result(max_iter: int) -> None:
    sample = three_overlapping_modes()

    H = KMeansPartitioner(max_iter=max_iter).partition(sample, 3, np.random.default_rng(0))
    labels = kmeans2(sample, 3, iter=max_iter, minit="++", missing="raise", rng=np.random.default_rng(0))[1]

    assert H.dtype == np.float64
    np.testing.assert_array_equal(H, np.eye(3)[labels])


def test_number_of_iterations_changes_the_matrix() -> None:
    sample = three_overlapping_modes()

    one_iteration = KMeansPartitioner(max_iter=1).partition(sample, 3, np.random.default_rng(0))
    default_iterations = KMeansPartitioner(max_iter=100).partition(sample, 3, np.random.default_rng(0))

    assert not np.array_equal(one_iteration, default_iterations)


def test_same_seed_gives_the_same_matrix() -> None:
    sample = three_overlapping_modes()

    first = KMeansPartitioner().partition(sample, 3, np.random.default_rng(7))
    second = KMeansPartitioner().partition(sample, 3, np.random.default_rng(7))

    np.testing.assert_array_equal(first, second)


def test_passed_generator_is_advanced() -> None:
    sample = three_overlapping_modes()
    rng = np.random.default_rng(7)
    state = rng.bit_generator.state

    KMeansPartitioner().partition(sample, 3, rng)

    assert rng.bit_generator.state != state


def test_empty_cluster_is_an_unlucky_run_without_warnings() -> None:
    X = np.array([3.0, 2.0, 3.0, 2.0, 3.0, 27.0, 9.0, 16.0, 11.0], dtype=np.float64)
    X.flags.writeable = False

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(InitializationError, match="k-means left one of the 4 clusters without points") as excinfo:
            KMeansPartitioner().partition(X, 4, np.random.default_rng(0))

    assert isinstance(excinfo.value.__cause__, ClusterError)


@pytest.mark.parametrize(
    ("X", "k"),
    [
        pytest.param(np.array([0.0, 0.25, 4.16197014e-203], dtype=np.float64), 3, id="three-points"),
        pytest.param(np.arange(1.0, 9.0) * 1e-170, 2, id="range-below-1e-162"),
    ],
)
def test_sample_of_a_tiny_scale_is_an_unlucky_run_without_warnings(X: UnivariateFloatArray, k: int) -> None:
    X.flags.writeable = False

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(InitializationError, match=f"^k-means left one of the {k} clusters without points"):
            KMeansPartitioner().partition(X, k, np.random.default_rng(0))


@pytest.mark.parametrize(
    ("X", "k"),
    [
        pytest.param(np.array([1e200, -1e200, 0.0, 1.0], dtype=np.float64), 2, id="squared-range"),
        pytest.param(np.array([1.7e308, -1.7e308, 0.0], dtype=np.float64), 2, id="range"),
        pytest.param(np.array([1e308, 1e308, 1e308], dtype=np.float64), 1, id="sum-of-values"),
        pytest.param(np.linspace(-1e152, 1e152, 200_000, dtype=np.float64), 2, id="sum-of-squared-distances"),
    ],
)
def test_rejects_a_sample_where_squared_distances_overflow(X: UnivariateFloatArray, k: int) -> None:
    X.flags.writeable = False

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="^The values of X are too large for k-means"):
            KMeansPartitioner().partition(X, k, np.random.default_rng(0))


def test_accepts_large_values_below_the_overflow_threshold() -> None:
    X = np.array([1e150, -1e150, 0.0, 1.0], dtype=np.float64)
    X.flags.writeable = False

    H = KMeansPartitioner().partition(X, 2, np.random.default_rng(0))

    validate_partition(H, n=4, k=2)


@given(partition_inputs(distinct_values=True))
def test_result_is_a_valid_matrix_or_an_unlucky_run(inputs: tuple[UnivariateFloatArray, int, int]) -> None:
    X, k, seed = inputs

    try:
        H = KMeansPartitioner().partition(X, k, np.random.default_rng(seed))
    except InitializationError:
        return

    validate_partition(H, n=len(X), k=k)
