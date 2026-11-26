"""Tests of deterministic partitioning into equal-sized segments of the sorted sample."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from hypothesis import given
from pysatl_mpest.initializers.cluster._table._validation import validate_partition
from pysatl_mpest.initializers.cluster.partitioners.quantile_partitioner import QuantilePartitioner
from pysatl_mpest.typings import UnivariateFloatArray

from tests.helpers.partitions import partition_inputs


def labels_of(values: list[float], k: int) -> list[int]:
    """Partition the values and return the cluster labels of their points."""

    sample = np.array(values, dtype=np.float64)
    sample.flags.writeable = False
    H = QuantilePartitioner().partition(sample, k, np.random.default_rng(0))

    return H.argmax(axis=1).tolist()


def test_column_is_segment_of_sorted_sample() -> None:
    sample = np.array([5.0, 1.0, 4.0, 2.0, 3.0, 9.0, 7.0], dtype=np.float64)
    sample.flags.writeable = False

    H = QuantilePartitioner().partition(sample, 3, np.random.default_rng(0))

    assert H.dtype == np.float64
    np.testing.assert_array_equal(H, [[0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]])


@pytest.mark.parametrize(
    ("n", "k"),
    [(7, 3), (6, 3), (5, 5), (10, 4), (4, 1), (11, 3)],
    ids=[
        "one-longer-segment",
        "equal-segments",
        "one-point-per-cluster",
        "two-longer-segments",
        "single-cluster",
        "two-of-three-longer",
    ],
)
def test_clusters_are_array_split_parts_of_sorted_sample(n: int, k: int) -> None:
    sample = np.random.default_rng(42).normal(size=n)
    sample.flags.writeable = False

    H = QuantilePartitioner().partition(sample, k, np.random.default_rng(0))

    assert H.shape == (n, k)
    for c in range(k):
        np.testing.assert_array_equal(np.sort(sample[H[:, c] == 1]), np.array_split(np.sort(sample), k)[c])


def test_repeated_values_do_not_move_the_boundary() -> None:
    assert labels_of([1.0, 2.0, 2.0, 2.0, 2.0, 3.0], 2) == [0, 0, 0, 1, 1, 1]


def test_among_equal_values_lower_index_goes_to_left_cluster() -> None:
    values = [2.0, 1.0, 0.0, 2.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 2.0]

    assert labels_of(values, 3) == [2, 0, 0, 2, 1, 1, 0, 2, 1, 0, 1, 2]


def test_divides_sample_of_identical_values() -> None:
    assert labels_of([0.1, 0.1, 0.1, 0.1, 0.1], 2) == [0, 0, 0, 1, 1]


def test_does_not_use_generator() -> None:
    sample = np.random.default_rng(42).normal(size=20)
    sample.flags.writeable = False
    rng = np.random.default_rng(3)
    state = rng.bit_generator.state

    H = QuantilePartitioner().partition(sample, 3, rng)
    other_H = QuantilePartitioner().partition(sample, 3, np.random.default_rng(99))

    assert rng.bit_generator.state == state
    np.testing.assert_array_equal(H, other_H)


@given(partition_inputs())
def test_partition_is_valid_ordered_and_equal_sized(inputs: tuple[UnivariateFloatArray, int, int]) -> None:
    X, k, seed = inputs

    H = validate_partition(QuantilePartitioner().partition(X, k, np.random.default_rng(seed)), n=len(X), k=k)

    sizes = H.sum(axis=0)
    assert sizes.max() - sizes.min() <= 1
    for c in range(k - 1):
        assert X[H[:, c] == 1].max() <= X[H[:, c + 1] == 1].min()
