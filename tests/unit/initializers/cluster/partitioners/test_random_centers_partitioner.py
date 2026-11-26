"""Tests of partitioning by random centers drawn from the distinct values of the sample."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import itertools

import numpy as np
from hypothesis import given
from pysatl_mpest.initializers.cluster._table._validation import validate_partition
from pysatl_mpest.initializers.cluster.partitioners.random_centers_partitioner import RandomCentersPartitioner
from pysatl_mpest.typings import UnivariateFloatArray

from tests.helpers.partitions import partition_inputs


def nearest_center_labels(values: list[float], centers: tuple[float, ...]) -> tuple[int, ...]:
    """Reference rule: the nearest center, on a tie the center with the smaller column."""

    labels = []
    for value in values:
        distances = [abs(value - center) for center in centers]
        labels.append(distances.index(min(distances)))
    return tuple(labels)


def test_same_seed_gives_same_matrix() -> None:
    sample = np.random.default_rng(42).normal(size=60)
    k = 3

    first = RandomCentersPartitioner().partition(sample, k, np.random.default_rng(5))
    second = RandomCentersPartitioner().partition(sample, k, np.random.default_rng(5))

    np.testing.assert_array_equal(first, second)


def test_reused_generator_gives_different_matrix() -> None:
    sample = np.random.default_rng(42).normal(size=60)
    rng = np.random.default_rng(1)

    first = RandomCentersPartitioner().partition(sample, 3, rng)
    second = RandomCentersPartitioner().partition(sample, 3, rng)

    assert not np.array_equal(first, second)


def test_distinct_centers_keep_repeated_values_together() -> None:
    sample = np.array([2.0, 7.0, 7.0, 2.0, 11.0, 2.0], dtype=np.float64)
    sample.flags.writeable = False

    for seed in range(50):
        H = RandomCentersPartitioner().partition(sample, 3, np.random.default_rng(seed))

        validate_partition(H, n=6, k=3)
        assert H.dtype == np.float64
        for column in range(3):
            assert len(np.unique(sample[H[:, column] == 1])) == 1


def test_equidistant_point_goes_to_smaller_column() -> None:
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    sample = np.array(values, dtype=np.float64)
    allowed = {nearest_center_labels(values, centers) for centers in itertools.permutations(values, 4)}

    for seed in range(50):
        H = RandomCentersPartitioner().partition(sample, 4, np.random.default_rng(seed))

        assert tuple(H.argmax(axis=1).tolist()) in allowed


@given(partition_inputs(distinct_values=True))
def test_repeated_values_stay_in_one_cluster(inputs: tuple[UnivariateFloatArray, int, int]) -> None:
    X, k, seed = inputs

    H = validate_partition(RandomCentersPartitioner().partition(X, k, np.random.default_rng(seed)), n=len(X), k=k)
    cluster_values = [set(X[H[:, column] == 1].tolist()) for column in range(k)]

    for first, second in itertools.combinations(cluster_values, 2):
        assert first.isdisjoint(second)
