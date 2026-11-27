"""Integration tests of ``HungarianMatcher`` with real component tables."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import re
from collections.abc import Callable, Sequence
from itertools import permutations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pysatl_core.families import ParametricFamily
from pysatl_core.types import FamilyName
from pysatl_mpest.initializers.cluster._table._lazy import LazyComponentTable
from pysatl_mpest.initializers.cluster.component_estimators.mle_component_estimator import MLEComponentEstimator
from pysatl_mpest.initializers.cluster.matchers.hungarian_matcher import HungarianMatcher
from pysatl_mpest.initializers.exceptions import InitializationError
from pysatl_mpest.typings import UnivariateFloatArray

from tests.helpers.families import family
from tests.helpers.partitions import MAX_CLUSTERS, partition_inputs, random_partition

ERROR_HEADER = "No assignment gives every family a suitable cluster. Impossible pairs:"
FACTORIES: tuple[Callable[[], ParametricFamily], ...] = (
    lambda: family(FamilyName.NORMAL),
    lambda: family(FamilyName.EXPONENTIAL),
    lambda: family(FamilyName.CONTINUOUS_UNIFORM),
    lambda: family(FamilyName.NORMAL).view(mu=0),
)


def make_table(
    values: Sequence[float],
    labels: Sequence[int],
    families: Sequence[ParametricFamily],
) -> LazyComponentTable:
    """A table over a hard partition given by cluster labels, with the real estimator."""

    X = np.asarray(values, dtype=np.float64)
    H = np.eye(len(families))[np.asarray(labels)]
    X.flags.writeable = False
    H.flags.writeable = False
    return LazyComponentTable(X, H, families, MLEComponentEstimator())


def test_exponential_gets_the_cluster_it_covers() -> None:
    normal = family(FamilyName.NORMAL).view(mu=0)
    expon = family(FamilyName.EXPONENTIAL)
    table = make_table([4.0, 5.0, 6.0, -1.0, 5.0], [0, 0, 0, 1, 1], [normal, expon])

    result = HungarianMatcher().match(table)

    np.testing.assert_array_equal(result, [1, 0])


def test_identical_normals_are_ordered_by_cluster_mean_after_exponential_gets_positive_cluster() -> None:
    normal = family(FamilyName.NORMAL)
    expon = family(FamilyName.EXPONENTIAL)
    table = make_table(
        [5.0, 6.0, 7.0, 8.0, 9.0, -5.0, -4.0, -3.0, -2.0, -13.5, -12.5, -11.5, -10.5, -9.5, -8.5, -7.5],
        [0] * 5 + [1] * 4 + [2] * 7,
        [normal, normal, expon],
    )

    result = HungarianMatcher().match(table)

    np.testing.assert_array_equal(result, [2, 1, 0])


def test_single_group_gets_clusters_by_mean_despite_impossible_pairs() -> None:
    expon = family(FamilyName.EXPONENTIAL)
    table = make_table([20.0, 21.0, 22.0, 23.0, -1.0, 5.0, 6.0, 7.0], [0] * 4 + [1] * 4, [expon, expon])

    result = HungarianMatcher().match(table)

    np.testing.assert_array_equal(result, [1, 0])


@pytest.mark.parametrize(
    ("values", "labels", "make_families", "pairs"),
    [
        pytest.param(
            [-9.0, -8.0, -7.0, -6.0, -3.0, 1.0, 2.0, 3.0],
            [0] * 4 + [1] * 4,
            [lambda: family(FamilyName.EXPONENTIAL), lambda: family(FamilyName.NORMAL)],
            [(0, 0), (0, 1)],
            id="one-exponential",
        ),
        pytest.param(
            [-9.0, -8.0, -7.0, -6.0, -3.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0],
            [0] * 4 + [1] * 4 + [2] * 4,
            [
                lambda: family(FamilyName.EXPONENTIAL),
                lambda: family(FamilyName.EXPONENTIAL),
                lambda: family(FamilyName.NORMAL),
            ],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            id="two-identical-exponentials",
        ),
    ],
)
def test_no_assignment_error_lists_impossible_pairs_with_reasons_of_same_table(
    values: list[float],
    labels: list[int],
    make_families: list[Callable[[], ParametricFamily]],
    pairs: list[tuple[int, int]],
) -> None:
    families = [make_family() for make_family in make_families]
    table = make_table(values, labels, families)
    failure_lines = []
    for j, c in pairs:
        failure = table.failure(j, c)
        assert failure is not None
        assert failure.startswith(f"family {j} (Exponential), cluster {c}: ")
        failure_lines.append(f"\n{failure}")
    expected = ERROR_HEADER + "".join(failure_lines)

    with pytest.raises(InitializationError, match=rf"^{re.escape(expected)}$"):
        HungarianMatcher().match(table)


@settings(deadline=None)
@given(
    inputs=partition_inputs(),
    family_indices=st.lists(st.integers(0, len(FACTORIES) - 1), min_size=MAX_CLUSTERS, max_size=MAX_CLUSTERS),
)
def test_random_partitions_give_optimal_assignment_or_initialization_error(
    inputs: tuple[UnivariateFloatArray, int, int], family_indices: list[int]
) -> None:
    X, k, seed = inputs
    H = random_partition(len(X), k, seed)
    families = [FACTORIES[index]() for index in family_indices[:k]]
    table = LazyComponentTable(X, H, families, MLEComponentEstimator())
    loglikelihoods = np.array([[table.loglikelihood(j, c) for c in range(k)] for j in range(k)])

    groups = table.groups()
    means = table.cluster_means()
    if len(groups) == 1:
        result = HungarianMatcher().match(table)

        np.testing.assert_array_equal(result, sorted(range(k), key=lambda c: (means[c], c)))
        return

    best = max(sum(loglikelihoods[j, permutation[j]] for j in range(k)) for permutation in permutations(range(k)))
    if best == -np.inf:
        with pytest.raises(InitializationError, match=rf"^{re.escape(ERROR_HEADER)}\n"):
            HungarianMatcher().match(table)
        return

    result = HungarianMatcher().match(table)

    np.testing.assert_array_equal(np.sort(result), np.arange(k))
    assert all(np.isfinite(loglikelihoods[j, result[j]]) for j in range(k))
    total = sum(loglikelihoods[j, result[j]] for j in range(k))
    assert total == pytest.approx(best, rel=1e-9, abs=1e-9)
