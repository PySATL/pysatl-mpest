"""Hypothesis strategy and a random hard partition for property tests of the initializers."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from hypothesis import strategies as st
from pysatl_mpest.typings import MultivariateFloatArray, UnivariateFloatArray

MAX_POINTS = 30
MAX_CLUSTERS = 5
GRID_STEP = 0.25
GRID_HALF_WIDTH = 20
LARGEST_VALUE = 1e6
LARGEST_SEED = 2**32 - 1


@st.composite
def partition_inputs(draw: st.DrawFn, *, distinct_values: bool = False) -> tuple[UnivariateFloatArray, int, int]:
    """Draw a sample, a number of clusters and a seed for a partitioning method.

    The sample is a read-only ``float64`` array of 1 to ``MAX_POINTS`` points. Each point is either a node of
    a grid with step ``GRID_STEP``, so that repeated values and equal distances to centers are frequent, or a
    finite real value from ``[-LARGEST_VALUE, LARGEST_VALUE]``. The number of clusters is from 1 to
    ``MAX_CLUSTERS`` and does not exceed the number of points, or the number of distinct values when
    ``distinct_values`` is true. The test creates the generator itself: ``np.random.default_rng(seed)``.

    Returns
    -------
    tuple[UnivariateFloatArray, int, int]
        The sample ``X``, the number of clusters ``k`` and the seed.
    """

    grid_value = st.integers(-GRID_HALF_WIDTH, GRID_HALF_WIDTH).map(lambda step: step * GRID_STEP)
    real_value = st.floats(-LARGEST_VALUE, LARGEST_VALUE, allow_nan=False, allow_infinity=False)
    X = np.array(draw(st.lists(grid_value | real_value, min_size=1, max_size=MAX_POINTS)), dtype=np.float64)
    X.flags.writeable = False

    available = len(np.unique(X)) if distinct_values else len(X)
    k = draw(st.integers(1, min(MAX_CLUSTERS, available)))
    seed = draw(st.integers(0, LARGEST_SEED))
    return X, k, seed


def random_partition(n: int, k: int, seed: int) -> MultivariateFloatArray:
    """Make a random hard partition of ``n`` points into ``k`` clusters, none of them empty.

    The result is a read-only ``float64`` matrix of shape ``(n, k)`` with a single one in each row. Every
    cluster gets at least one point because a pair of a family and a cluster is evaluated only on a non-empty
    cluster. The same arguments give the same partition. Requires ``n >= k >= 1``; the arguments are not
    checked.
    """

    rng = np.random.default_rng(seed)
    labels = rng.permutation(np.concatenate([np.arange(k), rng.integers(k, size=n - k)]))
    H = np.eye(k)[labels]
    H.flags.writeable = False
    return H
