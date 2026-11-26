"""Hypothesis strategy of inputs for property tests of partitioning methods."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from hypothesis import strategies as st
from pysatl_mpest.typings import UnivariateFloatArray

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
