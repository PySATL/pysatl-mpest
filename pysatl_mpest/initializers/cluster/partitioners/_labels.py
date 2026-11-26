"""
Conversion of cluster labels to a partition matrix.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from numpy.typing import ArrayLike

from ....typings import MultivariateFloatArray


def one_hot(labels: ArrayLike, k: int) -> MultivariateFloatArray:
    """Turn cluster labels into a partition matrix.

    The label of a point is the number of its cluster, from ``0`` to ``k - 1``. A label outside this range
    gives a row without a 1: the caller sees it as a wrong structure of the partition (``validate_partition``).

    Parameters
    ----------
    labels : ArrayLike
        One-dimensional integer labels of the ``n`` points.
    k : int
        Number of clusters.

    Returns
    -------
    MultivariateFloatArray
        Matrix of shape ``(n, k)`` of ``float64`` with a 1 in row ``i`` and column ``labels[i]`` and zeros
        elsewhere.
    """

    return np.asarray(np.asarray(labels)[:, np.newaxis] == np.arange(k), dtype=np.float64)
