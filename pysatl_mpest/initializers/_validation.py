"""
Pure checks of the ``initializers`` package: partition structure, matcher permutation and thresholds.

A partitioning method may be user-provided, so :func:`validate_partition` turns its output into a
``float64`` matrix and separates structural errors (a bug in the method) from unlucky partitions
(empty clusters). The thresholds are separate predicates so that each boundary can be tested exactly
at the point of equality.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from numpy.typing import ArrayLike

from ..core._numerics import NUMERICAL_TOLERANCE
from ..typings import MultivariateFloatArray
from .exceptions import InitializationError

RELATIVE_WIDTH_TOLERANCE = 1e-6
"""Fraction of the sample spread at or below which the interquartile width of a component counts as degenerate."""


def validate_partition(H: ArrayLike, n: int, k: int) -> MultivariateFloatArray:
    """Check that a partitioning method returned a hard partition and convert it to ``float64``.

    A hard partition of ``n`` points into ``k`` clusters is an ``(n, k)`` matrix where row ``i``
    holds a single 1 in the column of the cluster that point ``i`` belongs to and zeros elsewhere.

    Parameters
    ----------
    H : ArrayLike
        Partition matrix with elements of type ``bool``, integer or real.
    n : int
        Number of points in the sample.
    k : int
        Number of clusters.

    Returns
    -------
    MultivariateFloatArray
        The same partition as a ``float64`` matrix; row and column order is preserved.

    Raises
    ------
    ValueError
        The output has the wrong shape, values other than 0 and 1, or a row without exactly one 1.
        This is a bug in the partitioning method; another partition will not fix it.
    InitializationError
        A cluster has no points. This partition is unlucky; another partition may succeed.
    """

    matrix = np.asarray(H)
    if matrix.shape != (n, k):
        raise ValueError(f"A partition of {n} points into {k} clusters must have shape ({n}, {k}), got {matrix.shape}.")
    if not np.isin(matrix, [0, 1]).all():
        raise ValueError("A partition matrix must contain only zeros and ones.")
    if not (matrix.sum(axis=1) == 1).all():
        raise ValueError("Every row of a partition matrix must contain exactly one 1.")

    empty_clusters = np.flatnonzero(matrix.sum(axis=0) == 0).tolist()
    if empty_clusters:
        raise InitializationError(f"The partition has empty clusters: {empty_clusters}.")

    return np.asarray(matrix, dtype=np.float64)


def validate_permutation(assignment: ArrayLike, k: int) -> None:
    """Check that a matcher returned a permutation of cluster indices ``0…k-1``.

    The matcher assigns clusters to families: family ``j`` receives cluster ``assignment[j]``.

    Parameters
    ----------
    assignment : ArrayLike
        One-dimensional integer array of length ``k``.
    k : int
        Number of clusters, equal to the number of families.

    Raises
    ------
    ValueError
        The value is not a permutation of ``0…k-1``; the message contains the received value.
    """

    received = np.asarray(assignment)
    is_permutation = (
        received.ndim == 1 and np.issubdtype(received.dtype, np.integer) and sorted(received.tolist()) == list(range(k))
    )
    if not is_permutation:
        raise ValueError(f"The matcher must return a permutation of length {k}, got {received.tolist()}.")


def cluster_is_large_enough(size: int, min_cluster_size: int) -> bool:
    """Whether a cluster has at least ``min_cluster_size`` points (non-strict)."""

    return size >= min_cluster_size


def has_enough_distinct_values(n_distinct: int, n_free_parameters: int) -> bool:
    """Whether a cluster has at least as many distinct values as the family has free parameters."""

    return n_distinct >= n_free_parameters


def component_is_wide_enough(width: float, scale: float) -> bool:
    """Whether the interquartile width of a component exceeds ``RELATIVE_WIDTH_TOLERANCE`` of ``scale`` (strict).

    ``scale`` is the spread of the whole sample.
    """

    return width > RELATIVE_WIDTH_TOLERANCE * scale


def exceeds_m_step_tolerance(value: float) -> bool:
    """Whether ``value`` exceeds the absolute M-step tolerance (strict).

    Temporary check until the M-step gets a relative tolerance (issue #51): M-step strategies only
    update a parameter when the quantity is above ``NUMERICAL_TOLERANCE``, so a start below the
    threshold would stay stuck.
    """

    return value > NUMERICAL_TOLERANCE
