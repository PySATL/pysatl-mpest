"""Deterministic partitioning into segments of the sorted sample with equal numbers of points."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np

from ....typings import MultivariateFloatArray, UnivariateFloatArray
from ._labels import one_hot
from .base import Partitioner


class QuantilePartitioner(Partitioner):
    """Divides the sorted sample into consecutive segments with equal numbers of points.

    The sample is sorted in stable order, and the sorted order is cut into ``k`` consecutive segments. Segment
    sizes follow ``numpy.array_split``: they differ by at most one point, and the first ``n mod k`` segments are
    one point longer. Column ``c`` of the partition is the ``c``-th segment in ascending order of the values.
    Equal values at a boundary fall into neighboring clusters; the boundary is not shifted. The method is
    deterministic and does not use the generator, so repeating it in a multistart is useless.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        partition
    """

    def partition(self, X: UnivariateFloatArray, k: int, rng: np.random.Generator) -> MultivariateFloatArray:
        """Divide the sample into consecutive segments of its sorted order.

        Parameters
        ----------
        X : UnivariateFloatArray
            One-dimensional sample of finite values; the caller ensures that it has at least ``k`` points.
        k : int
            Number of clusters.
        rng : numpy.random.Generator
            Not used.

        Returns
        -------
        MultivariateFloatArray
            Matrix of shape ``(n, k)``. Column ``c`` is the ``c``-th segment of the sorted sample in ascending
            order of the values.
        """

        order = np.argsort(X, kind="stable")
        sizes = [len(part) for part in np.array_split(order, k)]
        sorted_labels = np.repeat(np.arange(k), sizes)
        ranks = np.argsort(order)
        labels = sorted_labels[ranks]

        return one_hot(labels, k)
