"""Partitioning by random centers drawn from the sample."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np

from ....typings import MultivariateFloatArray, UnivariateFloatArray
from ._labels import one_hot
from .base import Partitioner


class RandomCentersPartitioner(Partitioner):
    """Draws centers from the distinct values of the sample and assigns points to the nearest center.

    Without replacement, ``k`` distinct values of the sample become the centers. Every point belongs to the
    center nearest by absolute distance; when distances are equal, the center with the smaller column number
    is chosen. Unlike k-means, the centers are not refined. Every center is itself a point of the sample and
    lies in its own cluster, so no cluster is empty. The method is a cheap source of different partitions for
    a multistart.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        partition
    """

    def partition(self, X: UnivariateFloatArray, k: int, rng: np.random.Generator) -> MultivariateFloatArray:
        """Divide the sample between randomly drawn centers.

        Parameters
        ----------
        X : UnivariateFloatArray
            One-dimensional sample of finite values; the caller ensures that it has at least ``k`` distinct values.
        k : int
            Number of clusters.
        rng : numpy.random.Generator
            Generator from which the centers are drawn.

        Returns
        -------
        MultivariateFloatArray
            Partition matrix of shape ``(n, k)``; column ``j`` is the cluster of the ``j``-th drawn center.
        """

        centers = rng.choice(np.unique(X), size=k, replace=False)
        labels = np.abs(X[:, np.newaxis] - centers).argmin(axis=1)

        return one_hot(labels, k)
