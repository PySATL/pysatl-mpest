"""
Contract of a partitioning method: how a sample is divided into clusters.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

import numpy as np

from ..typings import MultivariateFloatArray, UnivariateFloatArray


class Partitioner(ABC):
    """Divides a sample into ``k`` clusters while the component parameters are still unknown.

    A partitioning method knows nothing about the families of the mixture, and the partition is hard:
    every point belongs to exactly one cluster. A method that needs randomness takes it only from the
    ``rng`` argument and does not store a generator in the object; a deterministic method ignores
    ``rng``.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        partition
    """

    @abstractmethod
    def partition(self, X: UnivariateFloatArray, k: int, rng: np.random.Generator) -> MultivariateFloatArray:
        """Divide the sample into ``k`` clusters.

        The order of the columns is arbitrary: the number of a cluster means nothing until a matcher
        assigns the cluster to a family. The caller checks the structure of the result: a wrong
        structure is a :class:`ValueError`, an empty cluster an :class:`~.InitializationError`.

        Parameters
        ----------
        X : UnivariateFloatArray
            One-dimensional sample of finite values; the caller ensures that it has at least ``k`` points.
        k : int
            Number of clusters.
        rng : numpy.random.Generator
            The only source of randomness of the method.

        Returns
        -------
        MultivariateFloatArray
            Matrix of shape ``(n, k)`` with exactly one 1 in every row, in the column of the cluster of
            that point. Its elements may also be ``bool`` or integer; the caller converts them to
            ``float64``.

        Raises
        ------
        InitializationError
            If the method itself finds the partition unlucky, for example a cluster without points.
            Another partition may succeed.
        """
