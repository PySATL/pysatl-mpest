"""k-means partitioning on ``scipy.cluster.vq.kmeans2``."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from scipy.cluster.vq import ClusterError, kmeans2

from ....typings import MultivariateFloatArray, UnivariateFloatArray
from ...exceptions import InitializationError
from ._labels import one_hot
from .base import Partitioner


class KMeansPartitioner(Partitioner):
    """Partitions a sample by one run of k-means with k-means++ initial centers.

    The run is single rather than the best of several: repeated runs with different randomness are the
    responsibility of a multistart.

    Parameters
    ----------
    max_iter : int or numpy.integer, default=100
        Number of iterations. Exactly this many iterations are performed and convergence is not checked.
        The default is 100 because the default ten iterations of ``kmeans2`` are often not enough.

    Attributes
    ----------
    max_iter : int
        Number of iterations passed to ``kmeans2``.

    Raises
    ------
    ValueError
        If ``max_iter`` is not a positive integer.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        partition
    """

    def __init__(self, max_iter: int | np.integer = 100) -> None:
        if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
            raise ValueError(f"max_iter must be a positive integer, got {max_iter!r}.")

        self.max_iter = int(max_iter)

    def partition(self, X: UnivariateFloatArray, k: int, rng: np.random.Generator) -> MultivariateFloatArray:
        """Divide the sample into ``k`` clusters by one k-means run.

        Parameters
        ----------
        X : UnivariateFloatArray
            One-dimensional sample of finite values.
        k : int
            Number of clusters.
        rng : numpy.random.Generator
            Source of randomness; it is passed to ``kmeans2`` and is advanced.

        Returns
        -------
        MultivariateFloatArray
            Partition matrix of shape ``(n, k)``; the order of the columns is arbitrary.

        Raises
        ------
        InitializationError
            k-means left one of the clusters without points. Another run may succeed, except on a sample of a
            tiny scale: with a range below about ``1e-162`` the squared distances vanish and every run fails, so the
            data must be rescaled. NumPy warnings of ``kmeans2`` on such a sample are suppressed: the error explains
            the failure.
        ValueError
            The values of the sample are so large that squared distances overflow ``float64``; the data
            must be rescaled.
        """

        with np.errstate(over="ignore"):
            largest = len(X) * max(np.abs(X).max(), (X.max() - X.min()) ** 2)
        if not np.isfinite(largest):
            raise ValueError("The values of X are too large for k-means: squared distances overflow. Rescale the data.")

        try:
            with np.errstate(invalid="ignore"):
                _, labels = kmeans2(X, k, iter=self.max_iter, minit="++", missing="raise", rng=rng)
        except ClusterError as error:
            raise InitializationError(f"k-means left one of the {k} clusters without points.") from error

        return one_hot(labels, k)
