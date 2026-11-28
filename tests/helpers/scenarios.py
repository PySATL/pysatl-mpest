"""
Samples, partitioners and estimators shared by the end-to-end scenarios of the initializers.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable

import numpy as np
from pysatl_mpest.estimators import ECM, ECME, EM
from pysatl_mpest.estimators.iterative import StepBreakpointer
from pysatl_mpest.initializers import (
    KMeansPartitioner,
    Partitioner,
    QuantilePartitioner,
    RandomCentersPartitioner,
    SklearnPartitioner,
)
from pysatl_mpest.optimizers import ScipyNelderMead
from pysatl_mpest.typings import UnivariateFloatArray
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering


def example_sample() -> UnivariateFloatArray:
    """The sample of the design's task section: two normal components and one exponential component."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(-8, 1, 200), rng.normal(-3, 0.7, 200), rng.exponential(2, 300)])


def make_kmeans_partitioner() -> Partitioner:
    """The scipy k-means partitioning method."""

    return KMeansPartitioner()


def make_quantile_partitioner() -> Partitioner:
    """The quantile partitioning method."""

    return QuantilePartitioner()


def make_random_centers_partitioner() -> Partitioner:
    """The random-centers partitioning method."""

    return RandomCentersPartitioner()


def make_sklearn_kmeans_partitioner() -> Partitioner:
    """The scikit-learn k-means partitioning method with one run."""

    return SklearnPartitioner(KMeans(n_init=1))


def make_sklearn_agglomerative_partitioner() -> Partitioner:
    """The scikit-learn agglomerative partitioning method."""

    return SklearnPartitioner(AgglomerativeClustering())


def make_sklearn_spectral_partitioner() -> Partitioner:
    """The scikit-learn spectral partitioning method."""

    return SklearnPartitioner(SpectralClustering())


PARTITIONER_FACTORIES: tuple[Callable[[], Partitioner], ...] = (
    make_kmeans_partitioner,
    make_quantile_partitioner,
    make_random_centers_partitioner,
    make_sklearn_kmeans_partitioner,
    make_sklearn_agglomerative_partitioner,
)
PARTITIONER_IDS = ("kmeans", "quantile", "random-centers", "sklearn-kmeans", "sklearn-agglomerative")


def make_em(steps: int) -> EM:
    """An EM estimator with a fixed number of iterations."""

    return EM([StepBreakpointer(steps)], [], ScipyNelderMead())


def make_ecm(steps: int) -> ECM:
    """An ECM estimator with a fixed number of iterations."""

    return ECM([StepBreakpointer(steps)], [], ScipyNelderMead())


def make_ecme(steps: int) -> ECME:
    """An ECME estimator with a fixed number of iterations."""

    return ECME([StepBreakpointer(steps)], [], ScipyNelderMead())
