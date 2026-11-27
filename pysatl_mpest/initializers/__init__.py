"""
Initializers build a starting :class:`~pysatl_mpest.core.MixtureModel` from a sample and a list of component families.

The package is under development: it exports the base :class:`Initializer`, the cluster initializer
:class:`ClusterInitializer` with its role contracts :class:`Partitioner`, :class:`ComponentEstimator` and
:class:`Matcher` and the protocol :class:`ComponentTable`, the errors, the partitioning methods
:class:`KMeansPartitioner`, :class:`QuantilePartitioner`, :class:`RandomCentersPartitioner` and
:class:`SklearnPartitioner`, the component estimator :class:`MLEComponentEstimator`, and the matchers
:class:`HungarianMatcher` and :class:`IdentityMatcher`. :class:`SklearnPartitioner` needs scikit-learn, an optional
dependency installed with ``pip install pysatl-mpest[sklearn]``; the rest of the package works without it. A
multistart and a refining initializer come later.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .cluster import (
    ComponentEstimator,
    ComponentTable,
    HungarianMatcher,
    IdentityMatcher,
    KMeansPartitioner,
    Matcher,
    MLEComponentEstimator,
    Partitioner,
    QuantilePartitioner,
    RandomCentersPartitioner,
    SklearnPartitioner,
)
from .cluster_initializer import ClusterInitializer
from .exceptions import ComponentEstimationError, InitializationError
from .initializer import Initializer

__all__ = [
    "ClusterInitializer",
    "ComponentEstimationError",
    "ComponentEstimator",
    "ComponentTable",
    "HungarianMatcher",
    "IdentityMatcher",
    "InitializationError",
    "Initializer",
    "KMeansPartitioner",
    "MLEComponentEstimator",
    "Matcher",
    "Partitioner",
    "QuantilePartitioner",
    "RandomCentersPartitioner",
    "SklearnPartitioner",
]
