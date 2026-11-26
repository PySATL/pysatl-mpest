"""
Initializers build a starting :class:`~pysatl_mpest.core.MixtureModel` from a sample and a list of component families.

The package is under development: it exports the base :class:`Initializer`, the role contracts :class:`Partitioner`,
:class:`ComponentEstimator` and :class:`Matcher`, the protocol :class:`ComponentTable`, the errors and the
partitioning methods :class:`KMeansPartitioner`, :class:`QuantilePartitioner` and :class:`RandomCentersPartitioner`.
Ready component estimators, matchers and initializers come later.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .cluster import (
    ComponentEstimator,
    ComponentTable,
    KMeansPartitioner,
    Matcher,
    Partitioner,
    QuantilePartitioner,
    RandomCentersPartitioner,
)
from .exceptions import ComponentEstimationError, InitializationError
from .initializer import Initializer

__all__ = [
    "ComponentEstimationError",
    "ComponentEstimator",
    "ComponentTable",
    "InitializationError",
    "Initializer",
    "KMeansPartitioner",
    "Matcher",
    "Partitioner",
    "QuantilePartitioner",
    "RandomCentersPartitioner",
]
