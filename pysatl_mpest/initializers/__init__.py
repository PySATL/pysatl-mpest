"""
Initializers build a starting :class:`~pysatl_mpest.core.MixtureModel` from a sample and a list of component families.

The package is under development: it exports the base :class:`Initializer`, the role contracts :class:`Partitioner`,
:class:`ComponentEstimator` and :class:`Matcher`, the protocol :class:`ComponentTable` and the errors. Ready
partitioning methods, component estimators, matchers and initializers come later.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from .component_estimator import ComponentEstimator
from .component_table import ComponentTable
from .exceptions import ComponentEstimationError, InitializationError
from .initializer import Initializer
from .matcher import Matcher
from .partitioner import Partitioner

__all__ = [
    "ComponentEstimationError",
    "ComponentEstimator",
    "ComponentTable",
    "InitializationError",
    "Initializer",
    "Matcher",
    "Partitioner",
]
