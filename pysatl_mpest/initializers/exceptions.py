"""
Errors of the initializers.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


class InitializationError(Exception):
    """An initializer could not build a starting mixture from this partition of the sample.

    Another partition may succeed, so a multistart skips a run that raises this error. It does not
    inherit :class:`ValueError`, so that ``except ValueError``, which catches errors of the input and
    the configuration, does not catch unlucky partitions.
    """


class ComponentEstimationError(InitializationError):
    """The parameters of a component cannot be estimated from a cluster.

    The cluster has too little data, or the estimate is invalid or degenerate. When clusters are
    matched to families, such a pair of a family and a cluster becomes impossible.
    """
