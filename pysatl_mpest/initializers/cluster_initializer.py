"""Cluster initializer: partitioning, component table, matching and assembly of the starting mixture."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence

import numpy as np
from pysatl_core.families import ParametricFamily

from ..core import MixtureModel, _components
from ..typings import UnivariateFloatArray
from .cluster import ComponentEstimator, HungarianMatcher, Matcher, MLEComponentEstimator, Partitioner
from .cluster._table._lazy import LazyComponentTable
from .cluster._table._validation import cluster_is_large_enough, validate_partition, validate_permutation
from .exceptions import InitializationError
from .initializer import Initializer


class ClusterInitializer(Initializer):
    """Builds a starting mixture by clustering the sample and estimating one component per cluster.

    The initializer performs five steps: it checks the input, partitions the sample into as many clusters as
    there are families, creates a lazy component table, matches clusters to families and assembles the mixture.
    The partitioning method is required; the component estimator and the matcher are optional and default to
    new :class:`~.MLEComponentEstimator` and :class:`~.HungarianMatcher` objects for every initializer.

    ``min_cluster_size`` protects the estimation from clusters that are too small. By default it is twice the
    largest number of free parameters among the families; the value ``1`` turns the check into requiring only
    non-empty clusters. All randomness comes from the generator passed by :meth:`initialize`. The initializer
    has no state: every call creates a new component table and does not modify the roles or the threshold.

    Parameters
    ----------
    partitioner : Partitioner
        Method that divides the sample into clusters.
    component_estimator : ComponentEstimator | None, optional
        Estimator of the free parameters of one component. ``None`` creates a new
        :class:`~.MLEComponentEstimator`.
    matcher : Matcher | None, optional
        Matcher of clusters to families. ``None`` creates a new :class:`~.HungarianMatcher`.
    min_cluster_size : int | numpy.integer | None, optional
        Smallest allowed number of points in a cluster. ``None`` uses twice the largest number of free
        parameters among the families on every call.

    Attributes
    ----------
    partitioner : Partitioner
        The partitioning method, the same object that was passed to the constructor.
    component_estimator : ComponentEstimator
        The component estimator, the passed object or the created default one.
    matcher : Matcher
        The matcher, the passed object or the created default one.
    min_cluster_size : int | None
        The explicit cluster size threshold as ``int``, or ``None`` when the default is computed per call.

    Raises
    ------
    TypeError
        If a role is not an instance of the base class of that role.
    ValueError
        If ``min_cluster_size`` is not a positive integer or ``None``.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        initialize
    """

    partitioner: Partitioner
    component_estimator: ComponentEstimator
    matcher: Matcher
    min_cluster_size: int | None

    def __init__(
        self,
        partitioner: Partitioner,
        *,
        component_estimator: ComponentEstimator | None = None,
        matcher: Matcher | None = None,
        min_cluster_size: int | np.integer | None = None,
    ) -> None:
        component_estimator = MLEComponentEstimator() if component_estimator is None else component_estimator
        matcher = HungarianMatcher() if matcher is None else matcher

        if not isinstance(partitioner, Partitioner):
            raise TypeError(f"partitioner must be a Partitioner, got {type(partitioner).__name__}.")
        if not isinstance(component_estimator, ComponentEstimator):
            raise TypeError(
                f"component_estimator must be a ComponentEstimator, got {type(component_estimator).__name__}."
            )
        if not isinstance(matcher, Matcher):
            raise TypeError(f"matcher must be a Matcher, got {type(matcher).__name__}.")
        if min_cluster_size is not None and (
            isinstance(min_cluster_size, bool)
            or not isinstance(min_cluster_size, (int, np.integer))
            or min_cluster_size < 1
        ):
            raise ValueError(f"min_cluster_size must be a positive integer or None, got {min_cluster_size!r}.")

        self.partitioner = partitioner
        self.component_estimator = component_estimator
        self.matcher = matcher
        self.min_cluster_size = None if min_cluster_size is None else int(min_cluster_size)

    def _initialize(
        self, X: UnivariateFloatArray, families: Sequence[ParametricFamily], rng: np.random.Generator
    ) -> MixtureModel:
        """Build the starting mixture by the cluster scheme.

        Parameters
        ----------
        X : UnivariateFloatArray
            The checked sample.
        families : Sequence[ParametricFamily]
            One family per component of the mixture.
        rng : numpy.random.Generator
            The only source of randomness of the partitioning method.

        Returns
        -------
        MixtureModel
            The starting mixture assembled by the component table.

        Raises
        ------
        ValueError
            If the sample cannot provide clusters of the required size, has fewer distinct values than families,
            or has a non-finite or vanishing scale because of the magnitude of its values. Also if the structure
            of the partition is wrong or the matcher did not return a permutation.
        NotImplementedError
            If the component estimator does not support at least one family.
        InitializationError
            If a cluster is empty or smaller than the threshold, or no assignment gives every family a suitable
            cluster. Its subclass :class:`~.ComponentEstimationError` may be raised while the mixture is assembled.
        """

        n = len(X)
        k = len(families)
        m = (
            self.min_cluster_size
            if self.min_cluster_size is not None
            else 2 * max(len(_components.family_free_parameter_names(family)) for family in families)
        )
        if n < k * m:
            raise ValueError(
                f"X has {n} points, fewer than the {k * m} needed for {k} clusters of at least {m} points."
            )

        d = len(np.unique(X))
        if d < k:
            raise ValueError(f"X has {d} distinct values, fewer than the {k} families.")

        with np.errstate(all="ignore"):
            scale = np.std(X)
        if not np.isfinite(scale):
            raise ValueError("The standard deviation of X is not finite: the values are too large. Rescale the data.")
        if scale == 0 and d > 1:
            raise ValueError(
                f"The standard deviation of X is zero although X has {d} distinct values: "
                "the values are too close. Rescale the data."
            )

        unsupported = [j for j, family in enumerate(families) if not self.component_estimator.supports(family)]
        if unsupported:
            items = ", ".join(_family_item(families, j) for j in unsupported)
            raise NotImplementedError(f"{type(self.component_estimator).__name__} does not support {items}.")

        H = validate_partition(self.partitioner.partition(X, k, rng), n, k)
        sizes = np.count_nonzero(H, axis=0)
        small_clusters = [c for c, size in enumerate(sizes) if not cluster_is_large_enough(size, m)]
        if small_clusters:
            items = ", ".join(f"cluster {c} has {sizes[c]}" for c in small_clusters)
            raise InitializationError(f"The partition has clusters with fewer than {m} points: {items}.")

        table = LazyComponentTable(X, H, families, self.component_estimator)
        assignment = self.matcher.match(table)
        validate_permutation(assignment, k)
        return table.mixture(np.asarray(assignment))


def _family_item(families: Sequence[ParametricFamily], j: int) -> str:
    """Human-readable item of the family at index ``j`` for the unsupported-family error."""

    name, parametrization, _ = _components.family_key(families[j])
    return f"families[{j}] ({name} in parametrization {parametrization})"
