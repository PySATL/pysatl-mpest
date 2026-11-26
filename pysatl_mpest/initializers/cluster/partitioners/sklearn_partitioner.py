"""Partitioning by a scikit-learn clusterer.

scikit-learn is optional and is imported only when the partitioner is created.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import inspect
import warnings
from typing import TYPE_CHECKING

import numpy as np

from ....typings import MultivariateFloatArray, UnivariateFloatArray
from ._labels import one_hot
from .base import Partitioner

if TYPE_CHECKING:
    from sklearn.base import ClusterMixin

_SEED_BOUND = 2**32


class SklearnPartitioner(Partitioner):
    """Partitions a sample by a scikit-learn clusterer with an ``n_clusters`` parameter.

    Every partition is computed by an unfitted copy of the clusterer. The copy receives ``n_clusters=k`` and,
    if the clusterer has a ``random_state`` parameter, an integer seed drawn from the generator of the call.
    The object passed by the user is neither changed nor fitted. scikit-learn is an optional dependency and is
    required to create this partitioner (``pip install pysatl-mpest[sklearn]``).

    Parameters
    ----------
    clusterer : ClusterMixin
        Clusterer inheriting ``ClusterMixin`` and taking the number of clusters as its ``n_clusters`` parameter.
        Its values of ``n_clusters`` and ``random_state`` are replaced for every partition.

    Attributes
    ----------
    clusterer : ClusterMixin
        The clusterer passed by the user, unchanged.

    Raises
    ------
    ImportError
        scikit-learn is not installed.
    TypeError
        ``clusterer`` is not an instance of ``ClusterMixin``, or it cannot label the points of a sample (it has
        no ``fit_predict``, as ``FeatureAgglomeration``, which clusters features).
    ValueError
        ``clusterer`` has no ``n_clusters`` parameter.

    Warns
    -----
    UserWarning
        The value of ``n_clusters`` or ``random_state`` differs from the default of the clusterer constructor.
        The value is ignored because the partitioner sets its own value for every partition.

    Notes
    -----
    Clusterers that choose the number of clusters themselves, such as ``DBSCAN``, are not supported. A
    deterministic clusterer, such as ``AgglomerativeClustering``, gives the same partition for any generator.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        partition
    """

    def __init__(self, clusterer: "ClusterMixin") -> None:
        try:
            from sklearn.base import ClusterMixin
        except ImportError as error:
            raise ImportError(
                "SklearnPartitioner requires scikit-learn; install it with 'pip install pysatl-mpest[sklearn]'."
            ) from error

        if not isinstance(clusterer, ClusterMixin):
            raise TypeError(
                f"clusterer must be an instance of sklearn.base.ClusterMixin, got {type(clusterer).__name__}."
            )

        if not hasattr(clusterer, "fit_predict"):
            raise TypeError(
                f"{type(clusterer).__name__} cannot label the points of a sample: "
                "SklearnPartitioner needs a clusterer with fit_predict."
            )

        params = clusterer.get_params()
        if "n_clusters" not in params:
            raise ValueError(
                f"{type(clusterer).__name__} has no n_clusters parameter: "
                "SklearnPartitioner supports only clusterers that take the number of clusters."
            )

        parameters = inspect.signature(type(clusterer).__init__).parameters
        for name in ("n_clusters", "random_state"):
            if name in params:
                value = params[name]
                default = parameters[name].default
                if default is not inspect.Parameter.empty and value != default:
                    warnings.warn(
                        f"{name}={value!r} of the clusterer is ignored: "
                        "SklearnPartitioner sets its own value for every partition.",
                        stacklevel=2,
                    )

        self.clusterer = clusterer

    def partition(self, X: UnivariateFloatArray, k: int, rng: np.random.Generator) -> MultivariateFloatArray:
        """Divide the sample into ``k`` clusters by the scikit-learn clusterer.

        The method raises no errors of its own: errors of the clusterer pass through unchanged, and a cluster
        without points gives an empty column.

        Parameters
        ----------
        X : UnivariateFloatArray
            One-dimensional sample of finite values; it is passed to the clusterer as one feature column.
        k : int
            Number of clusters.
        rng : numpy.random.Generator
            Generator from which an integer seed is drawn for a clusterer with a ``random_state`` parameter.

        Returns
        -------
        MultivariateFloatArray
            Partition matrix of shape ``(n, k)``; column ``c`` is the cluster of points with label ``c``.
        """

        from sklearn.base import clone

        copy = clone(self.clusterer)
        copy.set_params(n_clusters=k)
        if "random_state" in copy.get_params():
            seed = int(rng.integers(_SEED_BOUND))
            copy.set_params(random_state=seed)

        labels = copy.fit_predict(X[:, np.newaxis])
        return one_hot(labels, k)
