"""Integration tests of partitioning by real scikit-learn clusterers."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import subprocess
import sys
import warnings
from collections.abc import Callable
from functools import partial

import numpy as np
import pytest
from hypothesis import assume, given, settings
from pysatl_mpest.initializers.cluster._table._validation import validate_partition
from pysatl_mpest.initializers.cluster.partitioners.sklearn_partitioner import SklearnPartitioner
from pysatl_mpest.initializers.exceptions import InitializationError
from pysatl_mpest.typings import UnivariateFloatArray
from sklearn.base import ClusterMixin
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture

from tests.helpers.partitions import partition_inputs

IMPORT_SCRIPT = """
import sys

import pysatl_mpest.initializers
import pysatl_mpest.initializers.cluster.partitioners.sklearn_partitioner

loaded = sorted(name for name in sys.modules if name.split(".")[0] == "sklearn")
assert not loaded, loaded
"""

WITHOUT_SKLEARN_SCRIPT = """
import sys

sys.modules["sklearn"] = None

import numpy as np
from pysatl_mpest.initializers import KMeansPartitioner
from pysatl_mpest.initializers.cluster.partitioners.sklearn_partitioner import SklearnPartitioner

H = KMeansPartitioner().partition(np.array([0.0, 1.0, 5.0, 6.0]), 2, np.random.default_rng(0))
assert H.shape == (4, 2), H.shape

try:
    SklearnPartitioner(object())
except ImportError as error:
    assert "pysatl-mpest[sklearn]" in str(error), error
else:
    raise AssertionError("SklearnPartitioner was created without scikit-learn")
"""


def three_distant_groups() -> UnivariateFloatArray:
    """Read-only sample of three distant normal groups, sixty points each."""

    rng = np.random.default_rng(42)
    sample = np.concatenate([rng.normal(0.0, 1.0, 60), rng.normal(10.0, 1.0, 60), rng.normal(20.0, 1.0, 60)])
    sample.flags.writeable = False
    return sample


@pytest.mark.parametrize(
    "clusterer_class",
    [KMeans, AgglomerativeClustering, MiniBatchKMeans],
    ids=["kmeans", "agglomerative", "minibatch"],
)
def test_default_clusterer_divides_three_distant_groups_without_warnings(
    clusterer_class: type[ClusterMixin],
) -> None:
    sample = three_distant_groups()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        partitioner = SklearnPartitioner(clusterer_class())
        H = validate_partition(partitioner.partition(sample, 3, np.random.default_rng(1)), n=180, k=3)

    assert H.sum(axis=0).tolist() == [60, 60, 60]
    for group in (slice(0, 60), slice(60, 120), slice(120, 180)):
        assert len(np.unique(H[group].argmax(axis=1))) == 1


@pytest.mark.parametrize(
    "clusterer_class",
    [KMeans, AgglomerativeClustering, MiniBatchKMeans],
    ids=["kmeans", "agglomerative", "minibatch"],
)
def test_user_clusterer_is_unchanged_and_one_partitioner_works_with_different_k(
    clusterer_class: type[ClusterMixin],
) -> None:
    sample = three_distant_groups()
    clusterer = clusterer_class()
    before = clusterer.get_params()
    partitioner = SklearnPartitioner(clusterer)

    partitioner.partition(sample, 3, np.random.default_rng(1))
    H = partitioner.partition(sample, 2, np.random.default_rng(1))

    validate_partition(H, n=180, k=2)
    assert clusterer.get_params() == before
    assert not hasattr(clusterer, "labels_")


@pytest.mark.parametrize(
    "clusterer_class",
    [KMeans, AgglomerativeClustering, MiniBatchKMeans],
    ids=["kmeans", "agglomerative", "minibatch"],
)
def test_same_seed_gives_same_matrix(clusterer_class: type[ClusterMixin]) -> None:
    sample = three_distant_groups()
    partitioner = SklearnPartitioner(clusterer_class())

    first = partitioner.partition(sample, 3, np.random.default_rng(1))
    second = partitioner.partition(sample, 3, np.random.default_rng(1))

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("make_clusterer", "pattern"),
    [
        (partial(KMeans, n_clusters=5), r"^n_clusters=5 of the clusterer is ignored"),
        (partial(MiniBatchKMeans, random_state=0), r"^random_state=0 of the clusterer is ignored"),
        (partial(AgglomerativeClustering, n_clusters=3), r"^n_clusters=3 of the clusterer is ignored"),
    ],
    ids=["kmeans-n-clusters", "minibatch-random-state", "agglomerative-n-clusters"],
)
def test_non_default_value_warns(make_clusterer: Callable[[], ClusterMixin], pattern: str) -> None:
    with pytest.warns(UserWarning, match=pattern):
        SklearnPartitioner(make_clusterer())


def test_rejects_clusterer_without_number_of_clusters() -> None:
    with pytest.raises(ValueError, match=r"^DBSCAN has no n_clusters parameter"):
        SklearnPartitioner(DBSCAN())


def test_rejects_estimator_that_is_not_clusterer() -> None:
    with pytest.raises(TypeError, match=r"got GaussianMixture\.$"):
        SklearnPartitioner(GaussianMixture())


@pytest.mark.parametrize(
    "clusterer_class",
    [KMeans, AgglomerativeClustering, MiniBatchKMeans],
    ids=["kmeans", "agglomerative", "minibatch"],
)
@settings(deadline=None, max_examples=40)
@given(inputs=partition_inputs())
def test_result_is_valid_partition_or_has_empty_cluster(
    clusterer_class: type[ClusterMixin], inputs: tuple[UnivariateFloatArray, int, int]
) -> None:
    X, k, seed = inputs
    assume(len(X) >= 2)  # noqa: PLR2004

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        H = SklearnPartitioner(clusterer_class()).partition(X, k, np.random.default_rng(seed))

    try:
        validate_partition(H, n=len(X), k=k)
    except InitializationError:
        return


def test_importing_package_does_not_import_scikit_learn() -> None:
    result = subprocess.run([sys.executable, "-c", IMPORT_SCRIPT], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


def test_package_works_without_scikit_learn_and_partitioner_reports_it() -> None:
    result = subprocess.run([sys.executable, "-c", WITHOUT_SKLEARN_SCRIPT], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
