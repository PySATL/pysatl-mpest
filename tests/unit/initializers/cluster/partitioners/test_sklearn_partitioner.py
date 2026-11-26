"""Tests of partitioning by a scikit-learn clusterer."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import sys
import warnings
from collections.abc import Callable
from typing import ClassVar

import numpy as np
import pytest
from pysatl_mpest.initializers.cluster.partitioners.sklearn_partitioner import SklearnPartitioner
from pysatl_mpest.typings import UnivariateFloatArray
from sklearn.base import BaseEstimator, ClusterMixin

SCRIPTED_ERROR = RuntimeError("scripted failure")
IGNORED_TAIL = r": SklearnPartitioner sets its own value for every partition\.$"


class ScriptedClusterer(ClusterMixin, BaseEstimator):
    """Clusterer double: remembers every fitted copy and returns scripted labels."""

    fitted: ClassVar[list["ScriptedClusterer"]] = []

    def __init__(self, n_clusters: int = 2, random_state: int | None = None, labels: list[int] | None = None) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.labels = labels

    def fit(self, X: np.ndarray, y: None = None) -> "ScriptedClusterer":
        type(self).fitted.append(self)
        self.seen_shape_ = X.shape
        self.labels_ = np.arange(len(X)) % self.n_clusters if self.labels is None else np.asarray(self.labels)
        return self


class RoundRobinClusterer(ClusterMixin, BaseEstimator):
    """Clusterer double without ``random_state``: assigns labels round-robin."""

    def __init__(self, n_clusters: int = 2) -> None:
        self.n_clusters = n_clusters

    def fit(self, X: np.ndarray, y: None = None) -> "RoundRobinClusterer":
        self.labels_ = np.arange(len(X)) % self.n_clusters
        return self


class RequiredNClustersClusterer(ClusterMixin, BaseEstimator):
    """Clusterer double whose ``n_clusters`` parameter has no default value."""

    def __init__(self, n_clusters: int) -> None:
        self.n_clusters = n_clusters


class EpsOnlyClusterer(ClusterMixin, BaseEstimator):
    """Clusterer double whose only parameter is ``eps``."""

    def __init__(self, eps: float = 0.5) -> None:
        self.eps = eps


class FailingClusterer(ClusterMixin, BaseEstimator):
    """Clusterer double whose fitting always raises the scripted error."""

    def __init__(self, n_clusters: int = 2) -> None:
        self.n_clusters = n_clusters

    def fit(self, X: np.ndarray, y: None = None) -> "FailingClusterer":
        raise SCRIPTED_ERROR


@pytest.fixture
def fitted_copies() -> list[ScriptedClusterer]:
    """Return the list of fitted scripted copies after clearing it."""

    ScriptedClusterer.fitted.clear()
    return ScriptedClusterer.fitted


@pytest.fixture
def sample() -> UnivariateFloatArray:
    """Read-only sample of seven equally spaced points."""

    values = np.arange(7.0)
    values.flags.writeable = False
    return values


class TestCreation:
    def test_requires_scikit_learn_with_install_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clusterer = ScriptedClusterer()
        monkeypatch.setitem(sys.modules, "sklearn", None)
        monkeypatch.setitem(sys.modules, "sklearn.base", None)

        with pytest.raises(
            ImportError,
            match=(
                r"^SklearnPartitioner requires scikit-learn; install it with 'pip install "
                r"pysatl-mpest\[sklearn\]'\.$"
            ),
        ) as excinfo:
            SklearnPartitioner(clusterer)

        assert isinstance(excinfo.value.__cause__, ImportError)

    @pytest.mark.parametrize(
        ("make_clusterer", "type_name"),
        [(lambda: object(), "object"), (lambda: ScriptedClusterer, "type")],
        ids=["plain-object", "class-instead-of-instance"],
    )
    def test_rejects_object_that_is_not_clusterer(self, make_clusterer: Callable[[], object], type_name: str) -> None:
        with pytest.raises(
            TypeError,
            match=rf"^clusterer must be an instance of sklearn\.base\.ClusterMixin, got {type_name}\.$",
        ):
            SklearnPartitioner(make_clusterer())

    def test_rejects_clusterer_without_number_of_clusters(self) -> None:
        with pytest.raises(
            ValueError,
            match=(
                r"^EpsOnlyClusterer has no n_clusters parameter: "
                r"SklearnPartitioner supports only clusterers that take the number of clusters\.$"
            ),
        ):
            SklearnPartitioner(EpsOnlyClusterer())

    def test_stores_passed_clusterer(self) -> None:
        user = ScriptedClusterer()

        assert SklearnPartitioner(user).clusterer is user


class TestReplacedParameters:
    def test_non_default_number_of_clusters_warns_once(self) -> None:
        with pytest.warns(
            UserWarning,
            match=r"^n_clusters=5 of the clusterer is ignored" + IGNORED_TAIL,
        ) as record:
            SklearnPartitioner(ScriptedClusterer(n_clusters=5))

        assert len(record) == 1
        assert record[0].category is UserWarning

    def test_non_default_random_state_warns_once(self) -> None:
        with pytest.warns(
            UserWarning,
            match=r"^random_state=0 of the clusterer is ignored" + IGNORED_TAIL,
        ) as record:
            SklearnPartitioner(ScriptedClusterer(random_state=0))

        assert len(record) == 1
        assert record[0].category is UserWarning

    def test_two_replaced_values_warn_once_each(self) -> None:
        with pytest.warns(UserWarning, match=r"of the clusterer is ignored") as record:
            SklearnPartitioner(ScriptedClusterer(n_clusters=5, random_state=0))

        assert len(record) == 2  # noqa: PLR2004
        assert {str(w.message).split()[0] for w in record} == {"n_clusters=5", "random_state=0"}

    @pytest.mark.parametrize(
        "make_clusterer",
        [
            lambda: ScriptedClusterer(),
            lambda: ScriptedClusterer(n_clusters=2, random_state=None),
            lambda: RoundRobinClusterer(),
            lambda: RequiredNClustersClusterer(n_clusters=4),
        ],
        ids=["defaults", "explicit-defaults", "no-random-state", "n-clusters-without-default"],
    )
    def test_default_values_do_not_warn(self, make_clusterer: Callable[[], object]) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SklearnPartitioner(make_clusterer())


class TestPartition:
    def test_copy_receives_clusters_seed_and_column_sample(
        self, sample: UnivariateFloatArray, fitted_copies: list[ScriptedClusterer]
    ) -> None:
        user = ScriptedClusterer()

        SklearnPartitioner(user).partition(sample, 3, np.random.default_rng(5))

        assert len(fitted_copies) == 1
        copy = fitted_copies[0]
        assert copy is not user
        assert copy.n_clusters == 3  # noqa: PLR2004
        assert type(copy.random_state) is int
        assert 0 <= copy.random_state < 2**32
        assert copy.seen_shape_ == (7, 1)
        assert user.get_params() == {"n_clusters": 2, "random_state": None, "labels": None}
        assert not hasattr(user, "labels_")

    def test_same_generator_state_gives_same_seed(
        self, sample: UnivariateFloatArray, fitted_copies: list[ScriptedClusterer]
    ) -> None:
        partitioner = SklearnPartitioner(ScriptedClusterer())

        partitioner.partition(sample, 3, np.random.default_rng(5))
        partitioner.partition(sample, 3, np.random.default_rng(5))

        assert len(fitted_copies) == 2  # noqa: PLR2004
        first, second = fitted_copies
        assert first.random_state == second.random_state

    def test_successive_calls_draw_different_seeds(
        self, sample: UnivariateFloatArray, fitted_copies: list[ScriptedClusterer]
    ) -> None:
        partitioner = SklearnPartitioner(ScriptedClusterer())
        rng = np.random.default_rng(0)

        for _ in range(5):
            partitioner.partition(sample, 3, rng)

        seeds = [copy.random_state for copy in fitted_copies]
        assert len(seeds) == 5  # noqa: PLR2004
        assert len(set(seeds)) == 5  # noqa: PLR2004

    def test_clusterer_without_random_state_divides_sample(self, sample: UnivariateFloatArray) -> None:
        H = SklearnPartitioner(RoundRobinClusterer()).partition(sample, 3, np.random.default_rng(0))

        assert H.dtype == np.float64
        np.testing.assert_array_equal(H, np.eye(3)[np.arange(7) % 3])

    def test_labels_become_partition_matrix(self, fitted_copies: list[ScriptedClusterer]) -> None:
        sample = np.arange(4.0)
        sample.flags.writeable = False

        H = SklearnPartitioner(ScriptedClusterer(labels=[1, 1, 0, 2])).partition(sample, 3, np.random.default_rng(0))

        assert H.dtype == np.float64
        np.testing.assert_array_equal(H, [[0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])

    def test_label_outside_range_leaves_row_without_one(self, fitted_copies: list[ScriptedClusterer]) -> None:
        sample = np.arange(4.0)
        sample.flags.writeable = False

        H = SklearnPartitioner(ScriptedClusterer(labels=[0, -1, 1, 0])).partition(sample, 2, np.random.default_rng(0))

        np.testing.assert_array_equal(H, [[1, 0], [0, 0], [0, 1], [1, 0]])

    def test_clusterer_error_passes_through(self, sample: UnivariateFloatArray) -> None:
        with pytest.raises(RuntimeError, match="^scripted failure$") as excinfo:
            SklearnPartitioner(FailingClusterer()).partition(sample, 2, np.random.default_rng(0))

        assert excinfo.value is SCRIPTED_ERROR
