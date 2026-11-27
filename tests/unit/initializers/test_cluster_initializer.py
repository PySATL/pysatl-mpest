"""Unit tests for ``ClusterInitializer`` with scripted roles."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import re
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import pytest
from pysatl_core.families import ParametricFamily
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel, _components
from pysatl_mpest.initializers import cluster_initializer
from pysatl_mpest.initializers.cluster.component_estimators.base import ComponentEstimator
from pysatl_mpest.initializers.cluster.component_estimators.mle_component_estimator import MLEComponentEstimator
from pysatl_mpest.initializers.cluster.matchers.base import Matcher
from pysatl_mpest.initializers.cluster.matchers.hungarian_matcher import HungarianMatcher
from pysatl_mpest.initializers.cluster.partitioners.base import Partitioner
from pysatl_mpest.initializers.cluster.partitioners.kmeans_partitioner import KMeansPartitioner
from pysatl_mpest.initializers.cluster.table import ComponentTable
from pysatl_mpest.initializers.cluster_initializer import ClusterInitializer
from pysatl_mpest.initializers.exceptions import ComponentEstimationError, InitializationError
from pysatl_mpest.typings import MultivariateFloatArray, UnivariateFloatArray

from tests.helpers.families import family


class ScriptedPartitioner(Partitioner):
    """Partitioner double that returns a scripted matrix or raises a scripted error."""

    def __init__(self, events: list[str], result: object, error: Exception | None = None) -> None:
        self.events = events
        self.result = result
        self.error = error
        self.calls: list[tuple[UnivariateFloatArray, int, np.random.Generator]] = []

    def partition(self, X: UnivariateFloatArray, k: int, rng: np.random.Generator) -> MultivariateFloatArray:
        self.events.append("partition")
        self.calls.append((X, k, rng))
        if self.error is not None:
            raise self.error
        return cast(MultivariateFloatArray, self.result)


class ScriptedMatcher(Matcher):
    """Matcher double that returns a scripted assignment unchanged."""

    def __init__(self, events: list[str], assignment: object, error: Exception | None = None) -> None:
        self.events = events
        self.assignment = assignment
        self.error = error
        self.tables: list[ComponentTable] = []

    def _match(self, table: ComponentTable) -> Any:
        self.events.append("match")
        self.tables.append(table)
        if self.error is not None:
            raise self.error
        return self.assignment

    def _order(self, assignment: Any, table: ComponentTable) -> Any:
        return assignment


class ScriptedEstimator(ComponentEstimator):
    """Component estimator double with scripted support keys; estimation itself is never used."""

    def __init__(self, events: list[str], supported_keys: set[tuple[str, str]] | None = None) -> None:
        self.events = events
        self.supported_keys = supported_keys
        self.supports_calls: list[ParametricFamily] = []

    def supports(self, family: ParametricFamily) -> bool:
        self.events.append("supports")
        self.supports_calls.append(family)
        if self.supported_keys is None:
            return True
        name, parametrization, _ = _components.family_key(family)
        return (name, parametrization) in self.supported_keys

    def estimate(self, X: UnivariateFloatArray, h: UnivariateFloatArray, family: ParametricFamily) -> dict[str, float]:
        raise AssertionError("The initializer must not estimate components directly.")


def quack_partitioner() -> object:
    """Object with a ``partition`` method that is not a ``Partitioner``."""

    class QuackPartitioner:
        """Structural but not nominal partitioner."""

        def partition(self, X: UnivariateFloatArray, k: int, rng: np.random.Generator) -> MultivariateFloatArray:
            return np.eye(k)

    return QuackPartitioner()


@pytest.fixture
def events() -> list[str]:
    """Journal of role calls in a test."""

    return []


@pytest.fixture
def normal() -> ParametricFamily:
    """The Normal family without a view."""

    return family(FamilyName.NORMAL)


@pytest.fixture
def expon() -> ParametricFamily:
    """The Exponential family without a view."""

    return family(FamilyName.EXPONENTIAL)


@pytest.fixture
def uniform() -> ParametricFamily:
    """The continuous uniform family without a view."""

    return family(FamilyName.CONTINUOUS_UNIFORM)


def valid_mixture(families: Sequence[ParametricFamily]) -> MixtureModel:
    """A mixture from the given families that covers the positive samples of this file."""

    components = []
    for component_family in families:
        name, _, _ = _components.family_key(component_family)
        free = _components.family_free_parameter_names(component_family)
        if name == "Normal":
            parameters = {}
            if "mu" in free:
                parameters["mu"] = 4.0
            if "sigma" in free:
                parameters["sigma"] = 3.0
        elif name == "Exponential":
            parameters = {"lambda_": 0.25}
        elif name == "ContinuousUniform":
            parameters = {"lower_bound": 0.0, "upper_bound": 100.0}
        else:
            raise AssertionError(f"No scripted component for family {name}.")
        components.append(component_family.distribution(**parameters))

    return MixtureModel(components)


@pytest.fixture
def install_table(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> Callable[..., list[Any]]:
    """Install a recording double of ``LazyComponentTable`` and return the list of its instances."""

    def install(error: Exception | None = None, mixture: MixtureModel | None = None) -> list[Any]:
        created: list[Any] = []

        class RecordingTable:
            """Table double that records construction and mixture assembly."""

            def __init__(
                self,
                X: UnivariateFloatArray,
                H: MultivariateFloatArray,
                families: Sequence[ParametricFamily],
                component_estimator: ComponentEstimator,
            ) -> None:
                events.append("table")
                self.X = X
                self.H = H
                self.families = families
                self.component_estimator = component_estimator
                self.assignments: list[object] = []
                self.returned_mixture = valid_mixture(families) if mixture is None else mixture
                created.append(self)

            def mixture(self, assignment: object) -> MixtureModel:
                events.append("mixture")
                self.assignments.append(assignment)
                if error is not None:
                    raise error
                return self.returned_mixture

        monkeypatch.setattr(cluster_initializer, "LazyComponentTable", RecordingTable)
        return created

    return install


def sample(n: int) -> UnivariateFloatArray:
    """The positive sample ``1.0, ..., n`` as ``float64``."""

    return np.arange(1.0, n + 1.0, dtype=np.float64)


def partition_matrix(labels: list[int], k: int) -> MultivariateFloatArray:
    """Partition matrix built from cluster labels."""

    return np.eye(k)[labels]


def make_initializer(
    events: list[str],
    labels: list[int],
    k: int,
    assignment: object,
    *,
    min_cluster_size: int | None = None,
    estimator: ScriptedEstimator | None = None,
    partition: object | None = None,
    partition_error: Exception | None = None,
    match_error: Exception | None = None,
) -> tuple[ClusterInitializer, ScriptedPartitioner, ScriptedEstimator, ScriptedMatcher]:
    """An initializer with scripted roles and the roles themselves."""

    partitioner = ScriptedPartitioner(
        events, partition_matrix(labels, k) if partition is None else partition, partition_error
    )
    if estimator is None:
        estimator = ScriptedEstimator(events)
    matcher = ScriptedMatcher(events, assignment, match_error)
    initializer = ClusterInitializer(
        partitioner,
        component_estimator=estimator,
        matcher=matcher,
        min_cluster_size=min_cluster_size,
    )
    return initializer, partitioner, estimator, matcher


def fullmatch(text: str) -> str:
    """Regex that matches the whole text literally."""

    return f"^{re.escape(text)}$"


class TestConstruction:
    """Storage and validation of the roles and the cluster size threshold."""

    def test_stores_given_roles_and_threshold(self, events: list[str]) -> None:
        partitioner = ScriptedPartitioner(events, None)
        estimator = ScriptedEstimator(events)
        matcher = ScriptedMatcher(events, [1, 0])

        initializer = ClusterInitializer(
            partitioner, component_estimator=estimator, matcher=matcher, min_cluster_size=3
        )

        assert initializer.partitioner is partitioner
        assert initializer.component_estimator is estimator
        assert initializer.matcher is matcher
        assert initializer.min_cluster_size == 3  # noqa: PLR2004

    def test_creates_independent_default_roles(self, events: list[str]) -> None:
        first = ClusterInitializer(ScriptedPartitioner(events, None))
        second = ClusterInitializer(ScriptedPartitioner(events, None))

        assert type(first.component_estimator) is MLEComponentEstimator
        assert type(first.matcher) is HungarianMatcher
        assert first.min_cluster_size is None
        assert first.component_estimator is not second.component_estimator
        assert first.matcher is not second.matcher

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(3, id="python-int"),
            pytest.param(np.int64(3), id="numpy-int"),
        ],
    )
    def test_stores_positive_integer_threshold_as_int(self, events: list[str], value: object) -> None:
        initializer = ClusterInitializer(ScriptedPartitioner(events, None), min_cluster_size=value)

        assert initializer.min_cluster_size == 3  # noqa: PLR2004
        assert type(initializer.min_cluster_size) is int

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(0, id="zero"),
            pytest.param(-1, id="negative"),
            pytest.param(2.5, id="float"),
            pytest.param("3", id="string"),
            pytest.param(True, id="boolean"),
        ],
    )
    def test_rejects_invalid_threshold(self, events: list[str], value: object) -> None:
        expected = f"min_cluster_size must be a positive integer or None, got {value!r}."

        with pytest.raises(ValueError, match=fullmatch(expected)):
            ClusterInitializer(ScriptedPartitioner(events, None), min_cluster_size=value)

    @pytest.mark.parametrize(
        ("make_bad_role", "role", "type_name"),
        [
            pytest.param(lambda: KMeansPartitioner, "partitioner", "ABCMeta", id="partitioner-class"),
            pytest.param(quack_partitioner, "partitioner", "QuackPartitioner", id="partitioner-duck"),
            pytest.param(lambda: "mle", "component_estimator", "str", id="component-estimator"),
            pytest.param(lambda: "hungarian", "matcher", "str", id="matcher"),
        ],
    )
    def test_rejects_role_of_wrong_type(
        self, events: list[str], make_bad_role: Callable[[], object], role: str, type_name: str
    ) -> None:
        roles: dict[str, object] = {"partitioner": ScriptedPartitioner(events, None)}
        roles[role] = make_bad_role()
        role_class = "Partitioner" if role == "partitioner" else role.title().replace("_", "")
        expected = f"{role} must be a {role_class}, got {type_name}."

        with pytest.raises(TypeError, match=fullmatch(expected)):
            ClusterInitializer(**roles)


class TestInputCheck:
    """Checks of the sample, scale and estimator support before partitioning."""

    @pytest.mark.parametrize(
        ("make_families", "n", "m"),
        [
            pytest.param(
                lambda: [family(FamilyName.NORMAL), family(FamilyName.EXPONENTIAL)], 7, 4, id="normal-exponential"
            ),
            pytest.param(
                lambda: [family(FamilyName.EXPONENTIAL), family(FamilyName.NORMAL)], 7, 4, id="exponential-normal"
            ),
            pytest.param(
                lambda: [family(FamilyName.EXPONENTIAL), family(FamilyName.EXPONENTIAL)], 3, 2, id="two-exponentials"
            ),
        ],
    )
    def test_rejects_sample_that_cannot_cover_clusters_of_default_size(
        self, events: list[str], make_families: Callable[[], list[ParametricFamily]], n: int, m: int
    ) -> None:
        families = make_families()
        k = len(families)
        initializer, partitioner, _, _ = make_initializer(events, [0, 0, 1, 1], k, [1, 0])
        expected = f"X has {n} points, fewer than the {k * m} needed for {k} clusters of at least {m} points."

        with pytest.raises(ValueError, match=fullmatch(expected)):
            initializer.initialize(sample(n), families, random_state=42)

        assert partitioner.calls == []

    @pytest.mark.parametrize(
        ("make_families", "n", "labels"),
        [
            pytest.param(
                lambda: [family(FamilyName.NORMAL), family(FamilyName.EXPONENTIAL)],
                8,
                [0, 0, 0, 0, 1, 1, 1, 1],
                id="normal-exponential",
            ),
            pytest.param(
                lambda: [family(FamilyName.EXPONENTIAL), family(FamilyName.EXPONENTIAL)],
                4,
                [0, 0, 1, 1],
                id="two-exponentials",
            ),
            pytest.param(
                lambda: [family(FamilyName.NORMAL).view(mu=0), family(FamilyName.EXPONENTIAL)],
                4,
                [0, 0, 1, 1],
                id="fixed-normal-exponential",
            ),
        ],
    )
    def test_accepts_sample_at_default_threshold(
        self,
        events: list[str],
        install_table: Callable[..., list[Any]],
        make_families: Callable[[], list[ParametricFamily]],
        n: int,
        labels: list[int],
    ) -> None:
        families = make_families()
        initializer, partitioner, _, _ = make_initializer(events, labels, len(families), [1, 0])
        install_table()

        initializer.initialize(sample(n), families, random_state=42)

        assert len(partitioner.calls) == 1

    def test_default_threshold_is_computed_on_every_call(
        self,
        events: list[str],
        install_table: Callable[..., list[Any]],
        normal: ParametricFamily,
        expon: ParametricFamily,
    ) -> None:
        initializer, partitioner, _, _ = make_initializer(events, [0, 0, 1, 1], 2, [1, 0])
        install_table()
        expected = "X has 4 points, fewer than the 8 needed for 2 clusters of at least 4 points."

        initializer.initialize(sample(4), [expon, expon], random_state=42)
        with pytest.raises(ValueError, match=fullmatch(expected)):
            initializer.initialize(sample(4), [normal, expon], random_state=42)

        assert len(partitioner.calls) == 1
        assert initializer.min_cluster_size is None

    def test_rejects_sample_below_explicit_threshold(self, events: list[str], expon: ParametricFamily) -> None:
        families = [expon, expon]
        initializer, partitioner, _, _ = make_initializer(events, [0, 0, 1, 1, 1], 2, [1, 0], min_cluster_size=3)
        expected = "X has 5 points, fewer than the 6 needed for 2 clusters of at least 3 points."

        with pytest.raises(ValueError, match=fullmatch(expected)):
            initializer.initialize(sample(5), families, random_state=42)

        assert partitioner.calls == []

    def test_accepts_sample_at_explicit_threshold(
        self, events: list[str], install_table: Callable[..., list[Any]], expon: ParametricFamily
    ) -> None:
        initializer, partitioner, _, _ = make_initializer(events, [0, 0, 0, 1, 1, 1], 2, [1, 0], min_cluster_size=3)
        install_table()

        initializer.initialize(sample(6), [expon, expon], random_state=42)

        assert len(partitioner.calls) == 1

    def test_threshold_one_allows_two_points_for_two_families(
        self,
        events: list[str],
        install_table: Callable[..., list[Any]],
        normal: ParametricFamily,
        expon: ParametricFamily,
    ) -> None:
        initializer, partitioner, _, _ = make_initializer(events, [0, 1], 2, [1, 0], min_cluster_size=1)
        install_table()

        initializer.initialize(np.array([1.0, 2.0]), [normal, expon], random_state=42)

        assert len(partitioner.calls) == 1

    def test_rejects_sample_without_enough_distinct_values(self, events: list[str], expon: ParametricFamily) -> None:
        initializer, partitioner, _, _ = make_initializer(events, [0] * 8, 2, [1, 0], min_cluster_size=1)

        with pytest.raises(ValueError, match=fullmatch("X has 1 distinct values, fewer than the 2 families.")):
            initializer.initialize(np.full(8, 3.0), [expon, expon], random_state=42)

        assert partitioner.calls == []

    def test_accepts_sample_with_as_many_distinct_values_as_families(
        self, events: list[str], install_table: Callable[..., list[Any]], expon: ParametricFamily
    ) -> None:
        initializer, partitioner, _, _ = make_initializer(events, [0] * 4 + [1] * 4, 2, [1, 0], min_cluster_size=1)
        install_table()

        initializer.initialize(np.array([3.0] * 4 + [5.0] * 4), [expon, expon], random_state=42)

        assert len(partitioner.calls) == 1

    @pytest.mark.filterwarnings("error")
    def test_rejects_non_finite_scale(self, events: list[str], uniform: ParametricFamily) -> None:
        initializer, partitioner, _, _ = make_initializer(events, [0] * 8, 1, [0])
        X = np.arange(1.0, 9.0) * 1e155

        with pytest.raises(
            ValueError,
            match=fullmatch("The standard deviation of X is not finite: the values are too large. Rescale the data."),
        ):
            initializer.initialize(X, [uniform], random_state=42)

        assert partitioner.calls == []

    @pytest.mark.filterwarnings("error")
    @pytest.mark.parametrize(
        ("make_sample", "distinct"),
        [
            pytest.param(lambda: np.arange(1.0, 9.0) * 1e-170, 8, id="eight-distinct"),
            pytest.param(lambda: np.array([1e-170, 1e-170, 2e-170, 2e-170]), 2, id="two-distinct"),
        ],
    )
    def test_rejects_zero_scale_of_distinct_values(
        self,
        events: list[str],
        uniform: ParametricFamily,
        make_sample: Callable[[], UnivariateFloatArray],
        distinct: int,
    ) -> None:
        X = make_sample()
        initializer, partitioner, _, _ = make_initializer(events, [0] * len(X), 1, [0])
        expected = (
            f"The standard deviation of X is zero although X has {distinct} distinct values: "
            "the values are too close. Rescale the data."
        )

        with pytest.raises(ValueError, match=fullmatch(expected)):
            initializer.initialize(X, [uniform], random_state=42)

        assert partitioner.calls == []

    @pytest.mark.filterwarnings("error")
    def test_accepts_constant_sample_for_one_family(
        self, events: list[str], install_table: Callable[..., list[Any]], expon: ParametricFamily
    ) -> None:
        initializer, partitioner, _, _ = make_initializer(events, [0] * 8, 1, [0])
        install_table()

        initializer.initialize(np.full(8, 5.0), [expon], random_state=42)

        assert len(partitioner.calls) == 1

    def test_reports_all_unsupported_families_after_asking_every_family(
        self, events: list[str], normal: ParametricFamily, expon: ParametricFamily
    ) -> None:
        families = [normal.view(parametrization_name="meanVar"), normal, expon]
        estimator = ScriptedEstimator(events, {("Normal", "meanStd")})
        initializer, partitioner, _, _ = make_initializer(events, [0] * 12, 3, [0, 1, 2], estimator=estimator)
        expected = (
            f"{type(estimator).__name__} does not support families[0] (Normal in parametrization meanVar), "
            "families[2] (Exponential in parametrization rate)."
        )

        with pytest.raises(NotImplementedError, match=fullmatch(expected)):
            initializer.initialize(sample(12), families, random_state=42)

        assert events == ["supports", "supports", "supports"]
        assert len(estimator.supports_calls) == 3  # noqa: PLR2004
        assert all(estimator.supports_calls[j] is families[j] for j in range(3))
        assert partitioner.calls == []

    @pytest.mark.parametrize(
        ("make_sample", "make_families", "min_size", "supported_keys", "expected"),
        [
            pytest.param(
                lambda: sample(7),
                lambda: [family(FamilyName.NORMAL), family(FamilyName.EXPONENTIAL)],
                None,
                None,
                "X has 7 points, fewer than the 8 needed for 2 clusters of at least 4 points.",
                id="few-points",
            ),
            pytest.param(
                lambda: np.full(8, 3.0),
                lambda: [family(FamilyName.EXPONENTIAL), family(FamilyName.EXPONENTIAL)],
                1,
                None,
                "X has 1 distinct values, fewer than the 2 families.",
                id="few-distinct-values",
            ),
            pytest.param(
                lambda: np.arange(1.0, 9.0) * 1e155,
                lambda: [family(FamilyName.CONTINUOUS_UNIFORM)],
                None,
                None,
                "The standard deviation of X is not finite: the values are too large. Rescale the data.",
                id="non-finite-scale",
            ),
            pytest.param(
                lambda: np.arange(1.0, 9.0) * 1e-170,
                lambda: [family(FamilyName.CONTINUOUS_UNIFORM)],
                None,
                None,
                (
                    "The standard deviation of X is zero although X has 8 distinct values: "
                    "the values are too close. Rescale the data."
                ),
                id="zero-scale",
            ),
            pytest.param(
                lambda: sample(12),
                lambda: [
                    family(FamilyName.NORMAL).view(parametrization_name="meanVar"),
                    family(FamilyName.NORMAL),
                    family(FamilyName.EXPONENTIAL),
                ],
                None,
                {("Normal", "meanStd")},
                (
                    "ScriptedEstimator does not support families[0] (Normal in parametrization meanVar), "
                    "families[2] (Exponential in parametrization rate)."
                ),
                id="unsupported-family",
            ),
        ],
    )
    def test_failure_prevents_partitioning_and_table_creation(
        self,
        events: list[str],
        install_table: Callable[..., list[Any]],
        make_sample: Callable[[], UnivariateFloatArray],
        make_families: Callable[[], list[ParametricFamily]],
        min_size: int | None,
        supported_keys: set[tuple[str, str]] | None,
        expected: str,
    ) -> None:
        X = make_sample()
        families = make_families()
        k = len(families)
        estimator = ScriptedEstimator(events, supported_keys)
        initializer, partitioner, _, _ = make_initializer(
            events,
            [0] * len(X),
            k,
            list(range(k)),
            min_cluster_size=min_size,
            estimator=estimator,
        )
        tables = install_table()

        with pytest.raises((ValueError, NotImplementedError), match=fullmatch(expected)):
            initializer.initialize(X, families, random_state=42)

        assert partitioner.calls == []
        assert tables == []


class TestPartition:
    """Partitioning, structural validation and cluster size checks."""

    def test_partitioner_receives_sample_k_and_the_same_generator(
        self,
        events: list[str],
        install_table: Callable[..., list[Any]],
        normal: ParametricFamily,
        expon: ParametricFamily,
    ) -> None:
        initializer, partitioner, _, _ = make_initializer(events, [0] * 4 + [1] * 4, 2, [1, 0])
        tables = install_table()
        X = sample(8)
        generator = np.random.default_rng(42)

        initializer.initialize(X, [normal, expon], random_state=generator)

        assert len(partitioner.calls) == 1
        received_X, k, rng = partitioner.calls[0]
        assert received_X is tables[0].X
        np.testing.assert_array_equal(received_X, X)
        assert k == 2  # noqa: PLR2004
        assert rng is generator

    def test_bool_partition_is_converted_to_float64_for_table(
        self,
        events: list[str],
        install_table: Callable[..., list[Any]],
        normal: ParametricFamily,
        expon: ParametricFamily,
    ) -> None:
        labels = [0] * 4 + [1] * 4
        bool_partition = partition_matrix(labels, 2).astype(bool)
        initializer, partitioner, estimator, _ = make_initializer(events, labels, 2, [1, 0], partition=bool_partition)
        tables = install_table()
        families = [normal, expon]

        initializer.initialize(sample(8), families, random_state=42)

        received_X, _, _ = partitioner.calls[0]
        assert tables[0].X is received_X
        assert tables[0].H.dtype == np.float64
        np.testing.assert_array_equal(tables[0].H, bool_partition)
        assert tables[0].families is families
        assert tables[0].component_estimator is estimator

    def test_wrong_partition_shape_is_value_error_before_table(
        self,
        events: list[str],
        install_table: Callable[..., list[Any]],
        normal: ParametricFamily,
        expon: ParametricFamily,
    ) -> None:
        initializer, _, _, matcher = make_initializer(events, [0] * 4 + [1] * 4, 2, [1, 0], partition=np.zeros((8, 3)))
        tables = install_table()

        with pytest.raises(ValueError, match="^A partition of 8 points into 2 clusters must have shape"):
            initializer.initialize(sample(8), [normal, expon], random_state=42)

        assert tables == []
        assert matcher.tables == []

    def test_empty_cluster_is_initialization_error_before_table(
        self,
        events: list[str],
        install_table: Callable[..., list[Any]],
        normal: ParametricFamily,
        expon: ParametricFamily,
    ) -> None:
        initializer, _, _, matcher = make_initializer(events, [0] * 8, 2, [1, 0])
        tables = install_table()

        with pytest.raises(InitializationError, match="^The partition has empty clusters"):
            initializer.initialize(sample(8), [normal, expon], random_state=42)

        assert tables == []
        assert matcher.tables == []

    @pytest.mark.parametrize(
        ("make_families", "labels", "k", "min_size", "m", "items"),
        [
            pytest.param(
                lambda: [family(FamilyName.EXPONENTIAL), family(FamilyName.EXPONENTIAL)],
                [0, 0, 1, 1, 1, 1, 1],
                2,
                3,
                3,
                "cluster 0 has 2",
                id="explicit-threshold",
            ),
            pytest.param(
                lambda: [family(FamilyName.EXPONENTIAL), family(FamilyName.EXPONENTIAL)],
                [0, 0, 0, 1, 1, 1, 1],
                2,
                3,
                3,
                None,
                id="exact-explicit-threshold",
            ),
            pytest.param(
                lambda: [family(FamilyName.NORMAL), family(FamilyName.EXPONENTIAL)],
                [0, 0, 0, 1, 1, 1, 1, 1],
                2,
                None,
                4,
                "cluster 0 has 3",
                id="default-threshold",
            ),
            pytest.param(
                lambda: [family(FamilyName.EXPONENTIAL)] * 3,
                [0, 1, 1] + [2] * 9,
                3,
                3,
                3,
                "cluster 0 has 1, cluster 1 has 2",
                id="two-small-clusters",
            ),
            pytest.param(
                lambda: [family(FamilyName.EXPONENTIAL), family(FamilyName.EXPONENTIAL)],
                [0] * 5 + [1] * 2,
                2,
                3,
                3,
                "cluster 1 has 2",
                id="small-cluster-not-first",
            ),
            pytest.param(
                lambda: [family(FamilyName.EXPONENTIAL), family(FamilyName.EXPONENTIAL)],
                [0] + [1] * 7,
                2,
                1,
                1,
                None,
                id="threshold-one",
            ),
        ],
    )
    def test_enforces_cluster_size_threshold_after_partitioning(
        self,
        events: list[str],
        install_table: Callable[..., list[Any]],
        make_families: Callable[[], list[ParametricFamily]],
        labels: list[int],
        k: int,
        min_size: int | None,
        m: int,
        items: str | None,
    ) -> None:
        families = make_families()
        initializer, _, _, matcher = make_initializer(events, labels, k, list(range(k)), min_cluster_size=min_size)
        tables = install_table()

        if items is None:
            initializer.initialize(sample(len(labels)), families, random_state=42)

            assert len(tables) == 1
            return

        expected = f"The partition has clusters with fewer than {m} points: {items}."
        with pytest.raises(InitializationError, match=fullmatch(expected)) as exc_info:
            initializer.initialize(sample(len(labels)), families, random_state=42)

        assert type(exc_info.value) is InitializationError
        assert tables == []
        assert matcher.tables == []

    @pytest.mark.parametrize(
        "make_error",
        [
            pytest.param(lambda: InitializationError("boom"), id="initialization-error"),
            pytest.param(lambda: ValueError("boom"), id="value-error"),
        ],
    )
    def test_error_from_partitioner_passes_through_as_same_object(
        self,
        events: list[str],
        install_table: Callable[..., list[Any]],
        normal: ParametricFamily,
        expon: ParametricFamily,
        make_error: Callable[[], Exception],
    ) -> None:
        error = make_error()
        initializer, _, _, _ = make_initializer(events, [0] * 4 + [1] * 4, 2, [1, 0], partition_error=error)
        install_table()

        with pytest.raises(type(error), match="^boom$") as exc_info:
            initializer.initialize(sample(8), [normal, expon], random_state=42)

        assert exc_info.value is error


class TestAssembly:
    """Matching, assignment validation and assembly of the mixture."""

    def test_matcher_receives_created_table_and_result_is_table_mixture(
        self,
        events: list[str],
        install_table: Callable[..., list[Any]],
        normal: ParametricFamily,
        expon: ParametricFamily,
    ) -> None:
        assignment = np.array([1, 0])
        initializer, _, _, matcher = make_initializer(events, [0] * 4 + [1] * 4, 2, assignment)
        scripted_mixture = valid_mixture([normal, expon])
        tables = install_table(mixture=scripted_mixture)

        result = initializer.initialize(sample(8), [normal, expon], random_state=42)

        assert len(matcher.tables) == 1
        assert matcher.tables[0] is tables[0]
        assert len(tables[0].assignments) == 1
        assert tables[0].assignments[0] is assignment
        assert result is scripted_mixture
        assert result is tables[0].returned_mixture

    @pytest.mark.parametrize(
        "assignment",
        [
            pytest.param([0, 0], id="repeated-cluster"),
            pytest.param([0], id="too-short"),
            pytest.param(np.array([0.0, 1.0]), id="non-integer"),
        ],
    )
    def test_rejects_assignment_that_is_not_permutation_before_mixture(
        self,
        events: list[str],
        install_table: Callable[..., list[Any]],
        normal: ParametricFamily,
        expon: ParametricFamily,
        assignment: object,
    ) -> None:
        initializer, _, _, _ = make_initializer(events, [0] * 4 + [1] * 4, 2, assignment)
        tables = install_table()

        with pytest.raises(ValueError, match="^The matcher must return a permutation"):
            initializer.initialize(sample(8), [normal, expon], random_state=42)

        assert len(tables) == 1
        assert tables[0].assignments == []

    @pytest.mark.parametrize(
        "assignment",
        [
            pytest.param([1, 0], id="list"),
            pytest.param((1, 0), id="tuple"),
        ],
    )
    def test_list_and_tuple_assignments_reach_mixture_as_integer_arrays(
        self,
        events: list[str],
        install_table: Callable[..., list[Any]],
        normal: ParametricFamily,
        expon: ParametricFamily,
        assignment: object,
    ) -> None:
        initializer, _, _, _ = make_initializer(events, [0] * 4 + [1] * 4, 2, assignment)
        tables = install_table()

        initializer.initialize(sample(8), [normal, expon], random_state=42)

        received = tables[0].assignments[0]
        assert isinstance(received, np.ndarray)
        assert received.dtype.kind == "i"
        np.testing.assert_array_equal(received, [1, 0])

    def test_error_from_matcher_passes_through_as_same_object(
        self,
        events: list[str],
        install_table: Callable[..., list[Any]],
        normal: ParametricFamily,
        expon: ParametricFamily,
    ) -> None:
        error = InitializationError("no assignment")
        initializer, _, _, _ = make_initializer(events, [0] * 4 + [1] * 4, 2, [1, 0], match_error=error)
        install_table()

        with pytest.raises(InitializationError, match="^no assignment$") as exc_info:
            initializer.initialize(sample(8), [normal, expon], random_state=42)

        assert exc_info.value is error

    def test_error_from_table_mixture_passes_through_as_same_object(
        self,
        events: list[str],
        install_table: Callable[..., list[Any]],
        normal: ParametricFamily,
        expon: ParametricFamily,
    ) -> None:
        error = ComponentEstimationError("no component")
        initializer, _, _, _ = make_initializer(events, [0] * 4 + [1] * 4, 2, [1, 0])
        install_table(error)

        with pytest.raises(ComponentEstimationError, match="^no component$") as exc_info:
            initializer.initialize(sample(8), [normal, expon], random_state=42)

        assert exc_info.value is error

    def test_successful_call_order(
        self,
        events: list[str],
        install_table: Callable[..., list[Any]],
        normal: ParametricFamily,
        expon: ParametricFamily,
    ) -> None:
        initializer, _, _, _ = make_initializer(events, [0] * 4 + [1] * 4, 2, [1, 0])
        install_table()

        initializer.initialize(sample(8), [normal, expon], random_state=42)

        assert events == ["supports", "supports", "partition", "table", "match", "mixture"]

    def test_every_call_creates_new_table_and_roles_stay_unchanged(
        self,
        events: list[str],
        install_table: Callable[..., list[Any]],
        normal: ParametricFamily,
        expon: ParametricFamily,
    ) -> None:
        initializer, partitioner, estimator, matcher = make_initializer(
            events, [0] * 4 + [1] * 4, 2, [1, 0], min_cluster_size=3
        )
        tables = install_table()

        initializer.initialize(sample(8), [normal, expon], random_state=42)
        initializer.initialize(sample(8), [normal, expon], random_state=42)

        assert len(tables) == 2  # noqa: PLR2004
        assert tables[0] is not tables[1]
        assert initializer.partitioner is partitioner
        assert initializer.component_estimator is estimator
        assert initializer.matcher is matcher
        assert initializer.min_cluster_size == 3  # noqa: PLR2004
