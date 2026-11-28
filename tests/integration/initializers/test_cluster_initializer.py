"""Integration tests of ``ClusterInitializer`` with real partitioners, estimators and matchers."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import copy
import re
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pysatl_core.families import ParametricFamily
from pysatl_core.types import FamilyName
from pysatl_mpest.core import _components
from pysatl_mpest.initializers import (
    ClusterInitializer,
    ComponentEstimationError,
    HungarianMatcher,
    IdentityMatcher,
    InitializationError,
    KMeansPartitioner,
    Matcher,
    Partitioner,
    QuantilePartitioner,
    RandomCentersPartitioner,
    SklearnPartitioner,
)
from pysatl_mpest.initializers.cluster import ComponentTable
from pysatl_mpest.typings import UnivariateFloatArray, UnivariateIntArray
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering

from tests.helpers.families import family
from tests.helpers.partitions import MAX_CLUSTERS, partition_inputs

MODE_CENTER = 5.0


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


def make_hungarian_matcher() -> Matcher:
    """The exact matcher."""

    return HungarianMatcher()


def make_identity_matcher() -> Matcher:
    """The matcher that keeps the order of the columns."""

    return IdentityMatcher()


PARTITIONER_FACTORIES: tuple[Callable[[], Partitioner], ...] = (
    make_kmeans_partitioner,
    make_quantile_partitioner,
    make_random_centers_partitioner,
    make_sklearn_kmeans_partitioner,
    make_sklearn_agglomerative_partitioner,
)
PARTITIONER_IDS = ("kmeans", "quantile", "random-centers", "sklearn-kmeans", "sklearn-agglomerative")
MATCHER_FACTORIES: tuple[Callable[[], Matcher], ...] = (make_hungarian_matcher, make_identity_matcher)
MATCHER_IDS = ("hungarian", "identity")
FACTORIES: tuple[Callable[[], ParametricFamily], ...] = (
    lambda: family(FamilyName.NORMAL),
    lambda: family(FamilyName.EXPONENTIAL),
    lambda: family(FamilyName.CONTINUOUS_UNIFORM),
    lambda: family(FamilyName.NORMAL).view(mu=0),
)


def three_modes() -> UnivariateFloatArray:
    """A sample from three well-separated normal components."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(-10, 1, 150), rng.normal(0, 1, 150), rng.normal(10, 1, 150)])


def example_sample() -> UnivariateFloatArray:
    """A sample from two normal components and one exponential component."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(-8, 1, 200), rng.normal(-3, 0.7, 200), rng.exponential(2, 300)])


def modes_zero_and_ten() -> UnivariateFloatArray:
    """A sample from two well-separated normal components."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(0, 1, 300), rng.normal(10, 1, 300)])


def outlier_sample() -> UnivariateFloatArray:
    """A small normal sample with one distant outlier."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(0, 1, 20), [100.0]])


def repeated_values() -> UnivariateFloatArray:
    """Eight equal values."""

    return np.full(8, 5.0, dtype=np.float64)


def large_values() -> UnivariateFloatArray:
    """Eight values whose standard deviation overflows ``float64``."""

    return np.arange(1.0, 9.0) * 1e155


def close_values() -> UnivariateFloatArray:
    """Eight distinct values whose standard deviation vanishes in ``float64``."""

    return np.arange(1.0, 9.0) * 1e-170


def fullmatch(text: str) -> str:
    """Regex that matches the whole text literally."""

    return f"^{re.escape(text)}$"


def normal_families(count: int) -> list[ParametricFamily]:
    """A list of ``count`` plain Normal families."""

    return [family(FamilyName.NORMAL) for _ in range(count)]


class UserMatcher(Matcher):
    """A user-written matcher that implements only the matching step and returns a fixed result."""

    def __init__(self, result: object) -> None:
        self.result = result

    def _match(self, table: ComponentTable) -> UnivariateIntArray:
        return cast(UnivariateIntArray, self.result)


class TestKnownMixtures:
    """Starting mixtures built for samples with known components."""

    @pytest.mark.parametrize("make_matcher", MATCHER_FACTORIES, ids=MATCHER_IDS)
    @pytest.mark.parametrize("make_partitioner", PARTITIONER_FACTORIES, ids=PARTITIONER_IDS)
    def test_every_combination_builds_a_valid_mixture_on_three_modes(
        self, make_partitioner: Callable[[], Partitioner], make_matcher: Callable[[], Matcher]
    ) -> None:
        X = three_modes()
        families = normal_families(3)

        mixture = ClusterInitializer(make_partitioner(), matcher=make_matcher()).initialize(
            X, families, random_state=42
        )

        assert mixture.n_components == len(families)
        for j, component in enumerate(mixture.components):
            assert _components.component_family_key(component) == _components.family_key(families[j])
        assert np.isfinite(mixture.loglikelihood(X))

    @pytest.mark.parametrize("make_partitioner", PARTITIONER_FACTORIES, ids=PARTITIONER_IDS)
    def test_hungarian_matcher_orders_identical_normals_by_increasing_mu(
        self, make_partitioner: Callable[[], Partitioner]
    ) -> None:
        families = normal_families(3)

        mixture = ClusterInitializer(make_partitioner(), matcher=HungarianMatcher()).initialize(
            three_modes(), families, random_state=42
        )
        mus = [_components.free_parameters(component)["mu"] for component in mixture.components]

        assert mus[0] < mus[1] < mus[2]

    def test_kmeans_leaves_the_leftmost_component_on_the_left_for_seeds_zero_to_nine(self) -> None:
        X = three_modes()
        families = normal_families(3)

        for seed in range(10):
            mixture = ClusterInitializer(KMeansPartitioner(), matcher=HungarianMatcher()).initialize(
                X, families, random_state=seed
            )

            assert _components.free_parameters(mixture[0])["mu"] < -MODE_CENTER

    def test_identity_matcher_keeps_the_arbitrary_order_of_kmeans_columns(self) -> None:
        X = modes_zero_and_ten()
        families = normal_families(2)
        first_mus: list[float] = []

        for seed in range(10):
            mixture = ClusterInitializer(KMeansPartitioner(), matcher=IdentityMatcher()).initialize(
                X, families, random_state=seed
            )
            first_mus.append(_components.free_parameters(mixture[0])["mu"])

        assert any(mu < MODE_CENTER for mu in first_mus)
        assert any(mu > MODE_CENTER for mu in first_mus)

    @pytest.mark.parametrize("make_partitioner", PARTITIONER_FACTORIES, ids=PARTITIONER_IDS)
    def test_two_normals_and_an_exponential_are_separated_on_the_example_sample(
        self, make_partitioner: Callable[[], Partitioner]
    ) -> None:
        X = example_sample()
        families = [family(FamilyName.NORMAL), family(FamilyName.NORMAL), family(FamilyName.EXPONENTIAL)]

        mixture = ClusterInitializer(make_partitioner(), matcher=HungarianMatcher()).initialize(
            X, families, random_state=42
        )

        mu_0 = _components.free_parameters(mixture[0])["mu"]
        mu_1 = _components.free_parameters(mixture[1])["mu"]
        assert mu_0 < mu_1 < 0
        assert np.isfinite(mixture.loglikelihood(X))

    def test_kmeans_recovers_the_parameters_of_the_example_mixture(self) -> None:
        X = example_sample()
        families = [family(FamilyName.NORMAL), family(FamilyName.NORMAL), family(FamilyName.EXPONENTIAL)]

        mixture = ClusterInitializer(KMeansPartitioner(), matcher=HungarianMatcher()).initialize(
            X, families, random_state=42
        )

        assert _components.free_parameters(mixture[0])["mu"] == pytest.approx(-8, abs=0.2)
        assert _components.free_parameters(mixture[1])["mu"] == pytest.approx(-3, abs=0.2)
        assert _components.free_parameters(mixture[2])["lambda_"] == pytest.approx(0.5, abs=0.06)
        assert mixture.weights == pytest.approx([2 / 7, 2 / 7, 3 / 7], abs=0.01)

    @pytest.mark.parametrize(
        "make_partitioner",
        [
            pytest.param(make_kmeans_partitioner, id="kmeans"),
            pytest.param(make_quantile_partitioner, id="quantile"),
            pytest.param(make_sklearn_kmeans_partitioner, id="sklearn-kmeans"),
            pytest.param(make_sklearn_agglomerative_partitioner, id="sklearn-agglomerative"),
        ],
    )
    def test_fixed_mu_component_takes_the_left_mode_with_stable_partitioners(
        self, make_partitioner: Callable[[], Partitioner]
    ) -> None:
        families = [family(FamilyName.NORMAL).view(mu=0), family(FamilyName.NORMAL)]

        mixture = ClusterInitializer(make_partitioner(), matcher=HungarianMatcher()).initialize(
            modes_zero_and_ten(), families, random_state=42
        )

        assert _components.all_parameters(mixture[0])["mu"] == 0
        assert _components.all_parameters(mixture[0])["sigma"] == pytest.approx(1, abs=0.15)
        assert _components.all_parameters(mixture[1])["mu"] == pytest.approx(10, abs=0.1)

    def test_fixed_mu_component_keeps_its_fixed_parameter_with_random_centers(self) -> None:
        families = [family(FamilyName.NORMAL).view(mu=0), family(FamilyName.NORMAL)]

        mixture = ClusterInitializer(RandomCentersPartitioner(), matcher=HungarianMatcher()).initialize(
            modes_zero_and_ten(), families, random_state=42
        )

        assert _components.all_parameters(mixture[0])["mu"] == 0


class TestRandomState:
    """Handling of integer seeds and shared generators."""

    @pytest.mark.parametrize(
        "make_partitioner",
        [
            pytest.param(make_kmeans_partitioner, id="kmeans"),
            pytest.param(make_random_centers_partitioner, id="random-centers"),
            pytest.param(make_sklearn_kmeans_partitioner, id="sklearn-kmeans"),
        ],
    )
    def test_integer_seed_reproduces_the_mixture_after_other_seeded_calls(
        self, make_partitioner: Callable[[], Partitioner]
    ) -> None:
        X = three_modes()
        families = normal_families(3)
        initializer = ClusterInitializer(make_partitioner())

        first = initializer.initialize(X, families, random_state=7)
        initializer.initialize(X, families, random_state=8)
        third = ClusterInitializer(make_partitioner()).initialize(X, families, random_state=7)

        assert first == third

    def test_shared_generator_advances_and_changes_the_next_mixture(self) -> None:
        X = three_modes()
        families = normal_families(3)
        rng = np.random.default_rng(42)
        initial_state = copy.deepcopy(rng.bit_generator.state)
        initializer = ClusterInitializer(RandomCentersPartitioner())

        first = initializer.initialize(X, families, random_state=rng)
        state_after_first = copy.deepcopy(rng.bit_generator.state)
        second = initializer.initialize(X, families, random_state=rng)

        assert first != second
        assert state_after_first != initial_state


class TestClusterSize:
    """Common lower bound on the number of points in every cluster."""

    def test_default_threshold_accepts_eight_points_for_normal_and_exponential(self) -> None:
        X = np.asarray([1, 2, 3, 4, 10, 11, 12, 13], dtype=np.float64)
        families = [family(FamilyName.NORMAL), family(FamilyName.EXPONENTIAL)]

        mixture = ClusterInitializer(QuantilePartitioner()).initialize(X, families, random_state=42)

        assert mixture.n_components == len(families)

    def test_default_threshold_rejects_seven_points_for_normal_and_exponential(self) -> None:
        X = np.asarray([1, 2, 3, 4, 10, 11, 12], dtype=np.float64)
        families = [family(FamilyName.NORMAL), family(FamilyName.EXPONENTIAL)]
        expected = "X has 7 points, fewer than the 8 needed for 2 clusters of at least 4 points."

        with pytest.raises(ValueError, match=fullmatch(expected)):
            ClusterInitializer(QuantilePartitioner()).initialize(X, families, random_state=42)

    def test_default_threshold_accepts_four_points_for_two_exponentials(self) -> None:
        X = np.asarray([1, 2, 10, 11], dtype=np.float64)
        families = [family(FamilyName.EXPONENTIAL), family(FamilyName.EXPONENTIAL)]

        mixture = ClusterInitializer(QuantilePartitioner()).initialize(X, families, random_state=42)

        assert mixture.n_components == len(families)

    def test_default_threshold_counts_only_free_parameters(self) -> None:
        X = np.asarray([-1, 1, 10, 11], dtype=np.float64)
        families = [family(FamilyName.NORMAL).view(mu=0), family(FamilyName.EXPONENTIAL)]

        mixture = ClusterInitializer(QuantilePartitioner()).initialize(X, families, random_state=42)

        assert mixture.n_components == len(families)

    def test_outlier_cluster_is_rejected_by_the_default_threshold(self) -> None:
        families = normal_families(2)

        with pytest.raises(
            InitializationError, match=r"^The partition has clusters with fewer than 4 points: cluster [01] has 1\.$"
        ) as exc_info:
            ClusterInitializer(KMeansPartitioner()).initialize(outlier_sample(), families, random_state=42)

        error = exc_info.value
        assert type(error) is InitializationError
        assert re.fullmatch(r"The partition has clusters with fewer than 4 points: cluster [01] has 1\.", str(error))

    def test_threshold_one_allows_the_outlier_to_become_an_exponential_component(self) -> None:
        families = [family(FamilyName.EXPONENTIAL), family(FamilyName.NORMAL)]

        mixture = ClusterInitializer(KMeansPartitioner(), min_cluster_size=1).initialize(
            outlier_sample(), families, random_state=42
        )

        assert _components.free_parameters(mixture[0])["lambda_"] == pytest.approx(0.01)
        assert mixture.weights == pytest.approx([1 / 21, 20 / 21], rel=1e-12)

    def test_explicit_threshold_changes_the_required_number_of_points(self) -> None:
        X = np.asarray([1, 2, 3, 4, 5], dtype=np.float64)
        families = [family(FamilyName.NORMAL), family(FamilyName.EXPONENTIAL)]
        expected = "X has 5 points, fewer than the 6 needed for 2 clusters of at least 3 points."

        with pytest.raises(ValueError, match=fullmatch(expected)):
            ClusterInitializer(QuantilePartitioner(), min_cluster_size=3).initialize(X, families, random_state=42)


class TestErrors:
    """Failures of matching, component assembly, the result check and input validation."""

    def test_hungarian_matcher_reports_all_impossible_pairs_when_no_assignment_exists(self) -> None:
        X = np.asarray([-9, -8, -7, -6, -3, 1, 2, 3], dtype=np.float64)
        families = [family(FamilyName.EXPONENTIAL), family(FamilyName.NORMAL)]

        with pytest.raises(
            InitializationError, match="^No assignment gives every family a suitable cluster. Impossible pairs:"
        ) as exc_info:
            ClusterInitializer(QuantilePartitioner(), matcher=HungarianMatcher()).initialize(
                X, families, random_state=42
            )

        error = exc_info.value
        text = str(error)
        assert type(error) is InitializationError
        assert text.startswith("No assignment gives every family a suitable cluster. Impossible pairs:\n")
        assert any(line.startswith("family 0 (Exponential), cluster 0: ") for line in text.splitlines())
        assert any(line.startswith("family 0 (Exponential), cluster 1: 1 of 4 points") for line in text.splitlines())

    @pytest.mark.parametrize(
        ("make_matcher", "make_families"),
        [
            pytest.param(
                make_identity_matcher,
                [lambda: family(FamilyName.EXPONENTIAL), lambda: family(FamilyName.NORMAL)],
                id="identity-matcher",
            ),
            pytest.param(
                make_hungarian_matcher,
                [lambda: family(FamilyName.EXPONENTIAL), lambda: family(FamilyName.EXPONENTIAL)],
                id="single-group",
            ),
        ],
    )
    def test_unestimated_pair_fails_during_mixture_assembly(
        self,
        make_matcher: Callable[[], Matcher],
        make_families: list[Callable[[], ParametricFamily]],
    ) -> None:
        X = np.asarray([-9, -8, -7, -6, -3, 1, 2, 3], dtype=np.float64)
        families = [make_family() for make_family in make_families]

        with pytest.raises(ComponentEstimationError, match=r"^family 0 \(Exponential\), cluster 0: "):
            ClusterInitializer(QuantilePartitioner(), matcher=make_matcher()).initialize(X, families, random_state=42)

    def test_uncovered_sample_point_fails_the_result_check(self) -> None:
        X = np.asarray([-1, 5, 6, 7, 20, 21, 22, 23], dtype=np.float64)
        families = [family(FamilyName.EXPONENTIAL), family(FamilyName.EXPONENTIAL)]

        with pytest.raises(
            InitializationError, match=r"^The log-likelihood of the starting mixture on the sample is -inf\."
        ) as exc_info:
            ClusterInitializer(QuantilePartitioner()).initialize(X, families, random_state=42)

        error = exc_info.value
        assert type(error) is InitializationError
        assert "for example -1.0." in str(error)

    def test_uncovered_cluster_point_can_be_covered_by_another_component(self) -> None:
        X = np.asarray([-1, 5, 6, 7, 20, 21, 22, 23], dtype=np.float64)
        families = [family(FamilyName.EXPONENTIAL), family(FamilyName.NORMAL)]

        mixture = ClusterInitializer(QuantilePartitioner(), matcher=IdentityMatcher()).initialize(
            X, families, random_state=42
        )

        assert mixture.loglikelihood(X) == pytest.approx(-223.74, abs=0.01)

    def test_unsupported_parametrizations_are_listed_before_partitioning(self) -> None:
        families = [
            family(FamilyName.NORMAL).view(parametrization_name="meanVar"),
            family(FamilyName.EXPONENTIAL),
            family(FamilyName.EXPONENTIAL).view(parametrization_name="scale"),
        ]
        expected = (
            "MLEComponentEstimator does not support families[0] (Normal in parametrization meanVar), "
            "families[2] (Exponential in parametrization scale)."
        )

        with pytest.raises(NotImplementedError, match=fullmatch(expected)):
            ClusterInitializer(QuantilePartitioner()).initialize(three_modes(), families, random_state=42)

    def test_user_matcher_result_that_is_not_permutation_is_value_error(self) -> None:
        X = np.arange(1.0, 13.0)
        families = [family(FamilyName.NORMAL), family(FamilyName.EXPONENTIAL)]

        with pytest.raises(ValueError, match=r"^The matcher must return a permutation of length 2, got \[0, 5\]"):
            ClusterInitializer(QuantilePartitioner(), matcher=UserMatcher([0, 5])).initialize(
                X, families, random_state=42
            )

    def test_user_matcher_may_return_a_tuple(self) -> None:
        X = np.arange(1.0, 13.0)
        families = [family(FamilyName.NORMAL), family(FamilyName.EXPONENTIAL)]

        mixture = ClusterInitializer(QuantilePartitioner(), matcher=UserMatcher((1, 0))).initialize(
            X, families, random_state=42
        )

        assert _components.free_parameters(mixture[0])["mu"] == pytest.approx(9.5)

    @pytest.mark.parametrize(
        ("make_sample", "expected"),
        [
            pytest.param(
                repeated_values,
                "X has 1 distinct values, fewer than the 2 families.",
                id="one-distinct-value",
            ),
            pytest.param(
                large_values,
                "The standard deviation of X is not finite: the values are too large. Rescale the data.",
                id="too-large",
            ),
            pytest.param(
                close_values,
                "The standard deviation of X is zero although X has 8 distinct values: "
                "the values are too close. Rescale the data.",
                id="too-close",
            ),
        ],
    )
    @pytest.mark.filterwarnings("error")
    def test_invalid_input_is_rejected_before_partitioning_without_numpy_warnings(
        self, make_sample: Callable[[], UnivariateFloatArray], expected: str
    ) -> None:
        families = normal_families(2)

        with pytest.raises(ValueError, match=fullmatch(expected)):
            ClusterInitializer(QuantilePartitioner()).initialize(make_sample(), families, random_state=42)


@settings(deadline=None)
@pytest.mark.filterwarnings("error")
@given(
    inputs=partition_inputs(distinct_values=True),
    family_indices=st.lists(st.integers(0, len(FACTORIES) - 1), min_size=MAX_CLUSTERS, max_size=MAX_CLUSTERS),
    partitioner_index=st.integers(0, 2),
    matcher_index=st.integers(0, 1),
)
def test_real_roles_return_a_mixture_or_initialization_error(
    inputs: tuple[UnivariateFloatArray, int, int],
    family_indices: list[int],
    partitioner_index: int,
    matcher_index: int,
) -> None:
    X, k, seed = inputs
    assume(np.std(X) > 0 or len(np.unique(X)) == 1)
    families = [FACTORIES[index]() for index in family_indices[:k]]
    partitioner = PARTITIONER_FACTORIES[:3][partitioner_index]()
    matcher = MATCHER_FACTORIES[matcher_index]()

    try:
        mixture = ClusterInitializer(partitioner, matcher=matcher, min_cluster_size=1).initialize(X, families, seed)
    except InitializationError:
        return

    assert mixture.n_components == k


def normal_exponential_uniform_components() -> UnivariateFloatArray:
    """A sample from well-separated normal, exponential and continuous uniform components."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(-10, 1, 100), rng.exponential(1, 100), rng.uniform(10, 20, 100)])


def unequal_modes() -> UnivariateFloatArray:
    """A sample from three well-separated normal components with different weights."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(-10, 1, 100), rng.normal(0, 2, 200), rng.normal(10, 1, 300)])


def make_normal_exponential_uniform_families() -> list[ParametricFamily]:
    """Plain Normal, Exponential and ContinuousUniform families."""

    return [family(FamilyName.NORMAL), family(FamilyName.EXPONENTIAL), family(FamilyName.CONTINUOUS_UNIFORM)]


def make_sklearn_spectral_partitioner() -> Partitioner:
    """The scikit-learn spectral partitioning method."""

    return SklearnPartitioner(SpectralClustering())


class TestClusterStatistics:
    """Components of the start are the statistics of the clusters of the same partition."""

    @pytest.mark.parametrize(
        "make_partitioner",
        [
            pytest.param(make_kmeans_partitioner, id="kmeans"),
            pytest.param(make_sklearn_kmeans_partitioner, id="sklearn-kmeans"),
            pytest.param(make_sklearn_agglomerative_partitioner, id="sklearn-agglomerative"),
            pytest.param(make_sklearn_spectral_partitioner, id="sklearn-spectral"),
            pytest.param(make_quantile_partitioner, id="quantile"),
            pytest.param(make_random_centers_partitioner, id="random-centers"),
        ],
    )
    @pytest.mark.filterwarnings("error")
    def test_component_statistics_match_mean_ordered_clusters(
        self, make_partitioner: Callable[[], Partitioner]
    ) -> None:
        """Component means, scales and weights equal the statistics of the mean-ordered manual clusters."""

        X = unequal_modes()
        families = normal_families(3)
        partitioner = make_partitioner()
        H = partitioner.partition(X, 3, np.random.default_rng(42))
        means = np.array([X[H[:, cluster] > 0].mean() for cluster in range(3)])
        mean_ordered_clusters = np.argsort(means)

        start = ClusterInitializer(make_partitioner()).initialize(X, families, random_state=42)

        for j, cluster in enumerate(mean_ordered_clusters):
            points = X[H[:, cluster] > 0]
            parameters = _components.all_parameters(start[j])
            assert parameters["mu"] == pytest.approx(points.mean(), rel=1e-12)
            assert parameters["sigma"] == pytest.approx(np.std(points), rel=1e-12)
            assert start.weights[j] == pytest.approx(len(points) / len(X), rel=1e-12)

    @pytest.mark.filterwarnings("error")
    def test_segments_have_equal_sizes_and_determine_statistics(self) -> None:
        """Segment means, scales, rates and sizes determine the parameters and weights."""

        X = example_sample()
        families = [family(FamilyName.NORMAL), family(FamilyName.NORMAL), family(FamilyName.EXPONENTIAL)]
        segments = np.array_split(np.sort(X), 3)

        start = ClusterInitializer(QuantilePartitioner()).initialize(X, families, random_state=42)

        assert [len(segment) for segment in segments] == [234, 233, 233]
        assert start.weights == pytest.approx([234 / 700, 233 / 700, 233 / 700], rel=1e-12)
        for j in range(2):
            parameters = _components.all_parameters(start[j])
            assert parameters["mu"] == pytest.approx(segments[j].mean(), rel=1e-12)
            assert parameters["sigma"] == pytest.approx(np.std(segments[j]), rel=1e-12)
        assert _components.all_parameters(start[2])["lambda_"] == pytest.approx(1 / segments[2].mean(), rel=1e-12)

    @pytest.mark.filterwarnings("error")
    def test_every_component_matches_exactly_one_cluster_statistic(self) -> None:
        """Every family receives the only cluster whose closed-form statistics are its parameters."""

        X = normal_exponential_uniform_components()
        families = make_normal_exponential_uniform_families()
        H = RandomCentersPartitioner().partition(X, 3, np.random.default_rng(1))

        start = ClusterInitializer(RandomCentersPartitioner()).initialize(X, families, random_state=1)

        matched_clusters = []
        for j in range(len(families)):
            parameters = _components.all_parameters(start[j])
            matches = []
            for cluster in range(3):
                points = X[H[:, cluster] > 0]
                if j == 0:
                    expected = {"mu": float(points.mean()), "sigma": float(np.std(points))}
                elif j == 1:
                    expected = {"lambda_": float(1 / points.mean())}
                else:
                    expected = {"lower_bound": float(points.min()), "upper_bound": float(points.max())}
                if all(parameters[name] == pytest.approx(value, rel=1e-12) for name, value in expected.items()):
                    matches.append(cluster)

            assert len(matches) == 1
            cluster = matches[0]
            matched_clusters.append(cluster)
            assert start.weights[j] == pytest.approx(np.count_nonzero(H[:, cluster]) / len(X), rel=1e-12)

        assert len(set(matched_clusters)) == len(families)
