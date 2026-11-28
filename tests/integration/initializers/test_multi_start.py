"""Integration tests of ``MultiStartInitializer`` with real cluster initializers."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import re
import warnings
from collections.abc import Callable, Sequence

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from pysatl_core.families import ParametricFamily
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel, _components
from pysatl_mpest.initializers import (
    ClusterInitializer,
    ComponentEstimationError,
    HungarianMatcher,
    IdentityMatcher,
    InitializationError,
    Initializer,
    KMeansPartitioner,
    Matcher,
    MultiStartInitializer,
    Partitioner,
    QuantilePartitioner,
    RandomCentersPartitioner,
)
from pysatl_mpest.typings import UnivariateFloatArray

from tests.helpers.families import family
from tests.helpers.partitions import MAX_CLUSTERS, partition_inputs


def make_kmeans_partitioner() -> Partitioner:
    """The k-means partitioning method."""

    return KMeansPartitioner()


def make_quantile_partitioner() -> Partitioner:
    """The quantile partitioning method."""

    return QuantilePartitioner()


def make_random_centers_partitioner() -> Partitioner:
    """The random-centers partitioning method."""

    return RandomCentersPartitioner()


def make_hungarian_matcher() -> Matcher:
    """The exact matcher."""

    return HungarianMatcher()


def make_identity_matcher() -> Matcher:
    """The matcher that keeps the order of the columns."""

    return IdentityMatcher()


def make_quantile_cluster_initializer() -> ClusterInitializer:
    """The cluster initializer with the deterministic partitioning method."""

    return ClusterInitializer(QuantilePartitioner())


def make_identity_quantile_cluster_initializer() -> ClusterInitializer:
    """The cluster initializer with deterministic partitioning and the order-keeping matcher."""

    return ClusterInitializer(QuantilePartitioner(), matcher=IdentityMatcher())


def three_modes() -> UnivariateFloatArray:
    """A sample from three well-separated normal components."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(-10, 1, 150), rng.normal(0, 1, 150), rng.normal(10, 1, 150)])


def example_sample() -> UnivariateFloatArray:
    """A sample from two normal components and one exponential component."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(-8, 1, 200), rng.normal(-3, 0.7, 200), rng.exponential(2, 300)])


def three_normal_families() -> list[ParametricFamily]:
    """Three plain normal families."""

    return [family(FamilyName.NORMAL) for _ in range(3)]


def example_families() -> list[ParametricFamily]:
    """Families of the components of the example sample."""

    return [family(FamilyName.NORMAL), family(FamilyName.NORMAL), family(FamilyName.EXPONENTIAL)]


def manual_best(
    inner: Initializer, X: UnivariateFloatArray, families: Sequence[ParametricFamily], seed: int, n_init: int
) -> tuple[MixtureModel, int]:
    """Repeat the nested initializer manually and return the best mixture and the number of failures."""

    best: MixtureModel | None = None
    best_score = -np.inf
    failed = 0

    for child in np.random.default_rng(seed).spawn(n_init):
        try:
            mixture = inner.initialize(X, families, random_state=child)
        except InitializationError:
            failed += 1
            continue

        score = mixture.loglikelihood(X)
        if score > best_score:
            best_score = score
            best = mixture

    assert best is not None
    return best, failed


PARTITIONER_FACTORIES: tuple[Callable[[], Partitioner], ...] = (
    make_kmeans_partitioner,
    make_random_centers_partitioner,
)
MATCHER_FACTORIES: tuple[Callable[[], Matcher], ...] = (make_hungarian_matcher, make_identity_matcher)
FACTORIES: tuple[Callable[[], ParametricFamily], ...] = (
    lambda: family(FamilyName.NORMAL),
    lambda: family(FamilyName.EXPONENTIAL),
    lambda: family(FamilyName.CONTINUOUS_UNIFORM),
    lambda: family(FamilyName.NORMAL).view(mu=0),
)


class TestSelection:
    """Choice of the best starting mixture among the runs."""

    @pytest.mark.parametrize(
        ("make_sample", "make_families", "make_partitioner"),
        [
            pytest.param(
                three_modes,
                three_normal_families,
                make_random_centers_partitioner,
                id="three-modes-random-centers",
            ),
            pytest.param(
                example_sample,
                example_families,
                make_random_centers_partitioner,
                id="example-random-centers",
            ),
            pytest.param(example_sample, example_families, make_kmeans_partitioner, id="example-kmeans"),
        ],
    )
    @pytest.mark.filterwarnings(r"ignore:\d+ of \d+ runs failed:RuntimeWarning")
    def test_returns_best_mixture_of_manual_runs(
        self,
        make_sample: Callable[[], UnivariateFloatArray],
        make_families: Callable[[], list[ParametricFamily]],
        make_partitioner: Callable[[], Partitioner],
    ) -> None:
        X = make_sample()
        families = make_families()
        inner = ClusterInitializer(make_partitioner())

        result = MultiStartInitializer(inner, 10).initialize(X, families, random_state=42)
        expected, _ = manual_best(inner, X, families, 42, 10)

        assert result == expected
        assert result.loglikelihood(X) == expected.loglikelihood(X)

    @pytest.mark.parametrize(
        "make_partitioner",
        [
            pytest.param(make_kmeans_partitioner, id="kmeans"),
            pytest.param(make_random_centers_partitioner, id="random-centers"),
            pytest.param(make_quantile_partitioner, id="quantile"),
        ],
    )
    @pytest.mark.filterwarnings(r"ignore:\d+ of \d+ runs failed:RuntimeWarning")
    def test_result_is_a_valid_starting_mixture(self, make_partitioner: Callable[[], Partitioner]) -> None:
        X = example_sample()
        families = example_families()

        mixture = MultiStartInitializer(ClusterInitializer(make_partitioner()), 5).initialize(
            X, families, random_state=42
        )

        assert mixture.n_components == len(families)
        for j in range(len(families)):
            assert _components.component_family_key(mixture[j]) == _components.family_key(families[j])
        assert np.isfinite(mixture.loglikelihood(X))

    @pytest.mark.filterwarnings("error")
    def test_deterministic_nested_initializer_repeats_single_call(self) -> None:
        X = example_sample()
        families = example_families()
        inner = ClusterInitializer(QuantilePartitioner())
        expected = inner.initialize(X, families)

        for n_init in (1, 3):
            for seed in (0, 42):
                result = MultiStartInitializer(inner, n_init).initialize(X, families, random_state=seed)

                assert result == expected

    def test_nested_multistart_is_reproducible(self) -> None:
        X = three_modes()
        families = three_normal_families()
        inner = MultiStartInitializer(ClusterInitializer(RandomCentersPartitioner()), 2)
        initializer = MultiStartInitializer(inner, 3)

        first = initializer.initialize(X, families, random_state=42)
        second = initializer.initialize(X, families, random_state=42)

        assert first.n_components == len(families)
        assert first == second


class TestRandomState:
    """Reproducibility of the multistart and use of a shared generator."""

    @pytest.mark.parametrize(
        "make_partitioner",
        [
            pytest.param(make_random_centers_partitioner, id="random-centers"),
            pytest.param(make_kmeans_partitioner, id="kmeans"),
        ],
    )
    def test_integer_seed_reproduces_result_after_other_seeded_call(
        self, make_partitioner: Callable[[], Partitioner]
    ) -> None:
        X = three_modes()
        families = three_normal_families()
        initializer = MultiStartInitializer(ClusterInitializer(make_partitioner()), 4)

        first = initializer.initialize(X, families, random_state=42)
        initializer.initialize(X, families, random_state=7)
        third = MultiStartInitializer(ClusterInitializer(make_partitioner()), 4).initialize(
            X, families, random_state=42
        )

        assert first == third

    @pytest.mark.filterwarnings(r"ignore:\d+ of \d+ runs failed:RuntimeWarning")
    def test_shared_generator_gives_different_results(self) -> None:
        X = example_sample()
        families = example_families()
        rng = np.random.default_rng(42)
        initializer = MultiStartInitializer(ClusterInitializer(RandomCentersPartitioner()), 3)

        first = initializer.initialize(X, families, random_state=rng)
        second = initializer.initialize(X, families, random_state=rng)

        assert first != second

    @pytest.mark.filterwarnings(r"ignore:\d+ of \d+ runs failed:RuntimeWarning")
    def test_more_runs_are_not_worse_with_same_seed(self) -> None:
        X = example_sample()
        families = example_families()
        scores = []

        for n_init in (1, 2, 3, 5, 8, 13):
            mixture = MultiStartInitializer(ClusterInitializer(RandomCentersPartitioner()), n_init).initialize(
                X, families, random_state=42
            )
            scores.append(mixture.loglikelihood(X))

        assert all(later >= earlier for earlier, later in zip(scores, scores[1:]))


class TestFailedRuns:
    """Reporting of failed runs and passing input errors through."""

    def test_partial_failures_warn_once_and_keep_best_success(self) -> None:
        X = example_sample()
        families = example_families()
        inner = ClusterInitializer(RandomCentersPartitioner())
        expected, failed = manual_best(inner, X, families, 42, 10)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = MultiStartInitializer(inner, 10).initialize(X, families, random_state=42)

        assert result == expected
        assert len(caught) == 1
        assert caught[0].category is RuntimeWarning
        text = str(caught[0].message)
        pattern = (
            rf"{failed} of 10 runs failed; the last reason: "
            r"No assignment gives every family a suitable cluster\. Impossible pairs:\n.+"
        )
        assert re.fullmatch(pattern, text, re.DOTALL)

    @pytest.mark.parametrize(
        ("make_inner", "cause_type", "reason_start"),
        [
            pytest.param(
                make_quantile_cluster_initializer,
                InitializationError,
                "No assignment",
                id="no-assignment",
            ),
            pytest.param(
                make_identity_quantile_cluster_initializer,
                ComponentEstimationError,
                None,
                id="component-estimation",
            ),
        ],
    )
    def test_all_failures_raise_with_last_reason_as_cause(
        self,
        make_inner: Callable[[], ClusterInitializer],
        cause_type: type[InitializationError],
        reason_start: str | None,
    ) -> None:
        X = np.asarray([-9, -8, -7, -6, -3, 1, 2, 3], dtype=np.float64)
        families = [family(FamilyName.EXPONENTIAL), family(FamilyName.NORMAL)]

        with pytest.raises(InitializationError, match="^All 3 runs failed; the last reason: ") as exc_info:
            MultiStartInitializer(make_inner(), 3).initialize(X, families, random_state=42)

        error = exc_info.value
        assert type(error) is InitializationError
        assert re.fullmatch(r"All 3 runs failed; the last reason: .+", str(error), re.DOTALL)
        assert type(error.__cause__) is cause_type
        if reason_start is not None:
            assert str(error.__cause__).startswith(reason_start)

    @pytest.mark.parametrize(
        ("make_sample", "make_families", "error_type", "reason_start"),
        [
            pytest.param(
                three_modes,
                lambda: [family(FamilyName.NORMAL).view(parametrization_name="meanVar"), family(FamilyName.NORMAL)],
                NotImplementedError,
                "MLEComponentEstimator does not support",
                id="unsupported-parametrization",
            ),
            pytest.param(
                lambda: np.asarray([1, 2, 3, 4, 10, 11, 12], dtype=np.float64),
                lambda: [family(FamilyName.NORMAL), family(FamilyName.EXPONENTIAL)],
                ValueError,
                "X has 7 points",
                id="too-few-points",
            ),
        ],
    )
    @pytest.mark.filterwarnings("error")
    def test_input_errors_pass_through_without_warning(
        self,
        make_sample: Callable[[], UnivariateFloatArray],
        make_families: Callable[[], list[ParametricFamily]],
        error_type: type[Exception],
        reason_start: str,
    ) -> None:
        initializer = MultiStartInitializer(ClusterInitializer(QuantilePartitioner()), 3)

        with pytest.raises(error_type, match=rf"^{re.escape(reason_start)}"):
            initializer.initialize(make_sample(), make_families(), random_state=42)


@settings(deadline=None)
@pytest.mark.filterwarnings(r"ignore:\d+ of \d+ runs failed:RuntimeWarning")
@pytest.mark.filterwarnings("error")
@given(
    inputs=partition_inputs(distinct_values=True),
    family_indices=st.lists(st.integers(0, len(FACTORIES) - 1), min_size=MAX_CLUSTERS, max_size=MAX_CLUSTERS),
    partitioner_index=st.integers(0, len(PARTITIONER_FACTORIES) - 1),
    matcher_index=st.integers(0, len(MATCHER_FACTORIES) - 1),
    n_init=st.integers(1, 4),
)
def test_real_cluster_initializers_return_mixture_or_raise_initialization_error(
    inputs: tuple[UnivariateFloatArray, int, int],
    family_indices: list[int],
    partitioner_index: int,
    matcher_index: int,
    n_init: int,
) -> None:
    X, k, seed = inputs
    assume(np.std(X) > 0 or len(np.unique(X)) == 1)
    families = [FACTORIES[index]() for index in family_indices[:k]]
    partitioner = PARTITIONER_FACTORIES[partitioner_index]()
    matcher = MATCHER_FACTORIES[matcher_index]()
    inner = ClusterInitializer(partitioner, matcher=matcher, min_cluster_size=1)

    try:
        mixture = MultiStartInitializer(inner, n_init).initialize(X, families, random_state=seed)
    except InitializationError as error:
        assert type(error) is InitializationError
        assert isinstance(error.__cause__, InitializationError)
        return

    assert mixture.n_components == k
