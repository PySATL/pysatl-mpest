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
from pysatl_core.families import ParametricFamily, ParametricFamilyRegister
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel, _components
from pysatl_mpest.estimators import ECM, ECME, EM
from pysatl_mpest.estimators.iterative import PriorThresholdPruner, StepBreakpointer
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
    RefiningInitializer,
)
from pysatl_mpest.optimizers import ScipyNelderMead
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


def four_modes() -> UnivariateFloatArray:
    """A sample from four normal components, two of them small."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(-6, 1, 200), rng.normal(-2, 1, 60), rng.normal(3, 1, 200), rng.normal(7, 1, 60)])


def distant_small_mode() -> UnivariateFloatArray:
    """A sample with a small normal component far from the main one."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(0, 1, 300), rng.normal(100, 1, 30)])


def make_em(iterations: int) -> EM:
    """An EM estimator with the given number of iterations."""

    return EM([StepBreakpointer(iterations)], [], ScipyNelderMead())


def make_em3() -> EM:
    """An EM estimator with three iterations."""

    return make_em(3)


def make_ecm3() -> ECM:
    """An ECM estimator with three iterations."""

    return ECM([StepBreakpointer(3)], [], ScipyNelderMead())


def make_ecme3() -> ECME:
    """An ECME estimator with three iterations."""

    return ECME([StepBreakpointer(3)], [], ScipyNelderMead())


class TestRefinedRuns:
    """Multistart over runs refined by a mixture estimator."""

    @pytest.mark.filterwarnings("error")
    def test_choice_uses_fitted_mixture_quality(self) -> None:
        X = four_modes()
        families = [family(FamilyName.NORMAL) for _ in range(4)]
        refined = RefiningInitializer(ClusterInitializer(RandomCentersPartitioner()), make_em(10))

        result = MultiStartInitializer(refined, 8).initialize(X, families, random_state=0)
        expected, _ = manual_best(refined, X, families, 0, 8)

        plain_best, _ = manual_best(ClusterInitializer(RandomCentersPartitioner()), X, families, 0, 8)
        refined_plain_best = make_em(10).fit(X, plain_best)

        assert result == expected
        assert refined_plain_best.loglikelihood(X) < result.loglikelihood(X)

    @pytest.mark.parametrize(
        ("make_estimator", "fit_kwargs", "n_init"),
        [
            pytest.param(make_em3, {}, 4, id="em-4-runs"),
            pytest.param(
                make_ecm3,
                {"parameter_groups": {0: [{"sigma"}, {"mu"}]}},
                4,
                id="ecm-parameter-groups-4-runs",
            ),
            pytest.param(make_ecme3, {}, 2, id="ecme-2-runs"),
        ],
    )
    @pytest.mark.filterwarnings("error")
    def test_matches_manual_best_with_each_estimator(
        self,
        make_estimator: Callable[[], EM | ECM | ECME],
        fit_kwargs: dict[str, object],
        n_init: int,
    ) -> None:
        X = example_sample()
        families = example_families()
        refined = RefiningInitializer(ClusterInitializer(KMeansPartitioner()), make_estimator(), fit_kwargs=fit_kwargs)

        result = MultiStartInitializer(refined, n_init).initialize(X, families, random_state=42)
        expected, _ = manual_best(refined, X, families, 42, n_init)

        assert result == expected
        assert result.n_components == len(families)
        for j in range(len(families)):
            assert _components.component_family_key(result[j]) == _components.family_key(families[j])

    def test_run_with_pruned_component_is_skipped(self) -> None:
        X = distant_small_mode()
        families = [family(FamilyName.NORMAL), family(FamilyName.NORMAL)]
        estimator = EM([StepBreakpointer(2)], [PriorThresholdPruner(0.2)], ScipyNelderMead())
        refined = RefiningInitializer(ClusterInitializer(RandomCentersPartitioner()), estimator)
        n_init = 10
        expected, failed = manual_best(refined, X, families, 42, n_init)
        assert 0 < failed < n_init

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = MultiStartInitializer(refined, n_init).initialize(X, families, random_state=42)

        assert result == expected
        assert len(caught) == 1
        assert caught[0].category is RuntimeWarning
        expected_warning = f"{failed} of 10 runs failed; the last reason: The estimator returned 1 of 2 components."
        assert str(caught[0].message) == expected_warning

    def test_pruned_component_in_every_run_raises(self) -> None:
        X = distant_small_mode()
        families = [family(FamilyName.NORMAL), family(FamilyName.NORMAL)]
        estimator = EM([StepBreakpointer(2)], [PriorThresholdPruner(0.2)], ScipyNelderMead())
        refined = RefiningInitializer(ClusterInitializer(KMeansPartitioner()), estimator)
        message = "All 10 runs failed; the last reason: The estimator returned 1 of 2 components."

        with pytest.raises(InitializationError, match=rf"^{re.escape(message)}$") as exc_info:
            MultiStartInitializer(refined, 10).initialize(X, families, random_state=42)

        error = exc_info.value
        assert type(error) is InitializationError
        assert str(error) == message
        assert type(error.__cause__) is InitializationError

    @pytest.mark.filterwarnings("error")
    def test_estimator_error_interrupts_multistart(self) -> None:
        X = example_sample()
        families = example_families()
        estimator = ECM([StepBreakpointer(3)], [], ScipyNelderMead())
        refined = RefiningInitializer(
            ClusterInitializer(KMeansPartitioner()), estimator, fit_kwargs={"parameter_groups": {0: [{"mu"}]}}
        )

        with pytest.raises(ValueError, match="^Invalid parameter_groups"):
            MultiStartInitializer(refined, 3).initialize(X, families, random_state=42)

    @pytest.mark.filterwarnings(r"ignore:\d+ of \d+ runs failed:RuntimeWarning")
    @pytest.mark.filterwarnings("error")
    def test_integer_seed_reproduces_composition_after_other_seeded_call(self) -> None:
        X = example_sample()
        families = example_families()
        initializer = MultiStartInitializer(
            RefiningInitializer(ClusterInitializer(RandomCentersPartitioner()), make_em(3)), 3
        )

        first = initializer.initialize(X, families, random_state=42)
        initializer.initialize(X, families, random_state=7)
        third = MultiStartInitializer(
            RefiningInitializer(ClusterInitializer(RandomCentersPartitioner()), make_em(3)), 3
        ).initialize(X, families, random_state=42)

        assert first == third

    @pytest.mark.filterwarnings(r"ignore:\d+ of \d+ runs failed:RuntimeWarning")
    @pytest.mark.filterwarnings("error")
    def test_reverse_composition_refines_only_best_start(self) -> None:
        X = example_sample()
        families = example_families()
        multistart = MultiStartInitializer(ClusterInitializer(RandomCentersPartitioner()), 4)
        estimator = make_em(3)

        result = RefiningInitializer(multistart, estimator).initialize(X, families, random_state=42)
        start = multistart.initialize(X, families, random_state=42)
        expected = estimator.fit(X, start)

        assert result == expected


@settings(deadline=None)
@pytest.mark.filterwarnings(r"ignore:\d+ of \d+ runs failed:RuntimeWarning")
@pytest.mark.filterwarnings("error")
@given(
    inputs=partition_inputs(distinct_values=True),
    family_indices=st.lists(st.integers(0, len(FACTORIES) - 1), min_size=MAX_CLUSTERS, max_size=MAX_CLUSTERS),
    partitioner_index=st.integers(0, len(PARTITIONER_FACTORIES) - 1),
    matcher_index=st.integers(0, len(MATCHER_FACTORIES) - 1),
    n_init=st.integers(1, 3),
)
def test_refined_cluster_initializers_return_mixture_or_raise_initialization_error(
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
    refined = RefiningInitializer(inner, make_em(2))

    try:
        mixture = MultiStartInitializer(refined, n_init).initialize(X, families, random_state=seed)
    except InitializationError as error:
        assert type(error) is InitializationError
        assert isinstance(error.__cause__, InitializationError)
        return

    assert mixture.n_components == k


def four_distant_modes() -> UnivariateFloatArray:
    """A sample from four distant equal-sized normal components."""

    rng = np.random.default_rng(42)
    return np.concatenate(
        [rng.normal(-15, 1, 150), rng.normal(-5, 1, 150), rng.normal(5, 1, 150), rng.normal(15, 1, 150)]
    )


def stuck_sensor() -> UnivariateFloatArray:
    """A sample from two wide normal components and a narrow block of stuck-sensor values."""

    rng = np.random.default_rng(42)
    return np.concatenate(
        [rng.normal(0, 1000, 1000), rng.normal(6000, 1500, 1000), 15000 + rng.uniform(-0.001, 0.001, 40)]
    )


def outlier_sample() -> UnivariateFloatArray:
    """A sample from one normal component with a single distant outlier."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(0, 1, 20), [100.0]])


def example_truth() -> MixtureModel:
    """The true mixture of the example sample."""

    normal = family(FamilyName.NORMAL)
    expon = family(FamilyName.EXPONENTIAL)
    return MixtureModel(
        [normal(mu=-8, sigma=1), normal(mu=-3, sigma=0.7), expon(lambda_=0.5)], weights=[2 / 7, 2 / 7, 3 / 7]
    )


def distant_modes_truth() -> MixtureModel:
    """The true mixture of the four distant modes sample."""

    normal = family(FamilyName.NORMAL)
    return MixtureModel([normal(mu=m, sigma=1) for m in (-15, -5, 5, 15)], weights=[0.25] * 4)


def assert_example_tolerances(mixture: MixtureModel) -> None:
    """Check that the chain result is close to the truth of the example sample."""

    params = [_components.all_parameters(mixture[j]) for j in range(3)]
    assert params[0]["mu"] == pytest.approx(-8, abs=0.2)
    assert params[1]["mu"] == pytest.approx(-3, abs=0.2)
    assert params[2]["lambda_"] == pytest.approx(0.5, abs=0.06)
    assert list(mixture.weights) == pytest.approx([2 / 7, 2 / 7, 3 / 7], abs=0.01)


class TestTaskExamples:
    """The examples of the multistart use from the statement of the module design."""

    @pytest.mark.filterwarnings("error")
    def test_start_selection_example_recovers_true_mixture(self) -> None:
        """The first example of the design selects a start from which EM recovers the true mixture."""

        X = example_sample()

        normal = ParametricFamilyRegister.get(FamilyName.NORMAL)
        expon = ParametricFamilyRegister.get(FamilyName.EXPONENTIAL)

        initializer = ClusterInitializer(KMeansPartitioner())

        # The best of 10 runs by the likelihood of the starting mixture.
        start = MultiStartInitializer(initializer, n_init=10).initialize(X, [normal, normal, expon], random_state=42)

        result = EM(breakpointers=[StepBreakpointer(100)], pruners=[], optimizer=ScipyNelderMead()).fit(X, start)

        truth = example_truth()
        assert result.n_components == 3  # noqa: PLR2004
        for j, fam in enumerate([normal, normal, expon]):
            assert _components.component_family_key(result[j]) == _components.family_key(fam)
        assert_example_tolerances(result)
        assert result.loglikelihood(X) >= start.loglikelihood(X)
        assert result.loglikelihood(X) >= truth.loglikelihood(X)

    @pytest.mark.filterwarnings("error")
    def test_refined_selection_example_is_not_worse_than_plain(self) -> None:
        """The second example of the design selects a start not worse than the first one and EM recovers the truth."""

        X = example_sample()

        normal = ParametricFamilyRegister.get(FamilyName.NORMAL)
        expon = ParametricFamilyRegister.get(FamilyName.EXPONENTIAL)

        initializer = ClusterInitializer(KMeansPartitioner())

        # The best of 10 runs by the likelihood of the starting mixture.
        start = MultiStartInitializer(initializer, n_init=10).initialize(X, [normal, normal, expon], random_state=42)
        plain_score = start.loglikelihood(X)

        # The best of 10 runs by the result of a short EM.
        short_em = EM(breakpointers=[StepBreakpointer(20)], pruners=[], optimizer=ScipyNelderMead())
        refined = RefiningInitializer(initializer, short_em)
        start = MultiStartInitializer(refined, n_init=10).initialize(X, [normal, normal, expon], random_state=42)

        result = EM(breakpointers=[StepBreakpointer(100)], pruners=[], optimizer=ScipyNelderMead()).fit(X, start)

        truth = example_truth()
        assert start.loglikelihood(X) >= plain_score
        assert result.n_components == 3  # noqa: PLR2004
        for j, fam in enumerate([normal, normal, expon]):
            assert _components.component_family_key(result[j]) == _components.family_key(fam)
        assert_example_tolerances(result)
        assert result.loglikelihood(X) >= start.loglikelihood(X)
        assert result.loglikelihood(X) >= truth.loglikelihood(X)

    @pytest.mark.filterwarnings("error")
    def test_both_examples_lead_final_em_to_same_mixture(self) -> None:
        """The final EM reaches the same mixture from the starts of both examples of the design."""

        X = example_sample()

        normal = ParametricFamilyRegister.get(FamilyName.NORMAL)
        expon = ParametricFamilyRegister.get(FamilyName.EXPONENTIAL)

        initializer = ClusterInitializer(KMeansPartitioner())

        # The best of 10 runs by the likelihood of the starting mixture.
        start = MultiStartInitializer(initializer, n_init=10).initialize(X, [normal, normal, expon], random_state=42)
        first = EM(breakpointers=[StepBreakpointer(100)], pruners=[], optimizer=ScipyNelderMead()).fit(X, start)

        # The best of 10 runs by the result of a short EM.
        short_em = EM(breakpointers=[StepBreakpointer(20)], pruners=[], optimizer=ScipyNelderMead())
        refined = RefiningInitializer(initializer, short_em)
        start = MultiStartInitializer(refined, n_init=10).initialize(X, [normal, normal, expon], random_state=42)
        second = EM(breakpointers=[StepBreakpointer(100)], pruners=[], optimizer=ScipyNelderMead()).fit(X, start)

        assert first.loglikelihood(X) == pytest.approx(second.loglikelihood(X), abs=1e-6)
        for j in range(3):
            assert _components.all_parameters(first[j]) == pytest.approx(
                _components.all_parameters(second[j]), abs=1e-6
            )
        assert list(first.weights) == pytest.approx(list(second.weights), abs=1e-6)

    @pytest.mark.filterwarnings("error")
    def test_fixed_mean_survives_refined_example_chain(self) -> None:
        """A mean fixed through a view stays exact through the chain of the second example."""

        X = example_sample()

        normal = ParametricFamilyRegister.get(FamilyName.NORMAL)
        expon = ParametricFamilyRegister.get(FamilyName.EXPONENTIAL)

        initializer = ClusterInitializer(KMeansPartitioner())

        families = [normal.view(mu=-8), normal, expon]

        # The best of 10 runs by the result of a short EM.
        short_em = EM(breakpointers=[StepBreakpointer(20)], pruners=[], optimizer=ScipyNelderMead())
        refined = RefiningInitializer(initializer, short_em)
        start = MultiStartInitializer(refined, n_init=10).initialize(X, families, random_state=42)

        result = EM(breakpointers=[StepBreakpointer(100)], pruners=[], optimizer=ScipyNelderMead()).fit(X, start)

        params = [_components.all_parameters(result[j]) for j in range(3)]
        assert params[0]["mu"] == -8  # noqa: PLR2004
        assert params[1]["mu"] == pytest.approx(-3, abs=0.2)
        assert params[2]["lambda_"] == pytest.approx(0.5, abs=0.06)
        assert list(result.weights) == pytest.approx([2 / 7, 2 / 7, 3 / 7], abs=0.01)


class TestLocalOptima:
    """Escape from the local optima of single refined runs by the multistart selection."""

    @pytest.mark.filterwarnings(r"ignore:\d+ of \d+ runs failed:RuntimeWarning")
    @pytest.mark.filterwarnings("error")
    def test_selection_after_preliminary_iterations_recovers_all_modes(self) -> None:
        """The best of short refined runs escapes the local optima of single runs and leads EM to all modes."""

        X = four_distant_modes()
        families = [family(FamilyName.NORMAL) for _ in range(4)]
        inner = ClusterInitializer(RandomCentersPartitioner())
        refined = RefiningInitializer(inner, make_em(5))

        start = MultiStartInitializer(refined, n_init=10).initialize(X, families, random_state=42)

        single_scores = []
        for child in np.random.default_rng(42).spawn(10):
            try:
                single = refined.initialize(X, families, random_state=child)
            except InitializationError:
                continue
            single_scores.append(single.loglikelihood(X))
        assert any(score < start.loglikelihood(X) - 100 for score in single_scores)

        result = make_em(100).fit(X, start)
        truth = distant_modes_truth()

        mus = sorted(_components.all_parameters(result[j])["mu"] for j in range(4))
        assert mus == pytest.approx([-15, -5, 5, 15], abs=0.3)
        assert list(result.weights) == pytest.approx([0.25] * 4, abs=0.02)
        assert result.loglikelihood(X) >= truth.loglikelihood(X)

    @pytest.mark.filterwarnings(r"ignore:\d+ of \d+ runs failed:RuntimeWarning")
    @pytest.mark.filterwarnings("error")
    def test_selection_of_completed_runs_needs_no_further_fitting(self) -> None:
        """The best of full refined runs is already the fitted mixture of all modes."""

        X = four_distant_modes()
        families = [family(FamilyName.NORMAL) for _ in range(4)]
        inner = ClusterInitializer(RandomCentersPartitioner())
        refined = RefiningInitializer(inner, make_em(30))

        result = MultiStartInitializer(refined, n_init=10).initialize(X, families, random_state=42)

        single_scores = []
        for child in np.random.default_rng(42).spawn(10):
            try:
                single = refined.initialize(X, families, random_state=child)
            except InitializationError:
                continue
            single_scores.append(single.loglikelihood(X))
        assert any(score < result.loglikelihood(X) - 100 for score in single_scores)

        truth = distant_modes_truth()
        mus = sorted(_components.all_parameters(result[j])["mu"] for j in range(4))
        assert mus == pytest.approx([-15, -5, 5, 15], abs=0.3)
        assert list(result.weights) == pytest.approx([0.25] * 4, abs=0.02)
        assert result.loglikelihood(X) >= truth.loglikelihood(X)

        refitted = make_em(30).fit(X, result)
        assert abs(refitted.loglikelihood(X) - result.loglikelihood(X)) < 1e-6  # noqa: PLR2004


class TestDegenerateRuns:
    """Skipping of degenerate runs and quality of the winning start."""

    def test_too_narrow_cluster_runs_fail_and_do_not_win(self) -> None:
        """Runs with a too narrow cluster are skipped although such a start would win by likelihood."""

        X = stuck_sensor()
        families = [family(FamilyName.NORMAL), family(FamilyName.NORMAL)]
        inner = ClusterInitializer(RandomCentersPartitioner())

        errors = []
        for child in np.random.default_rng(13).spawn(10):
            try:
                inner.initialize(X, families, random_state=child)
            except InitializationError as error:
                errors.append(error)
        assert len(errors) > 0
        for error in errors:
            assert isinstance(error, ComponentEstimationError)
            assert "interquartile width" in str(error)
        last = errors[-1]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = MultiStartInitializer(inner, 10).initialize(X, families, random_state=13)

        assert len(caught) == 1
        assert caught[0].category is RuntimeWarning
        assert str(caught[0].message) == f"{len(errors)} of 10 runs failed; the last reason: {last}"

        for j in range(2):
            assert _components.all_parameters(result[j])["sigma"] > 100  # noqa: PLR2004

        normal = family(FamilyName.NORMAL)
        ordinary = X[:2000]
        stuck = X[2000:]
        manual = MixtureModel(
            [normal(mu=ordinary.mean(), sigma=np.std(ordinary)), normal(mu=stuck.mean(), sigma=np.std(stuck))],
            weights=[2000 / 2040, 40 / 2040],
        )
        assert manual.loglikelihood(X) > result.loglikelihood(X)

        fitted = make_em(30).fit(X, result)
        assert _components.all_parameters(fitted[0])["mu"] == pytest.approx(0, abs=500)
        assert _components.all_parameters(fitted[1])["mu"] == pytest.approx(6000, abs=500)

    def test_too_small_cluster_runs_are_skipped(self) -> None:
        """Runs with a too small cluster are skipped with one warning."""

        X = outlier_sample()
        families = [family(FamilyName.NORMAL), family(FamilyName.NORMAL)]
        inner = ClusterInitializer(RandomCentersPartitioner())

        errors = []
        successes = 0
        for child in np.random.default_rng(0).spawn(10):
            try:
                inner.initialize(X, families, random_state=child)
                successes += 1
            except InitializationError as error:
                errors.append(error)
        assert successes > 0
        assert len(errors) > 0
        for error in errors:
            assert str(error).startswith("The partition has clusters with fewer than 4 points")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = MultiStartInitializer(inner, 10).initialize(X, families, random_state=0)

        assert len(caught) == 1
        assert caught[0].category is RuntimeWarning
        assert str(caught[0].message) == f"{len(errors)} of 10 runs failed; the last reason: {errors[-1]}"
        assert all(w >= 4 / 21 for w in result.weights)
