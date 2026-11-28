"""End-to-end scenarios of the initialization requirements INIT-02, INIT-03, INIT-04 and INIT-06."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable

import numpy as np
import pytest
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.initializers import (
    ClusterInitializer,
    InitializationError,
    MultiStartInitializer,
    Partitioner,
    QuantilePartitioner,
    RandomCentersPartitioner,
    RefiningInitializer,
    SklearnPartitioner,
)
from pysatl_mpest.typings import UnivariateFloatArray
from sklearn.cluster import AgglomerativeClustering

from tests.helpers.families import family
from tests.helpers.scenarios import (
    example_sample,
    make_em,
    make_quantile_partitioner,
    make_sklearn_agglomerative_partitioner,
    make_sklearn_spectral_partitioner,
)


def four_distant_modes() -> UnivariateFloatArray:
    """A sample from four distant equal-sized normal components."""

    rng = np.random.default_rng(42)
    return np.concatenate(
        [rng.normal(-15, 1, 150), rng.normal(-5, 1, 150), rng.normal(5, 1, 150), rng.normal(15, 1, 150)]
    )


def unequal_modes() -> UnivariateFloatArray:
    """A sample from three well-separated normal components with different weights."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(-10, 1, 100), rng.normal(0, 2, 200), rng.normal(10, 1, 300)])


def normal_exponential_uniform_components() -> UnivariateFloatArray:
    """A sample from well-separated normal, exponential and continuous uniform components."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(-10, 1, 100), rng.exponential(1, 100), rng.uniform(10, 20, 100)])


def distant_modes_truth() -> MixtureModel:
    """The true mixture of the four distant modes sample."""

    normal = family(FamilyName.NORMAL)
    return MixtureModel([normal(mu=m, sigma=1) for m in (-15, -5, 5, 15)], weights=[0.25] * 4)


def single_run_scores(refined: RefiningInitializer, X: UnivariateFloatArray, seed: int, n_init: int) -> list[float]:
    """Log-likelihoods of the successful single refined runs with the children of the seed."""

    families = [family(FamilyName.NORMAL) for _ in range(4)]
    scores = []
    for child in np.random.default_rng(seed).spawn(n_init):
        try:
            scores.append(refined.initialize(X, families, random_state=child).loglikelihood(X))
        except InitializationError:
            continue
    return scores


def assert_four_modes(mixture: MixtureModel, X: UnivariateFloatArray) -> None:
    """Check that the mixture has found all four distant modes."""

    mus = sorted(component.parameters["mu"] for component in mixture)
    assert mus == pytest.approx([-15, -5, 5, 15], abs=0.3)
    assert list(mixture.weights) == pytest.approx([0.25] * 4, abs=0.02)
    assert mixture.loglikelihood(X) >= distant_modes_truth().loglikelihood(X)


class TestLocalOptima:
    """INIT-02: the multistart escapes the local optima in which single runs end."""

    @pytest.mark.filterwarnings(r"ignore:\d+ of \d+ runs failed:RuntimeWarning")
    @pytest.mark.filterwarnings("error")
    def test_best_of_short_refined_runs_leads_em_to_all_modes(self) -> None:
        """The best of short refined runs escapes the local optima of single runs and leads EM to all modes."""

        X = four_distant_modes()
        families = [family(FamilyName.NORMAL) for _ in range(4)]
        refined = RefiningInitializer(ClusterInitializer(RandomCentersPartitioner()), make_em(5))

        start = MultiStartInitializer(refined, n_init=10).initialize(X, families, random_state=42)

        assert any(score < start.loglikelihood(X) - 100 for score in single_run_scores(refined, X, 42, 10))
        assert_four_modes(make_em(100).fit(X, start), X)

    @pytest.mark.filterwarnings(r"ignore:\d+ of \d+ runs failed:RuntimeWarning")
    @pytest.mark.filterwarnings("error")
    def test_best_of_full_refined_runs_is_already_the_fit(self) -> None:
        """The best of full refined runs is already the fitted mixture of all modes."""

        X = four_distant_modes()
        families = [family(FamilyName.NORMAL) for _ in range(4)]
        refined = RefiningInitializer(ClusterInitializer(RandomCentersPartitioner()), make_em(30))

        result = MultiStartInitializer(refined, n_init=10).initialize(X, families, random_state=42)

        assert any(score < result.loglikelihood(X) - 100 for score in single_run_scores(refined, X, 42, 10))
        assert_four_modes(result, X)
        assert make_em(30).fit(X, result).loglikelihood(X) == pytest.approx(result.loglikelihood(X), abs=1e-6)


class TestClusteringStart:
    """INIT-03: a start from agglomerative or spectral clustering."""

    @pytest.mark.parametrize(
        "make_partitioner",
        [
            pytest.param(make_sklearn_agglomerative_partitioner, id="sklearn-agglomerative"),
            pytest.param(make_sklearn_spectral_partitioner, id="sklearn-spectral"),
        ],
    )
    @pytest.mark.filterwarnings("error")
    def test_em_reaches_the_meaningful_optimum(self, make_partitioner: Callable[[], Partitioner]) -> None:
        """Thirty EM iterations from the start recover the unequal normal modes."""

        X = unequal_modes()
        families = [family(FamilyName.NORMAL) for _ in range(3)]
        start = ClusterInitializer(make_partitioner()).initialize(X, families, random_state=42)

        result = make_em(30).fit(X, start)

        assert [component.parameters["mu"] for component in result] == pytest.approx([-10, 0, 10], abs=0.3)
        assert [component.parameters["sigma"] for component in result] == pytest.approx([1, 2, 1], abs=0.35)
        assert list(result.weights) == pytest.approx([1 / 6, 1 / 3, 1 / 2], abs=0.03)

    @pytest.mark.filterwarnings("error")
    def test_agglomerative_start_ignores_randomness(self) -> None:
        """Agglomerative clustering gives exactly the same mixture for every source of randomness."""

        X = unequal_modes()
        initializer = ClusterInitializer(SklearnPartitioner(AgglomerativeClustering()))

        mixtures = [
            initializer.initialize(X, [family(FamilyName.NORMAL) for _ in range(3)], random_state=random_state)
            for random_state in (0, 5, None, np.random.default_rng(7))
        ]

        assert all(mixture == mixtures[0] for mixture in mixtures[1:])


class TestQuantileStart:
    """INIT-04: the quantile start and the fit from it are fully reproducible."""

    @pytest.mark.filterwarnings("error")
    def test_random_state_does_not_change_the_start(self) -> None:
        """Quantile partitioning gives exactly the same mixture for every source of randomness."""

        X = example_sample()
        families = [family(FamilyName.NORMAL), family(FamilyName.NORMAL), family(FamilyName.EXPONENTIAL)]
        initializer = ClusterInitializer(QuantilePartitioner())

        random_states = (0, 1, None, np.random.default_rng(7))

        mixtures = [initializer.initialize(X, families, random_state=random_state) for random_state in random_states]

        assert all(mixture == mixtures[0] for mixture in mixtures[1:])

    @pytest.mark.parametrize(
        "make_partitioner",
        [
            pytest.param(make_quantile_partitioner, id="quantile"),
            pytest.param(make_sklearn_agglomerative_partitioner, id="sklearn-agglomerative"),
        ],
    )
    @pytest.mark.filterwarnings("error")
    def test_deterministic_start_gives_the_same_fit(self, make_partitioner: Callable[[], Partitioner]) -> None:
        """A deterministic start refined by EM gives one fit for every source of randomness."""

        X = example_sample()
        families = [family(FamilyName.NORMAL), family(FamilyName.NORMAL), family(FamilyName.EXPONENTIAL)]
        refined = RefiningInitializer(ClusterInitializer(make_partitioner()), make_em(20))

        random_states = (0, 1, None, np.random.default_rng(7))

        fits = [refined.initialize(X, families, random_state=random_state) for random_state in random_states]

        assert all(fit == fits[0] for fit in fits[1:])
        assert fits[0][0].parameters["mu"] == pytest.approx(-8, abs=0.2)
        assert fits[0][1].parameters["mu"] == pytest.approx(-3, abs=0.2)


class TestRandomCentersStart:
    """INIT-06: random centers generate different starts for any families."""

    @pytest.mark.filterwarnings("error")
    def test_ten_seeds_give_at_least_five_different_starts(self) -> None:
        """Ten seeds all succeed on three different families and give at least five different weight vectors."""

        X = normal_exponential_uniform_components()
        families = [family(FamilyName.NORMAL), family(FamilyName.EXPONENTIAL), family(FamilyName.CONTINUOUS_UNIFORM)]
        initializer = ClusterInitializer(RandomCentersPartitioner())

        weights = {
            tuple(np.round(initializer.initialize(X, families, random_state=seed).weights, 12)) for seed in range(10)
        }

        assert len(weights) >= 5  # noqa: PLR2004
