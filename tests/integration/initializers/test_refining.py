"""Integration tests of ``RefiningInitializer`` with real mixture estimators."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from pysatl_core.families import ParametricFamily
from pysatl_core.types import FamilyName
from pysatl_mpest.core._components import all_parameters, component_family_key, family_key
from pysatl_mpest.estimators import ECM, ECME, EM
from pysatl_mpest.estimators.iterative import PriorThresholdPruner, StepBreakpointer
from pysatl_mpest.initializers import (
    ClusterInitializer,
    InitializationError,
    KMeansPartitioner,
    QuantilePartitioner,
    RandomCentersPartitioner,
    RefiningInitializer,
)
from pysatl_mpest.optimizers import ScipyNelderMead
from pysatl_mpest.typings import UnivariateFloatArray

from tests.helpers.families import family


def example_sample() -> UnivariateFloatArray:
    """A sample from two normal components and one exponential component."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(-8, 1, 200), rng.normal(-3, 0.7, 200), rng.exponential(2, 300)])


def modes_zero_ten_sample() -> UnivariateFloatArray:
    """A sample from two normal components with modes zero and ten."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(0, 1, 300), rng.normal(10, 1, 300)])


def distant_small_mode_sample() -> UnivariateFloatArray:
    """A sample with a small normal component far from the main mode."""

    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(0, 1, 300), rng.normal(100, 1, 30)])


def normal_family() -> ParametricFamily:
    """The registered normal family."""

    return family(FamilyName.NORMAL)


def exponential_family() -> ParametricFamily:
    """The registered exponential family."""

    return family(FamilyName.EXPONENTIAL)


def example_families() -> list[ParametricFamily]:
    """Families of the components of the example sample."""

    return [normal_family(), normal_family(), exponential_family()]


def two_normal_families() -> list[ParametricFamily]:
    """Two plain normal families."""

    return [normal_family(), normal_family()]


def exponential_then_two_normal_families() -> list[ParametricFamily]:
    """One exponential family followed by two normal families."""

    return [exponential_family(), normal_family(), normal_family()]


def make_em3() -> EM:
    """An EM estimator that runs three iterations."""

    return EM([StepBreakpointer(3)], [], ScipyNelderMead())


def make_ecm3() -> ECM:
    """An ECM estimator that runs three iterations."""

    return ECM([StepBreakpointer(3)], [], ScipyNelderMead())


def make_ecme3() -> ECME:
    """An ECME estimator that runs three iterations."""

    return ECME([StepBreakpointer(3)], [], ScipyNelderMead())


class TestRefinement:
    """Refinement of real cluster initializer starts by real mixture estimators."""

    @pytest.mark.parametrize(
        "make_estimator",
        [
            pytest.param(make_em3, id="em"),
            pytest.param(make_ecm3, id="ecm"),
            pytest.param(make_ecme3, id="ecme"),
        ],
    )
    def test_matches_manual_fit_and_preserves_family_order(self, make_estimator: Callable[[], EM | ECM | ECME]) -> None:
        X = example_sample()
        families = example_families()
        inner = ClusterInitializer(KMeansPartitioner())
        estimator = make_estimator()

        result = RefiningInitializer(inner, estimator).initialize(X, families, random_state=42)

        start = inner.initialize(X, families, random_state=42)
        expected = estimator.fit(X, start)
        assert result == expected
        assert result.n_components == len(families)
        for j in range(len(families)):
            assert component_family_key(result[j]) == family_key(families[j])

    def test_improves_starting_loglikelihood(self) -> None:
        X = example_sample()
        families = example_families()
        inner = ClusterInitializer(QuantilePartitioner())
        estimator = EM([StepBreakpointer(20)], [], ScipyNelderMead())

        result = RefiningInitializer(inner, estimator).initialize(X, families, random_state=42)
        start = inner.initialize(X, families, random_state=42)

        assert result.loglikelihood(X) > start.loglikelihood(X)

    def test_keeps_fixed_parameter_value(self) -> None:
        X = modes_zero_ten_sample()
        normal = normal_family()
        families = [normal.view(mu=0), normal]
        inner = ClusterInitializer(KMeansPartitioner())
        estimator = EM([StepBreakpointer(5)], [], ScipyNelderMead())

        mixture = RefiningInitializer(inner, estimator).initialize(X, families, random_state=42)

        assert all_parameters(mixture[0])["mu"] == 0

    def test_leaves_generator_as_nested_initializer_leaves_it(self) -> None:
        X = example_sample()
        families = example_families()
        inner = ClusterInitializer(RandomCentersPartitioner())
        refining = RefiningInitializer(inner, make_em3())
        first_generator = np.random.default_rng(42)
        second_generator = np.random.default_rng(42)

        refining.initialize(X, families, random_state=first_generator)
        inner.initialize(X, families, random_state=second_generator)

        assert first_generator.bit_generator.state == second_generator.bit_generator.state

    def test_integer_seed_reproduces_result(self) -> None:
        X = example_sample()
        families = example_families()
        inner = ClusterInitializer(RandomCentersPartitioner())
        refining = RefiningInitializer(inner, make_em3())

        first = refining.initialize(X, families, random_state=42)
        second = refining.initialize(X, families, random_state=42)

        assert first == second


class TestFitKwargs:
    """Passing keyword arguments of the estimator through the initializer."""

    def test_once_in_iterations_controls_history_records(self) -> None:
        X = example_sample()
        families = example_families()
        estimator = EM([StepBreakpointer(10)], [], ScipyNelderMead())
        refining = RefiningInitializer(
            ClusterInitializer(KMeansPartitioner()), estimator, fit_kwargs={"once_in_iterations": 5}
        )

        refining.initialize(X, families, random_state=42)

        assert len(estimator.history) == 2  # noqa: PLR2004

    @pytest.mark.parametrize(
        ("make_estimator", "make_fit_kwargs"),
        [
            pytest.param(
                make_ecm3,
                lambda: {"parameter_groups": {0: [{"sigma"}, {"mu"}]}},
                id="ecm-parameter-groups",
            ),
            pytest.param(
                make_ecme3,
                lambda: {"q_indices_raw": [0, 1, 2]},
                id="ecme-strategy-indices",
            ),
        ],
    )
    def test_keyword_values_change_result(
        self, make_estimator: Callable[[], ECM | ECME], make_fit_kwargs: Callable[[], dict[str, Any]]
    ) -> None:
        X = example_sample()
        families = example_families()
        inner = ClusterInitializer(RandomCentersPartitioner())
        fit_kwargs = make_fit_kwargs()

        result = RefiningInitializer(inner, make_estimator(), fit_kwargs=fit_kwargs).initialize(
            X, families, random_state=42
        )

        start = inner.initialize(X, families, random_state=42)
        expected = make_estimator().fit(X, start, **fit_kwargs)
        without_keywords = make_estimator().fit(X, start)
        assert result == expected
        assert result != without_keywords

    @pytest.mark.parametrize(
        ("make_estimator", "make_fit_kwargs", "bad_name"),
        [
            pytest.param(make_em3, lambda: {"parameter_groups": {}}, "parameter_groups", id="em-parameter-groups"),
            pytest.param(make_ecm3, lambda: {"q_indices_raw": [0]}, "q_indices_raw", id="ecm-strategy-indices"),
            pytest.param(make_ecme3, lambda: {"strategy": "q-func"}, "strategy", id="ecme-strategy"),
            pytest.param(make_em3, lambda: {"X": None}, "X", id="em-sample"),
            pytest.param(make_ecm3, lambda: {"mixture": None}, "mixture", id="ecm-mixture"),
        ],
    )
    def test_rejects_invalid_name_at_creation(
        self,
        make_estimator: Callable[[], EM | ECM | ECME],
        make_fit_kwargs: Callable[[], dict[str, Any]],
        bad_name: str,
    ) -> None:
        estimator = make_estimator()
        match = rf"^fit_kwargs do not match the signature of {type(estimator).__name__}\.fit: "

        with pytest.raises(TypeError, match=match) as exc_info:
            RefiningInitializer(ClusterInitializer(KMeansPartitioner()), estimator, fit_kwargs=make_fit_kwargs())

        assert bad_name in str(exc_info.value)

    @pytest.mark.parametrize(
        ("make_estimator", "make_fit_kwargs"),
        [
            pytest.param(make_em3, lambda: {"once_in_iterations": 2}, id="em-once-in-iterations"),
            pytest.param(
                make_ecm3,
                lambda: {"once_in_iterations": 2, "parameter_groups": {}},
                id="ecm-history-and-groups",
            ),
            pytest.param(
                make_ecme3,
                lambda: {"q_indices_raw": [0], "odl_indices_raw": [1, 2]},
                id="ecme-strategy-indices",
            ),
        ],
    )
    def test_accepts_valid_names_at_creation(
        self, make_estimator: Callable[[], EM | ECM | ECME], make_fit_kwargs: Callable[[], dict[str, Any]]
    ) -> None:
        fit_kwargs = make_fit_kwargs()

        initializer = RefiningInitializer(
            ClusterInitializer(KMeansPartitioner()), make_estimator(), fit_kwargs=fit_kwargs
        )

        assert initializer.fit_kwargs == fit_kwargs

    def test_invalid_value_surfaces_from_estimator_at_call(self) -> None:
        X = example_sample()
        families = example_families()
        estimator = make_ecm3()
        refining = RefiningInitializer(
            ClusterInitializer(KMeansPartitioner()), estimator, fit_kwargs={"parameter_groups": {0: [{"mu"}]}}
        )

        with pytest.raises(ValueError, match="^Invalid parameter_groups"):
            refining.initialize(X, families, random_state=42)

    def test_group_component_numbers_follow_family_order(self) -> None:
        X = example_sample()
        families = exponential_then_two_normal_families()
        fit_kwargs = {"parameter_groups": {1: [{"sigma"}, {"mu"}], 2: [{"mu"}, {"sigma"}]}}
        successful_seeds = []

        for seed in range(10):
            refining = RefiningInitializer(
                ClusterInitializer(RandomCentersPartitioner()), make_ecm3(), fit_kwargs=fit_kwargs
            )
            try:
                refining.initialize(X, families, random_state=seed)
            except InitializationError:
                continue
            successful_seeds.append(seed)

        assert successful_seeds


class TestRemovedComponents:
    """Failure of a run whose estimator removes a component."""

    @pytest.mark.parametrize("seed", [pytest.param(seed, id=f"seed-{seed}") for seed in range(5)])
    def test_pruned_run_fails_with_exact_message(self, seed: int) -> None:
        X = distant_small_mode_sample()
        families = two_normal_families()
        estimator = EM([StepBreakpointer(1)], [PriorThresholdPruner(0.2)], ScipyNelderMead())
        refining = RefiningInitializer(ClusterInitializer(KMeansPartitioner()), estimator)

        with pytest.raises(InitializationError, match=r"^The estimator returned 1 of 2 components\.$") as exc_info:
            refining.initialize(X, families, random_state=seed)

        assert type(exc_info.value) is InitializationError

    def test_lower_threshold_keeps_both_components(self) -> None:
        X = distant_small_mode_sample()
        families = two_normal_families()
        estimator = EM([StepBreakpointer(1)], [PriorThresholdPruner(0.05)], ScipyNelderMead())
        refining = RefiningInitializer(ClusterInitializer(KMeansPartitioner()), estimator)

        mixture = refining.initialize(X, families, random_state=0)

        assert mixture.n_components == 2  # noqa: PLR2004
