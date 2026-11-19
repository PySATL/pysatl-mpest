"""Integration tests for the standalone EM estimator."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from pysatl_core.types import FamilyName
from pysatl_mpest import estimators
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.core import _components as cc
from pysatl_mpest.estimators import EM
from pysatl_mpest.estimators.iterative import (
    Breakpointer,
    LikelihoodBreakpointer,
    PriorThresholdPruner,
    Pruner,
    StepBreakpointer,
)
from pysatl_mpest.optimizers import ScipyNelderMead

from tests.helpers.estimator_data import (
    PRUNING_THRESHOLD,
    TRUE_WEIGHTS,
    fitted_arrays,
    normal,
    parameters,
    stray_start,
)
from tests.helpers.families import family
from tests.helpers.reference_em import reference_em


def em_for(
    iterations: int,
    pruners: Sequence[Pruner] | None = None,
    tolerance: float | None = None,
) -> EM:
    """An EM that runs exactly ``iterations`` iterations, or fewer if the likelihood stops changing."""

    breakpointers: list[Breakpointer] = [StepBreakpointer(iterations)]
    if tolerance is not None:
        breakpointers.append(LikelihoodBreakpointer(tolerance))
    return EM(breakpointers=breakpointers, pruners=pruners or [], optimizer=ScipyNelderMead())


def test_em_is_publicly_exported() -> None:
    assert "EM" in estimators.__all__
    assert estimators.EM is EM


class TestFitting:
    @pytest.mark.parametrize("iterations", [1, 5, 25])
    def test_reproduces_the_reference_em_iteration_by_iteration(
        self, X: NDArray[np.float64], start: MixtureModel, iterations: int
    ) -> None:
        fitted = em_for(iterations).fit(X, start)

        for actual, expected in zip(fitted_arrays(fitted), reference_em(X, start, iterations), strict=True):
            assert_allclose(actual, expected, rtol=1e-9)

    def test_keeps_a_parameter_fixed_through_a_view(self, X: NDArray[np.float64]) -> None:
        fixed = family(FamilyName.NORMAL).view(mu=0.0)(sigma=2.0)
        start = MixtureModel([fixed, normal(4.0, 2.0)], weights=[0.5, 0.5])
        parameters_before = parameters(start)
        weights_before = np.array(start.weights)

        fitted = em_for(25).fit(X, start)

        assert cc.fixed_parameters(fitted[0]) == {"mu": 0.0}
        assert cc.all_parameters(fitted[0])["mu"] == 0.0
        expected = reference_em(X, start, 25, fixed_mu=(True, False))
        for actual, reference in zip(fitted_arrays(fitted), expected, strict=True):
            assert_allclose(actual, reference, rtol=1e-9)
        assert parameters(start) == parameters_before
        assert_allclose(start.weights, weights_before)


class TestHistory:
    def test_records_every_n_iterations(self, X: NDArray[np.float64], start: MixtureModel) -> None:
        em = em_for(5)

        em.fit(X, start, once_in_iterations=2)

        assert [record.iteration for record in em.history] == [0, 2, 4]


class TestStopping:
    def test_likelihood_breakpointer_stops_before_the_step_limit(
        self, X: NDArray[np.float64], start: MixtureModel
    ) -> None:
        em = em_for(500, tolerance=1e-6)

        em.fit(X, start)

        assert 1 < len(em.history) < 500  # noqa: PLR2004


class TestNumericalOptimization:
    def test_with_numerical_strategy_matches_the_analytical_one(
        self, X: NDArray[np.float64], start: MixtureModel
    ) -> None:
        mean_var = family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=1.0, var=4.0)
        numerical_start = MixtureModel([mean_var, start[1]], weights=[0.5, 0.5])
        iterations = 15

        numerical = em_for(iterations).fit(X, numerical_start)
        analytical = em_for(iterations).fit(X, start)

        numerical_params, analytical_params = parameters(numerical), parameters(analytical)
        assert numerical[0].parametrization_name == "meanVar"
        assert numerical_params[0]["mu"] == pytest.approx(analytical_params[0]["mu"], abs=5e-3)
        assert numerical_params[0]["var"] == pytest.approx(analytical_params[0]["sigma"] ** 2, rel=5e-3)
        assert numerical_params[1] == pytest.approx(analytical_params[1], rel=5e-3)
        assert_allclose(numerical.weights, analytical.weights, atol=5e-3)


class TestPruning:
    def test_removes_a_component_with_a_negligible_weight(self, X: NDArray[np.float64]) -> None:
        em = em_for(15, pruners=[PriorThresholdPruner(PRUNING_THRESHOLD)])

        fitted = em.fit(X, stray_start((1,)))

        mu, sigma, weights = fitted_arrays(fitted)
        assert fitted.n_components == 2  # noqa: PLR2004
        assert_allclose(mu, [0.0, 6.0], atol=0.15)
        assert_allclose(sigma, [1.0, 1.5], atol=0.15)
        assert_allclose(weights, TRUE_WEIGHTS, atol=0.03)
