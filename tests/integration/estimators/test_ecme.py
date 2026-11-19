"""Integration tests for the ECME estimator."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators import ECME
from pysatl_mpest.estimators.iterative import PriorThresholdPruner, StepBreakpointer
from pysatl_mpest.optimizers import ScipyNelderMead

from tests.helpers.estimator_data import (
    PRUNING_THRESHOLD,
    TRUE_WEIGHTS,
    ecm_for,
    exponential,
    fitted_arrays,
    normal,
    parameters,
    stray_start,
)

MONOTONICITY_TOLERANCE = 1e-6  # Slack for numerical noise in the non-decrease assertion.
MIN_LOG_DENSITY_BOUND = -50.0  # Far above LOG_FLOOR = -1e4 of the ODL objective.


class TestStrategies:
    def test_with_q_function_strategy_equals_ecm(self, X: np.ndarray, start: MixtureModel) -> None:
        """Compare the ECME fit with ECM as the second estimator."""
        ecme = ECME(
            breakpointers=[StepBreakpointer(10)], pruners=[], optimizer=ScipyNelderMead(), default_strategy="q-func"
        )

        assert ecme.fit(X, start) == ecm_for(10).fit(X, start)

    @pytest.mark.parametrize(
        ("default_strategy", "q_indices", "odl_indices"),
        [("odl", None, None), ("q-func", None, 1), ("q-func", 0, 1)],
        ids=["odl everywhere", "odl for the second", "q-function and odl"],
    )
    def test_with_observed_data_likelihood_improves_the_fit(
        self,
        X: np.ndarray,
        start: MixtureModel,
        default_strategy: Literal["q-func", "odl"],
        q_indices: int | None,
        odl_indices: int | None,
    ) -> None:
        ecme = ECME(
            breakpointers=[StepBreakpointer(25)],
            pruners=[],
            optimizer=ScipyNelderMead(),
            default_strategy=default_strategy,
        )

        fitted = ecme.fit(X, start, q_indices_raw=q_indices, odl_indices_raw=odl_indices)

        assert fitted.loglikelihood(X) > start.loglikelihood(X)
        mu, sigma, weights = fitted_arrays(fitted)
        assert_allclose(mu, [0.0, 6.0], atol=0.2)
        assert_allclose(sigma, [1.0, 1.5], atol=0.2)
        assert_allclose(weights, TRUE_WEIGHTS, atol=0.05)
        assert len(ecme.history) == 25  # noqa: PLR2004


class TestUnsupportedData:
    def test_with_an_unsupported_point_matches_the_fit_on_clean_data(self) -> None:
        truth = MixtureModel([exponential(1.5), exponential(6.0)], weights=[0.4, 0.6])
        clean = np.asarray(truth.generate(size=600, random_state=42), dtype=np.float64)
        start = MixtureModel([exponential(0.5), exponential(3.0)])

        def estimator() -> ECME:
            """Create a fresh estimator with independent stopping state."""
            return ECME(
                breakpointers=[StepBreakpointer(50)],
                pruners=[],
                optimizer=ScipyNelderMead(),
                default_strategy="odl",
            )

        clean_fit = estimator().fit(clean, start)
        outlier_fit = estimator().fit(np.append(clean, -0.5), start)

        for actual, expected in zip(parameters(outlier_fit), parameters(clean_fit), strict=True):
            assert actual == pytest.approx(expected, rel=1e-3)
        assert_allclose(outlier_fit.weights, clean_fit.weights, rtol=1e-3)
        assert not np.allclose(
            [component["lambda_"] for component in parameters(outlier_fit)],
            [component["lambda_"] for component in parameters(start)],
        )


class TestLikelihoodMonotonicity:
    def test_loglikelihood_does_not_decrease_with_the_facade_block_order(self) -> None:
        """With soft E-step, no pruners and ScipyNelderMead the facade's order never decreases the likelihood.

        The data has a spike cluster, a diffuse cluster and a small side cluster, and the start puts a
        Q-function component between them with a weak far-away third component. With the old index-ordered
        blocks ``[ODL0, Q1, ODL2]`` the stale-background ODL update makes iteration 2 decrease the observed
        log-likelihood by about 2.35; the facade's order ``[Q1, ODL0, ODL2]`` must never decrease it.
        """
        rng = np.random.default_rng(42)
        X = np.concatenate(
            [
                rng.normal(0.349, 0.220, 44),
                rng.normal(3.218, 1.007, 42),
                rng.normal(-4.223, 1.011, 8),
            ]
        )
        start = MixtureModel(
            [normal(1.342, 1.056), normal(2.332, 1.897), normal(15.797, 1.025)],
            weights=[0.208, 0.778, 0.014],
        )

        ecme = ECME(
            breakpointers=[StepBreakpointer(10)], pruners=[], optimizer=ScipyNelderMead(), default_strategy="odl"
        )
        ecme.fit(X, start, q_indices_raw=1, odl_indices_raw=[0, 2])

        records = [ecme.history[i] for i in range(len(ecme.history))]
        assert len(records) == 10  # noqa: PLR2004
        likelihoods = [start.loglikelihood(X), *[record.mixture.loglikelihood(record.X) for record in records]]
        assert np.all(np.diff(likelihoods) >= -MONOTONICITY_TOLERANCE)
        # The ODL objective floors the mixture log-density at LOG_FLOOR = -1e4; it is never approached here.
        min_log_density = min(float(np.min(record.mixture.lpdf(record.X))) for record in records)
        assert min_log_density > MIN_LOG_DENSITY_BOUND


class TestPruning:
    @pytest.mark.parametrize("default_strategy", ["q-func", "odl"])
    def test_removes_a_component_with_a_negligible_weight(
        self, X: np.ndarray, default_strategy: Literal["q-func", "odl"]
    ) -> None:
        ecme = ECME(
            breakpointers=[StepBreakpointer(12)],
            pruners=[PriorThresholdPruner(PRUNING_THRESHOLD)],
            optimizer=ScipyNelderMead(),
            default_strategy=default_strategy,
        )

        fitted = ecme.fit(X, stray_start((0,)))

        assert fitted.n_components == 2  # noqa: PLR2004
        assert_allclose(fitted_arrays(fitted)[0], [0.0, 6.0], atol=0.2)
        assert_allclose(fitted.weights, TRUE_WEIGHTS, atol=0.05)
