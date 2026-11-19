"""Integration tests for the ECM estimator."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.core import _components as cc
from pysatl_mpest.estimators import ECM
from pysatl_mpest.estimators.iterative import LikelihoodBreakpointer, PriorThresholdPruner, StepBreakpointer
from pysatl_mpest.optimizers import Optimizer, ScipyNelderMead, ScipyPowell
from scipy.special import logsumexp

from tests.helpers.estimator_data import (
    N_SAMPLES,
    PRUNING_THRESHOLD,
    TRUE_WEIGHTS,
    ecm_for,
    exponential,
    fitted_arrays,
    normal,
    parameters,
    sample_of,
    stray_start,
)
from tests.helpers.families import family
from tests.helpers.reference_em import reference_em

LIKELIHOOD_SLACK = 1e-9  # A numerical tolerance for a likelihood that should not decrease.


def normal_exponential(mu: float, sigma: float) -> ParametricFamilyDistribution:
    """A Normal component in the ``exponential`` parametrization: ``a = -1/(2*sigma**2)``, ``b = mu/sigma**2``."""
    a = -1.0 / (2 * sigma**2)
    b = mu / sigma**2
    return family(FamilyName.NORMAL)(parametrization_name="exponential", a=a, b=b)


@pytest.fixture
def exponential_normal_sample() -> tuple[MixtureModel, np.ndarray]:
    truth = MixtureModel([exponential(1.5), normal(8.0, 1.0)], weights=[0.5, 0.5])
    return truth, sample_of(truth)


def pruning_ecm(iterations: int = 15) -> ECM:
    return ECM(
        breakpointers=[StepBreakpointer(iterations)],
        pruners=[PriorThresholdPruner(PRUNING_THRESHOLD)],
        optimizer=ScipyNelderMead(),
    )


class TestFitting:
    @pytest.mark.parametrize("iterations", [1, 5, 25])
    def test_reproduces_the_reference_em_iteration_by_iteration(
        self, X: np.ndarray, start: MixtureModel, iterations: int
    ) -> None:
        fitted = ecm_for(iterations).fit(X, start)

        for actual, expected in zip(fitted_arrays(fitted), reference_em(X, start, iterations)):
            assert_allclose(actual, expected, rtol=1e-9)

    def test_history_records_hold_distinct_mixtures_matching_reference_em(
        self, X: np.ndarray, start: MixtureModel
    ) -> None:
        iterations = 5
        ecm = ecm_for(iterations)

        fitted = ecm.fit(X, start)

        records = [ecm.history[i] for i in range(len(ecm.history))]
        assert len(records) == iterations
        assert len({id(record.mixture) for record in records}) == iterations
        assert all(record.mixture is not start for record in records)
        assert fitted is records[-1].mixture
        for i, record in enumerate(records):
            for actual, expected in zip(fitted_arrays(record.mixture), reference_em(X, start, i + 1)):
                assert_allclose(actual, expected, rtol=1e-9)

    def test_recovers_the_parameters_of_a_mixture(self, X: np.ndarray, start: MixtureModel) -> None:
        fitted = ecm_for(300, tolerance=1e-9).fit(X, start)
        mu, sigma, weights = fitted_arrays(fitted)

        assert_allclose(mu, [0.0, 6.0], atol=0.15)
        assert_allclose(sigma, [1.0, 1.5], atol=0.15)
        assert_allclose(weights, TRUE_WEIGHTS, atol=0.03)

    def test_never_decreases_the_log_likelihood(self, X: np.ndarray, start: MixtureModel) -> None:
        ecm = ecm_for(40)
        fitted = ecm.fit(X, start)

        likelihoods = [record.mixture.loglikelihood(X) for record in (ecm.history[i] for i in range(len(ecm.history)))]

        assert len(likelihoods) == 40  # noqa: PLR2004
        assert np.all(np.diff(likelihoods) >= -LIKELIHOOD_SLACK)
        assert likelihoods[-1] > start.loglikelihood(X)
        assert fitted.loglikelihood(X) == pytest.approx(likelihoods[-1])

    def test_reaches_a_likelihood_not_below_the_truth(
        self, X: np.ndarray, true_mixture: MixtureModel, start: MixtureModel
    ) -> None:
        fitted = ecm_for(300, tolerance=1e-9).fit(X, start)

        assert fitted.loglikelihood(X) >= true_mixture.loglikelihood(X) - 1e-6

    def test_keeps_a_parameter_fixed_through_a_view(self, X: np.ndarray) -> None:
        fixed = family(FamilyName.NORMAL).view(mu=0.0)(sigma=2.0)
        start = MixtureModel([fixed, normal(4.0, 2.0)], weights=[0.5, 0.5])

        fitted = ecm_for(25).fit(X, start)

        assert cc.fixed_parameters(fitted[0]) == {"mu": 0.0}
        assert cc.all_parameters(fitted[0])["mu"] == 0.0
        expected = reference_em(X, start, 25, fixed_mu=(True, False))
        for actual, reference in zip(fitted_arrays(fitted), expected):
            assert_allclose(actual, reference, rtol=1e-9)

    def test_does_not_modify_the_initial_mixture(self, X: np.ndarray, start: MixtureModel) -> None:
        components_before = start.components
        log_weights_before = start.log_weights.copy()

        ecm_for(5).fit(X, start)

        assert start.components == components_before
        assert np.array_equal(start.log_weights, log_weights_before)

    def test_is_deterministic(self, X: np.ndarray, start: MixtureModel) -> None:
        assert ecm_for(10).fit(X, start) == ecm_for(10).fit(X, start)

    def test_recovers_a_mixture_of_an_exponential_and_a_normal(
        self,
        exponential_normal_sample: tuple[MixtureModel, np.ndarray],
    ) -> None:
        _, X = exponential_normal_sample
        start = MixtureModel([exponential(0.5), normal(6.0, 2.0)], weights=[0.5, 0.5])

        fitted = ecm_for(300, tolerance=1e-9).fit(X, start)

        assert cc.all_parameters(fitted[0])["lambda_"] == pytest.approx(1.5, rel=0.1)
        assert cc.all_parameters(fitted[1]) == pytest.approx({"mu": 8.0, "sigma": 1.0}, abs=0.15)
        assert_allclose(fitted.weights, [0.5, 0.5], atol=0.03)


class TestStopping:
    def test_step_breakpointer_sets_the_number_of_iterations(self, X: np.ndarray, start: MixtureModel) -> None:
        ecm = ecm_for(7)

        ecm.fit(X, start)

        assert len(ecm.history) == 7  # noqa: PLR2004

    def test_likelihood_breakpointer_stops_before_the_step_limit(self, X: np.ndarray, start: MixtureModel) -> None:
        ecm = ecm_for(500, tolerance=1e-6)

        fitted = ecm.fit(X, start)

        assert 1 < len(ecm.history) < 500  # noqa: PLR2004
        assert_allclose(fitted_arrays(fitted)[2], TRUE_WEIGHTS, atol=0.03)

    def test_repeated_fits_start_from_a_clean_breakpoint_state(self) -> None:
        """Fitting the same ECM object repeatedly must give identical results."""
        rng = np.random.default_rng(42)
        truth = MixtureModel([exponential(0.5), exponential(2.0)], weights=[0.5, 0.5])
        X = np.asarray(truth.generate(size=600, random_state=rng), dtype=np.float64)
        start = MixtureModel([exponential(0.5), exponential(3.0)])
        ecm = ECM(
            breakpointers=[LikelihoodBreakpointer(1e-3), StepBreakpointer(50)],
            pruners=[],
            optimizer=ScipyNelderMead(),
        )

        iteration_counts = []
        fitted_mixtures = []
        for _ in range(3):
            # Each fit creates a fresh pipeline, so the history holds this fit's iterations only.
            fitted_mixtures.append(ecm.fit(X, start))
            iteration_counts.append(len(ecm.history))

        assert iteration_counts[0] == iteration_counts[1] == iteration_counts[2]
        assert fitted_mixtures[1] == fitted_mixtures[0]
        assert fitted_mixtures[2] == fitted_mixtures[0]


class TestNumericalOptimization:
    @pytest.mark.parametrize("optimizer_class", [ScipyNelderMead, ScipyPowell], ids=["nelder-mead", "powell"])
    def test_with_numerical_strategy_matches_the_analytical_one(
        self, X: np.ndarray, start: MixtureModel, optimizer_class: type[Optimizer]
    ) -> None:
        mean_var = family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=1.0, var=4.0)
        numerical_start = MixtureModel([mean_var, start[1]], weights=[0.5, 0.5])
        iterations = 15

        numerical = ecm_for(iterations, optimizer=optimizer_class()).fit(X, numerical_start)
        analytical = ecm_for(iterations).fit(X, start)

        numerical_params, analytical_params = parameters(numerical), parameters(analytical)
        assert numerical[0].parametrization_name == "meanVar"
        assert numerical_params[0]["mu"] == pytest.approx(analytical_params[0]["mu"], abs=5e-3)
        assert numerical_params[0]["var"] == pytest.approx(analytical_params[0]["sigma"] ** 2, rel=5e-3)
        assert numerical_params[1] == pytest.approx(analytical_params[1], rel=5e-3)
        assert_allclose(numerical.weights, analytical.weights, atol=5e-3)

    def test_with_numerical_strategy_keeps_a_fixed_parameter(self, X: np.ndarray, start: MixtureModel) -> None:
        fixed = family(FamilyName.NORMAL).view(parametrization_name="meanVar", mu=0.0)(var=4.0)
        numerical_start = MixtureModel([fixed, start[1]], weights=[0.5, 0.5])

        fitted = ecm_for(10).fit(X, numerical_start)

        assert cc.all_parameters(fitted[0])["mu"] == 0.0
        assert cc.all_parameters(fitted[0])["var"] != cc.all_parameters(fixed)["var"]


class TestPruning:
    @pytest.mark.parametrize("position", [0, 1, 2], ids=["first", "middle", "last"])
    def test_removes_a_component_with_a_negligible_weight(self, X: np.ndarray, position: int) -> None:
        fitted = pruning_ecm().fit(X, stray_start((position,)))

        mu, sigma, weights = fitted_arrays(fitted)
        assert fitted.n_components == 2  # noqa: PLR2004
        assert_allclose(mu, [0.0, 6.0], atol=0.15)
        assert_allclose(sigma, [1.0, 1.5], atol=0.15)
        assert_allclose(weights, TRUE_WEIGHTS, atol=0.03)

    @pytest.mark.parametrize("positions", [(0, 2), (1, 3), (0, 1)], ids=["0 and 2", "1 and 3", "0 and 1"])
    def test_removes_several_components_at_once(self, X: np.ndarray, positions: tuple[int, ...]) -> None:
        fitted = pruning_ecm().fit(X, stray_start(positions))

        assert fitted.n_components == 2  # noqa: PLR2004
        assert_allclose(fitted_arrays(fitted)[0], [0.0, 6.0], atol=0.15)

    def test_continues_like_a_fit_of_the_pruned_mixture(self, X: np.ndarray) -> None:
        start = stray_start((1,))
        after_first = ecm_for(1).fit(X, start)
        assert after_first.weights[1] < PRUNING_THRESHOLD
        pruned = after_first.without_component(1)

        with_pruner = pruning_ecm(iterations=10).fit(X, start)
        by_hand = ecm_for(9).fit(X, pruned)

        for actual, expected in zip(fitted_arrays(with_pruner), fitted_arrays(by_hand)):
            assert_allclose(actual, expected, rtol=1e-9)

    def test_keeps_the_history_consistent(self, X: np.ndarray) -> None:
        ecm = pruning_ecm(iterations=6)

        ecm.fit(X, stray_start((1,)))

        records = [ecm.history[i] for i in range(len(ecm.history))]
        assert [record.mixture.n_components for record in records][-1] == 2  # noqa: PLR2004
        for record in records:
            assert record.H is not None
            assert record.H.shape == (N_SAMPLES, record.mixture.n_components)

    def test_keeps_a_fixed_parameter(self, X: np.ndarray) -> None:
        fixed = family(FamilyName.NORMAL).view(mu=6.0)(sigma=2.0)
        start = MixtureModel([normal(0.0, 1.0), normal(3.0, 1.0), fixed], weights=[0.49, 0.02, 0.49])

        fitted = pruning_ecm().fit(X, start)

        assert fitted.n_components == 2  # noqa: PLR2004
        assert cc.fixed_parameters(fitted[1]) == {"mu": 6.0}
        assert cc.all_parameters(fitted[1])["sigma"] == pytest.approx(1.5, abs=0.15)


class TestUnsupportedData:
    def test_fit_with_a_point_outside_the_support_keeps_pdf_equal_to_exp_lpdf(self) -> None:
        start = MixtureModel([exponential(1.0), exponential(3.0)])
        data = np.array([-1.0, -2.0, 0.5, 1.0, 2.0, 0.1])

        fitted = ecm_for(1).fit(data, start)

        assert np.sum(np.exp(fitted.log_weights)) == pytest.approx(1.0)
        on_support = data[2:]
        assert fitted.pdf(on_support) == pytest.approx(np.exp(fitted.lpdf(on_support)))

    def test_likelihood_breakpointer_converges_with_an_unsupported_point(self) -> None:
        truth = MixtureModel([exponential(1.5), exponential(6.0)], weights=[0.4, 0.6])
        clean = np.asarray(truth.generate(size=600, random_state=42), dtype=np.float64)
        start = MixtureModel([exponential(0.5), exponential(3.0)])
        step_limit = 500
        reference = ecm_for(step_limit, tolerance=1e-6)
        with_outlier = ecm_for(step_limit, tolerance=1e-6)

        clean_fit = reference.fit(clean, start)
        outlier_fit = with_outlier.fit(np.append(clean, -0.5), start)

        assert 1 < len(reference.history) < step_limit
        assert len(with_outlier.history) == len(reference.history)
        for actual, expected in zip(parameters(outlier_fit), parameters(clean_fit), strict=True):
            assert actual == pytest.approx(expected, rel=1e-9)
        assert_allclose(outlier_fit.weights, clean_fit.weights, rtol=1e-9)
        assert np.isneginf(outlier_fit.loglikelihood(np.append(clean, -0.5)))


class TestParameterGroups:
    """Conditional maximization via ``parameter_groups``."""

    @pytest.mark.parametrize("iterations", [1, 5])
    def test_single_group_covering_all_free_parameters_reproduces_em(
        self, X: np.ndarray, start: MixtureModel, iterations: int
    ) -> None:
        # One group per component is the same maximization problem as EM's joint step.
        groups = {0: [{"mu", "sigma"}], 1: [{"mu", "sigma"}]}
        fitted = ecm_for(iterations).fit(X, start, parameter_groups=groups)

        for actual, expected in zip(fitted_arrays(fitted), reference_em(X, start, iterations)):
            assert_allclose(actual, expected, rtol=1e-9)

    def test_groups_cover_only_free_parameters_and_keep_fixed_ones(self, X: np.ndarray) -> None:
        fixed_a = -0.5
        fixed = family(FamilyName.NORMAL).view(parametrization_name="exponential", a=fixed_a)(b=1.0)
        start = MixtureModel([fixed, normal_exponential(3.0, 1.0)], weights=[0.5, 0.5])
        ecm = ECM(breakpointers=[StepBreakpointer(1)], pruners=[], optimizer=ScipyNelderMead())

        fitted = ecm.fit(X, start, parameter_groups={0: [{"b"}], 1: [{"a"}, {"b"}]})

        assert cc.fixed_parameters(fitted[0]) == {"a": fixed_a}
        assert cc.all_parameters(fitted[0])["a"] == fixed_a

    def test_conditional_steps_match_the_closed_form_reference(self) -> None:
        """One iteration of true ECM on Normal ``exponential`` components.

        With groups ``[{"a"}, {"b"}]`` the block for ``a`` maximizes Q at the old
        ``b``, then the block for ``b`` maximizes Q at the new ``a``. The numpy
        reference below performs the same closed-form conditional steps; the
        result must differ from EM's joint maximizer and from updating both
        groups off a single snapshot by more than the Nelder-Mead tolerance.
        """
        rng = np.random.default_rng(42)
        truth = MixtureModel([normal(0.0, 1.0), normal(6.0, 1.5)], weights=TRUE_WEIGHTS)
        X = np.asarray(truth.generate(size=N_SAMPLES, random_state=rng), dtype=np.float64)

        start_ab = [(-0.5, 2.0), (-0.5, 3.0)]  # (mu, sigma) = (2, 1) and (3, 1)
        start = MixtureModel([normal_exponential(2.0, 1.0), normal_exponential(3.0, 1.0)], weights=[0.5, 0.5])

        ecm = ECM(breakpointers=[StepBreakpointer(1)], pruners=[], optimizer=ScipyNelderMead())
        fitted = ecm.fit(X, start, parameter_groups={0: [{"a"}, {"b"}], 1: [{"a"}, {"b"}]})

        # Reference iteration on numpy: soft responsibilities, weights, and the
        # conditional maximizers of Q ~ a*m2 + b*m1 + b**2/(4*a) + 0.5*log(-a).
        def lpdf(a: float, b: float) -> np.ndarray:
            return a * X**2 + b * X + b**2 / (4 * a) - 0.5 * np.log(np.pi / (-a))

        log_joint = np.stack([lpdf(a, b) for a, b in start_ab], axis=1) + np.log(start.weights)
        H = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))
        total = H.sum(axis=0)
        m1 = (H * X[:, None]).sum(axis=0) / total
        m2 = (H * X[:, None] ** 2).sum(axis=0) / total

        tolerance = 1e-3  # Nelder-Mead accuracy on the conditional maximizers.
        min_gap = 0.05
        assert_allclose(fitted.weights, total / len(X), atol=tolerance)

        for component, (a_old, b_old), m1_j, m2_j in zip(fitted, start_ab, m1, m2, strict=True):
            a_ecm = (-1.0 - np.sqrt(1.0 + 4.0 * m2_j * b_old**2)) / (4.0 * m2_j)
            b_ecm = -2.0 * a_ecm * m1_j
            actual = cc.free_parameters(component)

            assert actual["a"] == pytest.approx(a_ecm, abs=tolerance)
            assert actual["b"] == pytest.approx(b_ecm, abs=tolerance)

            # EM's joint maximizer from the same snapshot: mu = m1, sigma**2 = m2 - m1**2.
            variance = m2_j - m1_j**2
            a_em, b_em = -1.0 / (2 * variance), m1_j / variance
            assert max(abs(actual["a"] - a_em), abs(actual["b"] - b_em)) > min_gap

            # Both groups updated from a single snapshot differ in ``b``.
            b_simultaneous = -2.0 * a_old * m1_j
            assert abs(actual["b"] - b_simultaneous) > min_gap
