"""Tests for the observed data likelihood strategy."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.core import _components as cc
from pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock
from pysatl_mpest.estimators.iterative._strategies import observed_data_likelihood_strategy
from pysatl_mpest.estimators.iterative._strategies.observed_data_likelihood import LOG_FLOOR
from pysatl_mpest.optimizers import Optimizer
from pytest_mock import MockerFixture
from scipy import stats
from scipy.special import logsumexp

from tests.helpers.families import family
from tests.helpers.pipeline import make_state

X = np.array([-1.0, 0.0, 0.5, 2.0, 4.5, 6.0])


def odl_block(component_id: int = 1, params: set[str] | None = None) -> OptimizationBlock:
    return OptimizationBlock(component_id, params, MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD)


def normal(mu: float, sigma: float) -> ParametricFamilyDistribution:
    return family(FamilyName.NORMAL)(mu=mu, sigma=sigma)


def negative_log_likelihood(weights: list[float], normals: list[tuple[float, float]], data: np.ndarray = X) -> float:
    """The negative log-likelihood of a mixture of normals, computed with scipy."""

    terms = [np.log(w) + stats.norm(mu, sigma).logpdf(data) for w, (mu, sigma) in zip(weights, normals)]
    return -float(np.sum(logsumexp(terms, axis=0)))


@pytest.fixture
def mixture() -> MixtureModel:
    return MixtureModel([normal(0.0, 1.0), normal(3.0, 2.0)], weights=[0.4, 0.6])


@pytest.fixture
def optimizer(mocker: MockerFixture) -> Optimizer:
    """An optimizer that returns the starting values, so that the target can be inspected."""
    mock = mocker.create_autospec(Optimizer, instance=True)
    mock.minimize.side_effect = lambda target, start: list(start)
    return mock


def captured_target(optimizer: Optimizer):
    return optimizer.minimize.call_args.args[0]  # type: ignore[attr-defined]


class TestObjective:
    """Check objective behavior."""

    def test_is_the_negative_log_likelihood_of_the_mixture(self, mixture: MixtureModel, optimizer: Optimizer) -> None:
        observed_data_likelihood_strategy(mixture[1], make_state(X, None, mixture), odl_block(), optimizer)
        target = captured_target(optimizer)

        assert target([2.0, 1.5]) == pytest.approx(
            negative_log_likelihood([0.4, 0.6], [(0.0, 1.0), (2.0, 1.5)]), rel=1e-12
        )

    def test_for_the_first_component_uses_the_others_as_background(
        self, mixture: MixtureModel, optimizer: Optimizer
    ) -> None:
        observed_data_likelihood_strategy(
            mixture[0], make_state(X, None, mixture), odl_block(component_id=0), optimizer
        )

        assert captured_target(optimizer)([-1.0, 0.5]) == pytest.approx(
            negative_log_likelihood([0.4, 0.6], [(-1.0, 0.5), (3.0, 2.0)]), rel=1e-12
        )

    def test_matches_single_component_log_likelihood(self, optimizer: Optimizer) -> None:
        mixture = MixtureModel([normal(0.0, 1.0)])

        observed_data_likelihood_strategy(
            mixture[0], make_state(X, None, mixture), odl_block(component_id=0), optimizer
        )

        assert captured_target(optimizer)([1.0, 2.0]) == pytest.approx(-float(np.sum(stats.norm(1.0, 2.0).logpdf(X))))

    def test_uses_the_normalized_weights_of_the_mixture(self, mixture: MixtureModel, optimizer: Optimizer) -> None:
        mixture = mixture.with_log_weights(np.log([2.0, 6.0]))

        observed_data_likelihood_strategy(mixture[1], make_state(X, None, mixture), odl_block(), optimizer)

        assert captured_target(optimizer)([2.0, 1.5]) == pytest.approx(
            negative_log_likelihood([0.25, 0.75], [(0.0, 1.0), (2.0, 1.5)]), rel=1e-12
        )

    def test_of_a_component_with_zero_weight_does_not_depend_on_it(self, optimizer: Optimizer) -> None:
        mixture = MixtureModel([normal(0.0, 1.0), normal(3.0, 2.0)], weights=[1.0, 0.0])

        observed_data_likelihood_strategy(mixture[1], make_state(X, None, mixture), odl_block(), optimizer)
        target = captured_target(optimizer)

        assert target([3.0, 2.0]) == pytest.approx(target([-5.0, 0.3]))

    def test_background_may_contain_components_with_fixed_parameters(self, optimizer: Optimizer) -> None:
        fixed_mu = family(FamilyName.NORMAL).view(mu=0.0)(sigma=1.0)
        mixture = MixtureModel([fixed_mu, normal(3.0, 2.0)], weights=[0.4, 0.6])

        observed_data_likelihood_strategy(mixture[1], make_state(X, None, mixture), odl_block(), optimizer)

        assert captured_target(optimizer)([2.0, 1.5]) == pytest.approx(
            negative_log_likelihood([0.4, 0.6], [(0.0, 1.0), (2.0, 1.5)]), rel=1e-12
        )

    def test_is_infinite_when_the_log_density_is_nan(self, optimizer: Optimizer) -> None:
        component = normal(0.0, 1.0)
        mixture = MixtureModel([component])
        data = np.array([np.nan])
        assert np.all(np.isnan(cc.lpdf(component, data)))

        observed_data_likelihood_strategy(
            component, make_state(data, None, mixture), odl_block(component_id=0), optimizer
        )

        with np.errstate(invalid="ignore"):
            assert captured_target(optimizer)([0.0, 1.0]) == np.inf

    def test_is_infinite_where_a_constraint_is_violated(self, mixture: MixtureModel, optimizer: Optimizer) -> None:
        observed_data_likelihood_strategy(mixture[1], make_state(X, None, mixture), odl_block(), optimizer)

        assert captured_target(optimizer)([2.0, -1.0]) == np.inf


class TestSupport:
    """Check support behavior."""

    def test_is_finite_and_exact_in_the_far_tail(self, mixture: MixtureModel, optimizer: Optimizer) -> None:
        far = np.array([-100.0, 100.0])
        assert np.all(mixture.pdf(far) == 0.0)

        observed_data_likelihood_strategy(mixture[1], make_state(far, None, mixture), odl_block(), optimizer)
        value = captured_target(optimizer)([3.0, 2.0])

        assert np.isfinite(value)
        assert value == pytest.approx(negative_log_likelihood([0.4, 0.6], [(0.0, 1.0), (3.0, 2.0)], far), rel=1e-12)

    def test_clips_finite_log_densities_below_the_floor(self, optimizer: Optimizer) -> None:
        component = normal(0.0, 1.0)
        mixture = MixtureModel([component])
        data = np.array([200.0])
        log_density = cc.lpdf(component, data)
        assert np.all(np.isfinite(log_density))
        assert np.all(log_density < LOG_FLOOR)

        observed_data_likelihood_strategy(
            component, make_state(data, None, mixture), odl_block(component_id=0), optimizer
        )

        assert captured_target(optimizer)([0.0, 1.0]) == -LOG_FLOOR

    @pytest.mark.filterwarnings("error")
    def test_penalizes_points_outside_every_support(self, optimizer: Optimizer) -> None:
        exponential = family(FamilyName.EXPONENTIAL)(lambda_=1.0)
        uniform = family(FamilyName.CONTINUOUS_UNIFORM)(lower_bound=0.0, upper_bound=1.0)
        mixture = MixtureModel([exponential, uniform])
        state = make_state(np.array([-1.0, 0.5]), None, mixture)

        observed_data_likelihood_strategy(uniform, state, odl_block(params={"upper_bound"}), optimizer)
        target = captured_target(optimizer)

        observed_data_likelihood_strategy(
            uniform, make_state(np.array([0.5]), None, mixture), odl_block(params={"upper_bound"}), optimizer
        )

        assert target([2.0]) - captured_target(optimizer)([2.0]) == pytest.approx(-LOG_FLOOR)

    def test_is_finite_with_unsupported_outliers(self, optimizer: Optimizer) -> None:
        mixture = MixtureModel(
            [family(FamilyName.EXPONENTIAL)(lambda_=0.5), family(FamilyName.EXPONENTIAL)(lambda_=3.0)]
        )
        data = np.array([-0.5, 0.2, 1.0, 2.0])

        observed_data_likelihood_strategy(mixture[1], make_state(data, None, mixture), odl_block(), optimizer)

        assert np.isfinite(captured_target(optimizer)([3.0]))

    @pytest.mark.parametrize("rate", [1.0, 3.0], ids=["rate-1", "rate-3"])
    def test_unsupported_point_adds_a_constant_cost(self, optimizer: Optimizer, rate: float) -> None:
        mixture = MixtureModel(
            [family(FamilyName.EXPONENTIAL)(lambda_=0.5), family(FamilyName.EXPONENTIAL)(lambda_=3.0)]
        )
        observed_data_likelihood_strategy(
            mixture[1], make_state(np.array([-0.5, 0.2, 1.0, 2.0]), None, mixture), odl_block(), optimizer
        )
        target_with_outlier = captured_target(optimizer)
        observed_data_likelihood_strategy(
            mixture[1], make_state(np.array([0.2, 1.0, 2.0]), None, mixture), odl_block(), optimizer
        )

        assert target_with_outlier([rate]) - captured_target(optimizer)([rate]) == pytest.approx(-LOG_FLOOR)


class TestParameterSelection:
    """Check parameterselection behavior."""

    def test_only_requested_values_are_optimized(self, mixture: MixtureModel, optimizer: Optimizer) -> None:
        _, new_params = observed_data_likelihood_strategy(
            mixture[1], make_state(X, None, mixture), odl_block(params={"sigma"}), optimizer
        )

        assert new_params == {"sigma": 2.0}
        assert optimizer.minimize.call_args.args[1] == [2.0]  # type: ignore[attr-defined]
        assert captured_target(optimizer)([1.5]) == pytest.approx(
            negative_log_likelihood([0.4, 0.6], [(0.0, 1.0), (3.0, 1.5)]), rel=1e-12
        )

    def test_fixed_values_of_the_component_are_kept(self, optimizer: Optimizer) -> None:
        fixed_mu = family(FamilyName.NORMAL).view(mu=3.0)(sigma=2.0)
        mixture = MixtureModel([normal(0.0, 1.0), fixed_mu], weights=[0.4, 0.6])

        _, new_params = observed_data_likelihood_strategy(
            fixed_mu, make_state(X, None, mixture), odl_block(), optimizer
        )

        assert new_params == {"sigma": 2.0}
        assert captured_target(optimizer)([1.5]) == pytest.approx(
            negative_log_likelihood([0.4, 0.6], [(0.0, 1.0), (3.0, 1.5)]), rel=1e-12
        )

    def test_nothing_to_optimize_skips_the_optimizer(self, mixture: MixtureModel, optimizer: Optimizer) -> None:
        result = observed_data_likelihood_strategy(
            mixture[1], make_state(X, None, mixture), odl_block(params=set()), optimizer
        )

        assert result == (1, {})
        optimizer.minimize.assert_not_called()  # type: ignore[attr-defined]

    def test_unknown_names_are_rejected(self, mixture: MixtureModel, optimizer: Optimizer) -> None:
        with pytest.raises(ValueError, match=r"\['var'\] are not parameters of the 'meanStd' parametrization"):
            observed_data_likelihood_strategy(
                mixture[1], make_state(X, None, mixture), odl_block(params={"var"}), optimizer
            )

        optimizer.minimize.assert_not_called()  # type: ignore[attr-defined]

    def test_fixed_values_cannot_be_requested(self, optimizer: Optimizer) -> None:
        fixed_mu = family(FamilyName.NORMAL).view(mu=3.0)(sigma=2.0)
        mixture = MixtureModel([normal(0.0, 1.0), fixed_mu])

        with pytest.raises(ValueError, match=r"\['mu'\] are fixed and cannot be optimized"):
            observed_data_likelihood_strategy(
                fixed_mu, make_state(X, None, mixture), odl_block(params={"mu", "sigma"}), optimizer
            )


class TestStrategyContract:
    """Check strategycontract behavior."""

    def test_does_not_modify_the_component_or_the_mixture(self, mixture: MixtureModel, optimizer: Optimizer) -> None:
        components_before = mixture.components
        log_weights_before = mixture.log_weights.copy()
        component = mixture[1]

        observed_data_likelihood_strategy(component, make_state(X, None, mixture), odl_block(), optimizer)

        assert cc.all_parameters(component) == {"mu": 3.0, "sigma": 2.0}
        assert mixture.components == components_before
        assert np.array_equal(mixture.log_weights, log_weights_before)

    def test_is_called_once_from_the_current_values(self, mixture: MixtureModel, optimizer: Optimizer) -> None:
        observed_data_likelihood_strategy(mixture[1], make_state(X, None, mixture), odl_block(), optimizer)

        optimizer.minimize.assert_called_once()  # type: ignore[attr-defined]
        assert optimizer.minimize.call_args.args[1] == [3.0, 2.0]  # type: ignore[attr-defined]

    def test_returns_the_component_id_and_floats(self, mixture: MixtureModel, optimizer: Optimizer) -> None:
        optimizer.minimize.side_effect = lambda target, start: [1.5, 2.5]  # type: ignore[attr-defined]

        component_id, new_params = observed_data_likelihood_strategy(
            mixture[1], make_state(X, None, mixture), odl_block(), optimizer
        )

        assert component_id == 1
        assert new_params == {"mu": 1.5, "sigma": 2.5}
        assert all(type(value) is float for value in new_params.values())

    def test_responsibilities_are_not_needed(self, mixture: MixtureModel, optimizer: Optimizer) -> None:
        state = make_state(X, None, mixture)

        assert state.H is None
        assert observed_data_likelihood_strategy(mixture[1], state, odl_block(), optimizer)[0] == 1
