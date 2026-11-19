"""
Unit tests for numerical Q-function optimization with an optimizer double.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.core import _components as cc
from pysatl_mpest.estimators.iterative import PipelineState
from pysatl_mpest.estimators.iterative._strategies import q_function_strategy
from pysatl_mpest.optimizers import Optimizer
from pytest_mock import MockerFixture
from scipy import stats

from tests.helpers.families import family
from tests.helpers.pipeline import make_state, q_block


@pytest.fixture
def component() -> ParametricFamilyDistribution:
    """A normal component in the ``meanVar`` parametrization, which has no analytical strategy."""
    return family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=1.0, var=2.0)


@pytest.fixture
def state() -> PipelineState:
    """Provide a sample with two components matching the columns of H."""
    return make_state(
        X=np.array([0.5, 1.0, 3.0]),
        H=np.array([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4]], dtype=np.float64),
        mixture=MixtureModel([family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=1.0, var=2.0)] * 2),
    )


@pytest.fixture
def optimizer(mocker: MockerFixture) -> Optimizer:
    """An optimizer that returns the values 3.0 and 4.0 (or just 3.0 for a single parameter)."""
    mock = mocker.create_autospec(Optimizer, instance=True)
    mock.minimize.side_effect = lambda target, start: [3.0, 4.0][: len(start)]
    return mock


def expected_q(state: PipelineState, mu: float, var: float) -> float:
    """Compute the negative Q-function of the first component with SciPy.

    Parameters
    ----------
    state : PipelineState
        Sample and responsibilities.
    mu : float
        Candidate mean.
    var : float
        Candidate variance.

    Returns
    -------
    float
        Negative weighted log-density.
    """
    assert state.H is not None
    return -float(np.dot(state.H[:, 0], stats.norm(loc=mu, scale=np.sqrt(var)).logpdf(state.X)))


class TestOptimizerDelegation:
    """Starting optimization and mapping its result."""

    def test_does_not_modify_the_original_component(
        self, component: ParametricFamilyDistribution, state: PipelineState, optimizer: Optimizer
    ) -> None:
        """Does not modify the original component."""
        q_function_strategy(component, state, q_block(), optimizer)

        assert cc.all_parameters(component) == {"mu": 1.0, "var": 2.0}

    def test_calls_the_optimizer_once_from_the_current_values(
        self, component: ParametricFamilyDistribution, state: PipelineState, optimizer: Optimizer
    ) -> None:
        """Calls the optimizer once from the current values."""
        q_function_strategy(component, state, q_block(), optimizer)

        optimizer.minimize.assert_called_once()  # type: ignore[attr-defined]
        _, start = optimizer.minimize.call_args.args  # type: ignore[attr-defined]
        assert start == [1.0, 2.0]

    def test_returns_the_optimizer_result_by_parameter_name(
        self, component: ParametricFamilyDistribution, state: PipelineState, optimizer: Optimizer
    ) -> None:
        """Returns the optimizer result by parameter name."""
        component_id, new_params = q_function_strategy(component, state, q_block(), optimizer)

        assert component_id == 0
        assert new_params == {"mu": 3.0, "var": 4.0}
        assert all(type(value) is float for value in new_params.values())


class TestObjective:
    """The negative Q-function passed to the optimizer."""

    def test_is_the_negative_q_function(
        self, component: ParametricFamilyDistribution, state: PipelineState, optimizer: Optimizer
    ) -> None:
        """Evaluate the negative weighted log-density."""
        q_function_strategy(component, state, q_block(), optimizer)
        target, _ = optimizer.minimize.call_args.args  # type: ignore[attr-defined]

        assert target([0.5, 3.0]) == pytest.approx(expected_q(state, mu=0.5, var=3.0), rel=1e-12)

    def test_is_infinite_where_a_constraint_is_violated(
        self, component: ParametricFamilyDistribution, state: PipelineState, optimizer: Optimizer
    ) -> None:
        """Is infinite where a constraint is violated."""
        q_function_strategy(component, state, q_block(), optimizer)
        target, _ = optimizer.minimize.call_args.args  # type: ignore[attr-defined]

        assert target([0.5, -1.0]) == np.inf
        assert target([0.5, 0.0]) == np.inf

    def test_points_with_zero_responsibility_do_not_matter(self, optimizer: Optimizer) -> None:
        """Points with zero responsibility do not matter."""
        uniform = family(FamilyName.CONTINUOUS_UNIFORM)(lower_bound=0.0, upper_bound=1.0)
        state = make_state(
            X=np.array([0.5, 5.0]), H=np.array([[1.0, 0.0], [0.0, 1.0]]), mixture=MixtureModel([uniform] * 2)
        )

        q_function_strategy(uniform, state, q_block(params={"upper_bound"}), optimizer)
        target, _ = optimizer.minimize.call_args.args  # type: ignore[attr-defined]

        assert target([2.0]) == pytest.approx(np.log(2.0))
        assert target([0.4]) == np.inf  # the point 0.5 leaves the support


class TestParameterSelection:
    """Requested, fixed, empty and unknown parameters."""

    def test_only_the_requested_parameters_are_optimized(
        self, component: ParametricFamilyDistribution, state: PipelineState, optimizer: Optimizer
    ) -> None:
        """Only the requested parameters are optimized."""
        _, new_params = q_function_strategy(component, state, q_block(params={"var"}), optimizer)
        target, start = optimizer.minimize.call_args.args  # type: ignore[attr-defined]

        assert new_params == {"var": 3.0}
        assert start == [2.0]
        assert target([3.0]) == pytest.approx(expected_q(state, mu=1.0, var=3.0), rel=1e-12)  # mu keeps its value

    def test_fixed_parameters_are_kept(self, state: PipelineState, optimizer: Optimizer) -> None:
        """Fixed parameters are kept."""
        fixed_mu = family(FamilyName.NORMAL).view(parametrization_name="meanVar", mu=1.0)(var=2.0)

        _, new_params = q_function_strategy(fixed_mu, state, q_block(), optimizer)
        target, start = optimizer.minimize.call_args.args  # type: ignore[attr-defined]

        assert new_params == {"var": 3.0}
        assert start == [2.0]
        assert target([3.0]) == pytest.approx(expected_q(state, mu=1.0, var=3.0), rel=1e-12)

    def test_nothing_to_optimize_skips_the_optimizer(
        self, component: ParametricFamilyDistribution, state: PipelineState, optimizer: Optimizer
    ) -> None:
        """Nothing to optimize skips the optimizer."""
        assert q_function_strategy(component, state, q_block(params=set()), optimizer) == (0, {})

        optimizer.minimize.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.parametrize("params", [{"sigma"}, {"mu", "sigm"}], ids=["unknown-sigma", "known-and-unknown"])
    def test_unknown_parameters_are_rejected(
        self, component: ParametricFamilyDistribution, state: PipelineState, optimizer: Optimizer, params: set[str]
    ) -> None:
        """Unknown parameters are rejected."""
        with pytest.raises(ValueError, match="are not parameters of the 'meanVar' parametrization"):
            q_function_strategy(component, state, q_block(params=params), optimizer)

        optimizer.minimize.assert_not_called()  # type: ignore[attr-defined]

    def test_fixed_parameters_cannot_be_requested(self, state: PipelineState, optimizer: Optimizer) -> None:
        """Fixed parameters cannot be requested."""
        fixed_mu = family(FamilyName.NORMAL).view(parametrization_name="meanVar", mu=1.0)(var=2.0)

        with pytest.raises(ValueError, match=r"\['mu'\] are fixed and cannot be optimized"):
            q_function_strategy(fixed_mu, state, q_block(params={"mu", "var"}), optimizer)


class TestResponsibilities:
    """Missing responsibilities are rejected."""

    def test_raises_if_h_is_not_computed(self, component: ParametricFamilyDistribution, optimizer: Optimizer) -> None:
        """Reject missing responsibilities before optimization."""
        state = make_state(X=np.array([1.0]), H=None, mixture=MixtureModel([component] * 2))

        with pytest.raises(ValueError, match=r"^Responsibility matrix H is not computed\.$"):
            q_function_strategy(component, state, q_block(), optimizer)
