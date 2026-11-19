"""Tests for ExpectationStep."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators.iterative import ExpectationStep, MaximizationStep, PipelineState, WeightsStep


@pytest.fixture
def initial_pipeline_state(mocker) -> PipelineState:
    mixture = mocker.create_autospec(MixtureModel, instance=True)
    mixture.component_lpdf_matrix.return_value = np.log([[0.6, 0.1], [0.8, 0.3]]).astype(np.float64)
    mixture.configure_mock(log_weights=np.log([0.7, 0.3]).astype(np.float64))

    return PipelineState(
        X=np.array([1.0, 2.0], dtype=np.float64),
        H=None,
        prev_mixture=None,
        curr_mixture=mixture,
        error=None,
    )


class TestConstruction:
    def test_defaults_to_soft_and_accepts_hard_assignment(self):
        default_step = ExpectationStep()
        assert default_step.is_soft is True

        soft_step = ExpectationStep(is_soft=True)
        assert soft_step.is_soft is True

        hard_step = ExpectationStep(is_soft=False)
        assert hard_step.is_soft is False


class TestNextSteps:
    def test_allows_weights_and_maximization_as_next_steps(self):
        step = ExpectationStep()
        assert step.available_next_steps == [WeightsStep, MaximizationStep]


class TestRun:
    def test_soft_assignment_computes_responsibilities(self, initial_pipeline_state):
        state = initial_pipeline_state
        step = ExpectationStep(is_soft=True)

        expected_h = np.array([[0.42 / 0.45, 0.03 / 0.45], [0.56 / 0.65, 0.09 / 0.65]], dtype=np.float64)

        result_state = step.run(state)

        assert result_state.H is not None
        assert_allclose(
            result_state.H, expected_h, rtol=1e-6, atol=1e-6, err_msg="Soft H matrix calculation is incorrect"
        )

    def test_hard_assignment_computes_responsibilities(self, initial_pipeline_state):
        state = initial_pipeline_state
        step = ExpectationStep(is_soft=False)

        expected_h_hard = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)

        result_state = step.run(state)

        assert result_state.H is not None
        assert_array_equal(
            result_state.H,
            expected_h_hard,
            err_msg="Hard H matrix calculation is incorrect",
        )

    def test_hard_assignment_leaves_zeros_for_unsupported_points(self, initial_pipeline_state) -> None:
        state = initial_pipeline_state
        state.X = np.array([1.0, -1.0, 2.0])
        state.curr_mixture.component_lpdf_matrix.return_value = np.array(
            [[np.log(0.6), np.log(0.1)], [-np.inf, -np.inf], [np.log(0.1), np.log(0.8)]]
        )

        result = ExpectationStep(is_soft=False).run(state)

        assert result.H is not None
        assert_array_equal(result.H, [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])

    def test_uses_component_log_densities_of_the_data(self, initial_pipeline_state):
        state = initial_pipeline_state

        ExpectationStep().run(state)

        state.curr_mixture.component_lpdf_matrix.assert_called_once_with(state.X)

    def test_returns_state_and_modifies_only_h(self, initial_pipeline_state):
        state = initial_pipeline_state
        step = ExpectationStep(is_soft=False)

        original_x_ref = state.X
        original_mixture_ref = state.curr_mixture
        original_prev_mixture_ref = state.prev_mixture
        original_error_ref = state.error

        result_state = step.run(state)

        assert isinstance(result_state, PipelineState)
        assert result_state is state
        assert state.H is not None
        assert isinstance(result_state.H, np.ndarray)

        assert result_state.X is original_x_ref
        assert result_state.curr_mixture is original_mixture_ref
        assert result_state.prev_mixture is original_prev_mixture_ref
        assert result_state.error is original_error_ref


class TestReset:
    def test_reset_does_not_change_the_next_run(self, initial_pipeline_state):
        state = initial_pipeline_state
        step = ExpectationStep(is_soft=True)

        step.reset()
        result_state = step.run(state)

        expected_h = np.array([[0.42 / 0.45, 0.03 / 0.45], [0.56 / 0.65, 0.09 / 0.65]], dtype=np.float64)
        assert result_state.H is not None
        assert_allclose(result_state.H, expected_h, rtol=1e-6, atol=1e-6)
