"""Tests for WeightsStep."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from numpy.testing import assert_allclose
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators.iterative import (
    MaximizationStep,
    PipelineState,
    PipelineStep,
    WeightsStep,
)

from tests.helpers.families import family
from tests.helpers.pipeline import make_state


@pytest.fixture
def normal_mixture() -> MixtureModel:
    normal = family(FamilyName.NORMAL)
    return MixtureModel([normal(mu=0.0, sigma=1.0), normal(mu=5.0, sigma=2.0)], weights=[0.3, 0.7])


@pytest.fixture
def state_with_H(normal_mixture: MixtureModel) -> PipelineState:
    X = np.array([1.0, 2.0, 3.0, 4.0])
    # Responsibility sums per component: [1.8, 2.2] out of 4 observations.
    H = np.array([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.2, 0.8]])
    return make_state(X, H, normal_mixture)


class TestConstruction:
    def test_is_a_pipeline_step_without_parameters(self):
        step = WeightsStep()

        assert isinstance(step, PipelineStep)


class TestNextSteps:
    def test_allows_maximization_as_the_next_step(self):
        step = WeightsStep()

        assert step.available_next_steps == [MaximizationStep]


class TestRun:
    def test_updates_weights_from_responsibilities(self, state_with_H: PipelineState, normal_mixture: MixtureModel):
        components_before = list(normal_mixture)
        log_weights_before = normal_mixture.log_weights.copy()

        result = WeightsStep().run(state_with_H)

        assert result is state_with_H
        assert state_with_H.curr_mixture is not normal_mixture
        assert_allclose(state_with_H.curr_mixture.weights, [0.45, 0.55], atol=1e-12)
        assert all(new is old for new, old in zip(state_with_H.curr_mixture, components_before, strict=True))
        # The mixture passed in the state is left unchanged.
        assert_allclose(normal_mixture.weights, [0.3, 0.7], atol=1e-12)
        np.testing.assert_array_equal(normal_mixture.log_weights, log_weights_before)

    def test_zero_responsibility_keeps_log_weight_finite(self, normal_mixture: MixtureModel):
        state = make_state(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([[1.0, 0.0]] * 4),
            normal_mixture,
        )

        WeightsStep().run(state)

        assert state.curr_mixture is not normal_mixture
        assert np.all(np.isfinite(state.curr_mixture.log_weights))
        assert_allclose(state.curr_mixture.weights, [1.0, 0.0], atol=1e-12)
        assert_allclose(normal_mixture.weights, [0.3, 0.7], atol=1e-12)

    def test_missing_responsibilities_set_error_and_keep_weights(
        self, normal_mixture: MixtureModel, state_with_H: PipelineState
    ):
        state = state_with_H
        state.H = None
        original_log_weights = normal_mixture.log_weights.copy()

        result = WeightsStep().run(state)

        assert result is state
        assert isinstance(state.error, ValueError)
        assert str(state.error) == "Responsibility matrix H is not computed."
        assert state.curr_mixture is normal_mixture
        np.testing.assert_array_equal(normal_mixture.log_weights, original_log_weights)

    def test_returns_same_state_and_preserves_other_fields(self, state_with_H: PipelineState):
        state = state_with_H
        original_X = state.X
        original_H = state.H
        original_prev_mixture = state.prev_mixture
        original_error = state.error
        mixture_before = state.curr_mixture

        result = WeightsStep().run(state)

        assert result is state
        assert result.X is original_X
        assert result.H is original_H
        assert result.prev_mixture is original_prev_mixture
        assert result.error is original_error
        assert result.curr_mixture is not mixture_before


class TestClearAfterPrune:
    def test_does_nothing(self, state_with_H: PipelineState):
        step = WeightsStep()

        step.clear_after_prune([0])

        assert_allclose(state_with_H.curr_mixture.weights, [0.3, 0.7], atol=1e-12)


class TestReset:
    def test_reset_does_not_change_the_next_run(self, state_with_H: PipelineState):
        step = WeightsStep()

        step.reset()
        result = step.run(state_with_H)

        assert_allclose(result.curr_mixture.weights, [0.45, 0.55], atol=1e-12)
