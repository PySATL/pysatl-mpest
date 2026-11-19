"""Tests for StepBreakpointer."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import pytest
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators.iterative import PipelineState, StepBreakpointer

from tests.helpers.families import family


@pytest.fixture
def dummy_state() -> PipelineState:
    dummy_mixture = MixtureModel([family(FamilyName.EXPONENTIAL)(lambda_=1.0)])
    return PipelineState(np.array([]), None, None, dummy_mixture, None)


@pytest.mark.parametrize("max_steps", [1, 5, 100], ids=["one-step", "five-steps", "one-hundred-steps"])
def test_initialization_sets_valid_max_steps_and_zero_counter(max_steps: int):
    breakpointer = StepBreakpointer(max_steps)
    assert breakpointer.max_steps == max_steps
    assert breakpointer._current_step == 0


def test_initialization_rejects_zero_max_steps_with_expected_message():
    expected_message = "The maximum number of steps must be greater than or equal to 1"
    with pytest.raises(ValueError, match=expected_message):
        StepBreakpointer(0)


@pytest.mark.parametrize("max_steps", [1, 3, 10], ids=["one-step", "three-steps", "ten-steps"])
def test_check_stops_at_configured_step(max_steps: int, dummy_state):
    breakpointer = StepBreakpointer(max_steps)

    for i in range(1, max_steps):
        should_stop = breakpointer.check(dummy_state)
        assert not should_stop, f"Should not stop at step {i} for max_steps={max_steps}"
        assert breakpointer._current_step == i, "Internal counter is incorrect"

    should_stop_final = breakpointer.check(dummy_state)
    assert should_stop_final, f"Should stop at the final step {max_steps}"


def test_counter_is_not_reset_by_check_after_stopping(dummy_state):
    """check keeps counting past the limit; only reset() zeroes the counter."""
    max_steps = 5
    breakpointer = StepBreakpointer(max_steps)

    for _ in range(max_steps):
        breakpointer.check(dummy_state)

    assert breakpointer._current_step == max_steps
    assert breakpointer.check(dummy_state)
    assert breakpointer._current_step == max_steps + 1


def test_breakpointer_is_reusable_after_reset(dummy_state):
    max_steps = 3
    breakpointer = StepBreakpointer(max_steps)

    assert not breakpointer.check(dummy_state)
    assert not breakpointer.check(dummy_state)
    assert breakpointer.check(dummy_state)
    assert breakpointer._current_step == max_steps

    breakpointer.reset()
    assert breakpointer._current_step == 0

    assert not breakpointer.check(dummy_state)
    assert not breakpointer.check(dummy_state)
    assert breakpointer.check(dummy_state)
    assert breakpointer._current_step == max_steps


def test_reset_restores_full_iteration_budget(dummy_state):
    """After reset() the breakpointer needs max_steps fresh checks to stop again."""
    max_steps = 3
    breakpointer = StepBreakpointer(max_steps)

    breakpointer.check(dummy_state)
    breakpointer.check(dummy_state)
    breakpointer.reset()

    for i in range(1, max_steps):
        assert not breakpointer.check(dummy_state), f"Should not stop at step {i} after reset"

    assert breakpointer.check(dummy_state)
