"""Tests for Pipeline class"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Callable

import numpy as np
import pytest
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.core import _components as cc
from pysatl_mpest.estimators.iterative import (
    Breakpointer,
    ExpectationStep,
    MaximizationStep,
    Pipeline,
    PipelineState,
    PipelineStep,
    Pruner,
    WeightsStep,
)
from pysatl_mpest.estimators.iterative._iteration_history import IterationRecord, IterationsHistory
from pysatl_mpest.optimizers import Optimizer
from pytest_mock import MockerFixture

from tests.helpers.families import family

# --- Mock objects for isolated testing ---


class MockStepA(PipelineStep):
    """A concrete mock step 'A' for testing sequences."""

    def __init__(self, call_log: list[str] | None = None):
        self.call_log = call_log if call_log is not None else []

    @property
    def available_next_steps(self) -> list[type[PipelineStep]]:
        return [MockStepB]

    def run(self, state: PipelineState) -> PipelineState:
        self.call_log.append("A")
        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        pass

    def reset(self) -> None:
        pass


class MockStepB(PipelineStep):
    """A concrete mock step 'B' that can follow 'A' and loop back."""

    def __init__(self, call_log: list[str] | None = None):
        self.call_log = call_log if call_log is not None else []

    @property
    def available_next_steps(self) -> list[type[PipelineStep]]:
        return [MockStepA]

    def run(self, state: PipelineState) -> PipelineState:
        self.call_log.append("B")
        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        pass

    def reset(self) -> None:
        pass


class MockSingleStep(PipelineStep):
    """A concrete mock step that can only be followed by itself."""

    def __init__(self, call_log: list[str] | None = None):
        self.call_log = call_log if call_log is not None else []

    @property
    def available_next_steps(self) -> list[type[PipelineStep]]:
        return [MockSingleStep]

    def run(self, state: PipelineState) -> PipelineState:
        if self.call_log is not None:
            self.call_log.append("SingleStep")
        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        pass

    def reset(self) -> None:
        pass


class MockStepWithError(MockSingleStep):
    """A mock step that simulates an error occurring."""

    def run(self, state: PipelineState) -> PipelineState:
        super().run(state)
        state.error = ValueError(f"Simulated error in {self.__class__.__name__}")
        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        pass


class ModifyingStep(PipelineStep):
    """A concrete mock step that actively modifies the mixture's state."""

    @property
    def available_next_steps(self) -> list[type[PipelineStep]]:
        return [ModifyingStep]

    def run(self, state: PipelineState) -> PipelineState:
        mixture = state.curr_mixture

        if mixture.n_components > 0:
            new_weights = np.zeros(mixture.n_components)
            new_weights[0] = 1.0
            mixture = mixture.with_log_weights(np.log(new_weights + 1e-30))

        if mixture.n_components > 0:
            comp = mixture.components[0]
            new_comp = cc.with_parameters(comp, lambda_=999.0)
            mixture = mixture.with_component(0, new_comp)

        if mixture.n_components > 1:
            mixture = mixture.without_component(1)

        state.curr_mixture = mixture
        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        pass

    def reset(self) -> None:
        pass


class ParameterIncrementStep(PipelineStep):
    """A mock step that predictably modifies mixture parameters on each run."""

    @property
    def available_next_steps(self) -> list[type[PipelineStep]]:
        return [ParameterIncrementStep]

    def run(self, state: PipelineState) -> PipelineState:
        mixture = state.curr_mixture
        components = [
            cc.with_parameters(component, lambda_=cc.all_parameters(component)["lambda_"] + 1.0)
            for component in mixture.components
        ]
        mixture = mixture.with_components(components)

        new_weights = mixture.weights.copy()
        if len(new_weights) > 1:
            new_weights[0] += 0.1
            new_weights[1] -= 0.1

            new_weights = np.clip(new_weights, 0, 1)
            new_weights /= np.sum(new_weights)

        state.curr_mixture = mixture.with_log_weights(np.log(new_weights + 1e-30))
        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        pass

    def reset(self) -> None:
        pass


class StopOnCheck(Breakpointer):
    """Stop on the specified one-based check call; reset by Pipeline.fit before every fit."""

    def __init__(self, call_number: int, call_log: list[str] | None = None):
        self.call_number = call_number
        self.calls = 0
        self.call_log = call_log

    def reset(self) -> None:
        self.calls = 0

    def check(self, state: PipelineState) -> bool:
        if self.call_log is not None:
            self.call_log.append("check_breakpointer")
        self.calls += 1
        return self.calls >= self.call_number


class RecordingBreakpointer(Breakpointer):
    """Log 'reset' and 'check' calls; never stops the pipeline on its own."""

    def __init__(self, log: list[str]):
        self.log = log

    def reset(self) -> None:
        self.log.append("reset")

    def check(self, state: PipelineState) -> bool:
        self.log.append("check")
        return False


class RemovingFirstComponentPruner(Pruner):
    """A mock object for Pruner that removes a component and logs its calls."""

    def __init__(self, name: str = "pruner", call_log: list[str] | None = None):
        self.name = name
        self.call_log = call_log if call_log is not None else []

    def prune(self, state: PipelineState) -> tuple[PipelineState, list[int]]:
        self.call_log.append(self.name)
        removed_components_indices = []
        if state.curr_mixture.n_components > 1:
            state.curr_mixture = state.curr_mixture.without_component(0)
            removed_components_indices.append(0)
        return (state, removed_components_indices)


# --- Fixtures for tests ---


@pytest.fixture
def initial_mixture() -> MixtureModel:
    """Provides a basic MixtureModel with two components."""
    components = [family(FamilyName.EXPONENTIAL)(lambda_=1.0), family(FamilyName.EXPONENTIAL)(lambda_=2.0)]
    return MixtureModel(components, weights=[0.5, 0.5])


@pytest.fixture
def sample_data() -> np.ndarray:
    """Provides a simple data array."""
    return np.array([1, 2, 6, 7], dtype=np.float64)


# --- Initialization (`__init__`) Tests ---


class TestConstruction:
    @pytest.mark.parametrize("container", [list, tuple], ids=["list", "tuple"])
    def test_accepts_sequence_containers(self, container: Callable):
        steps = container([MockSingleStep()])
        breakpointers = container([StopOnCheck(call_number=1)])
        pruners = container([RemovingFirstComponentPruner()])

        pipeline = Pipeline(steps, breakpointers, pruners)

        assert isinstance(pipeline.steps, list)
        assert isinstance(pipeline.breakpointers, list)
        assert isinstance(pipeline.pruners, list)
        assert isinstance(pipeline.history, IterationsHistory)

    def test_uses_no_pruners_by_default(self):
        steps = [MockSingleStep()]
        breakpointers = [StopOnCheck(call_number=1)]

        pipeline = Pipeline(steps, breakpointers, pruners=None)

        assert pipeline.pruners == []

    def test_requires_a_stopping_criterion(self):
        steps = [MockSingleStep()]

        with pytest.raises(
            ValueError,
            match=(
                r"^The 'breakpointers' list cannot be empty\. "
                r"At least one stopping criterion must be provided to prevent an infinite loop\.$"
            ),
        ):
            Pipeline(steps, [])

    def test_sets_history_frequency(self):
        steps = [MockSingleStep()]
        breakpointers = [StopOnCheck(call_number=1)]

        pipeline_default = Pipeline(steps, breakpointers)
        assert pipeline_default.history.once_in_iterations == 1

        pipeline_custom = Pipeline(steps, breakpointers, once_in_iterations=5)
        FREQUENCY_OF_LOGS = 5
        assert pipeline_custom.history.once_in_iterations == FREQUENCY_OF_LOGS


# --- Step Validation Tests ---


class TestStepValidation:
    def test_requires_at_least_one_step(self):
        breakpointers = [StopOnCheck(call_number=1)]
        with pytest.raises(ValueError, match=r"^The 'steps' list cannot be empty for a Pipeline\.$"):
            Pipeline([], breakpointers)

    def test_rejects_invalid_step_order(self):
        invalid_steps = [MockStepA(), MockStepA()]
        breakpointers = [StopOnCheck(call_number=1)]

        with pytest.raises(ValueError, match="Wrong pipeline configuration"):
            Pipeline(invalid_steps, breakpointers)

    def test_accepts_closed_step_cycle(self):
        valid_steps = [MockStepA(), MockStepB()]
        breakpointers = [StopOnCheck(call_number=1)]

        try:
            Pipeline(valid_steps, breakpointers)
        except ValueError:
            pytest.fail("Valid cyclic step configuration raised ValueError unexpectedly.")

    def test_the_last_pair_of_steps_is_checked(self):
        with pytest.raises(ValueError, match="Wrong pipeline configuration"):
            Pipeline([MockStepA(), MockStepB(), MockStepB()], [StopOnCheck(call_number=1)])

    def test_the_first_pair_of_steps_is_checked(self):
        with pytest.raises(ValueError, match="Wrong pipeline configuration"):
            Pipeline([MockStepA(), MockStepA(), MockStepB()], [StopOnCheck(call_number=1)])

    def test_the_pair_closing_the_cycle_is_checked(self):
        with pytest.raises(ValueError, match="Wrong pipeline configuration"):
            Pipeline([MockStepA(), MockStepB(), MockStepA()], [StopOnCheck(call_number=1)])

    def test_accepts_expectation_weights_maximization_sequence(self, mocker: MockerFixture):
        optimizer = mocker.MagicMock(spec=Optimizer)
        steps = [ExpectationStep(), WeightsStep(), MaximizationStep([], optimizer)]

        Pipeline(steps, [StopOnCheck(call_number=1)])

    def test_accepts_expectation_maximization_sequence(self, mocker: MockerFixture):
        optimizer = mocker.MagicMock(spec=Optimizer)
        steps = [ExpectationStep(), MaximizationStep([], optimizer)]

        Pipeline(steps, [StopOnCheck(call_number=1)])

    def test_rejects_maximization_weights_transition(self, mocker: MockerFixture):
        optimizer = mocker.MagicMock(spec=Optimizer)
        steps = [ExpectationStep(), MaximizationStep([], optimizer), WeightsStep()]

        with pytest.raises(ValueError, match="Wrong pipeline configuration"):
            Pipeline(steps, [StopOnCheck(call_number=1)])


# --- Fit Method Tests ---

NUMBER_OF_BLOCKS = 2


class TestFit:
    def test_runs_each_iteration_and_returns_updated_mixture(self, initial_mixture, sample_data):
        expected_n_components = 2

        steps = [ParameterIncrementStep()]
        breakpointers = [StopOnCheck(call_number=2)]
        pipeline = Pipeline(steps, breakpointers)

        # Expected state after 2 iterations with ParameterIncrementStep
        expected_final_lambdas = np.array([3.0, 4.0], dtype=np.float64)
        expected_final_weights = np.array([0.7, 0.3], dtype=np.float64)

        fitted_mixture = pipeline.fit(sample_data, initial_mixture)

        assert fitted_mixture.n_components == expected_n_components

        rtol = 1e-7

        np.testing.assert_allclose(
            fitted_mixture.weights,
            expected_final_weights,
            rtol=rtol,
            err_msg="Mixture weights did not reach the expected values.",
        )

        final_lambdas = np.array(
            [cc.all_parameters(comp)["lambda_"] for comp in fitted_mixture.components], dtype=np.float64
        )

        np.testing.assert_allclose(final_lambdas, expected_final_lambdas, rtol=rtol)

    def test_preserves_the_input_mixture(self, initial_mixture, sample_data):
        weights_before = initial_mixture.weights.copy()
        log_weights_before = initial_mixture.log_weights.copy()
        components_before = initial_mixture.components
        n_components_before = initial_mixture.n_components

        steps = [ModifyingStep()]
        breakpointers = [StopOnCheck(call_number=1)]
        pipeline = Pipeline(steps, breakpointers)

        pipeline.fit(sample_data, initial_mixture)

        np.testing.assert_array_equal(
            initial_mixture.weights, weights_before, err_msg="Original mixture's weights were modified."
        )
        np.testing.assert_array_equal(
            initial_mixture.log_weights, log_weights_before, err_msg="Original mixture's log-weights were modified."
        )
        assert initial_mixture.n_components == n_components_before
        assert initial_mixture.components == components_before

    def test_runs_steps_in_configured_order(self, initial_mixture, sample_data):
        call_log = []

        # Configuration: A -> B -> A (cycle)
        steps = [MockStepA(call_log), MockStepB(call_log)]
        breakpointers = [StopOnCheck(call_number=2)]

        pipeline = Pipeline(steps, breakpointers)
        pipeline.fit(sample_data, initial_mixture)

        assert call_log == ["A", "B", "A", "B"]

    def test_stops_when_any_breakpointer_reaches_its_limit(self, initial_mixture, sample_data):
        expected_checks = 4

        call_log = []
        breakpointers = [
            StopOnCheck(call_number=99, call_log=call_log),
            StopOnCheck(call_number=2, call_log=call_log),
        ]

        steps = [MockSingleStep()]

        pipeline = Pipeline(steps, breakpointers)
        pipeline.fit(sample_data, initial_mixture)

        assert call_log.count("check_breakpointer") == expected_checks

    def test_warns_and_returns_after_step_error(self, initial_mixture, sample_data):
        call_log = []
        steps = [MockStepWithError(call_log)]
        breakpointers = [StopOnCheck(call_number=9)]

        pipeline = Pipeline(steps, breakpointers)

        with pytest.warns(RuntimeWarning) as record:
            result_mixture = pipeline.fit(sample_data, initial_mixture)

        assert len(record) == 1
        assert "stopped prematurely due to an error" in str(record[0].message)
        assert isinstance(result_mixture, MixtureModel)
        assert call_log == ["SingleStep"]

    def test_applies_component_pruning(self, initial_mixture, sample_data):
        expected_n_components = 2
        assert initial_mixture.n_components == expected_n_components

        call_log = []
        steps = [MockSingleStep()]
        pruners = [RemovingFirstComponentPruner("pruner", call_log)]
        breakpointers = [StopOnCheck(call_number=1)]

        pipeline = Pipeline(steps, breakpointers, pruners)
        fitted_mixture = pipeline.fit(sample_data, initial_mixture)

        assert call_log.count("pruner") == 1
        assert fitted_mixture.n_components == 1

    def test_runs_pruners_in_configured_order(self, initial_mixture, sample_data):
        call_log = []
        steps = [MockSingleStep()]
        pruners = [
            RemovingFirstComponentPruner(name="pruner1", call_log=call_log),
            RemovingFirstComponentPruner(name="pruner2", call_log=call_log),
        ]
        breakpointers = [StopOnCheck(call_number=1)]

        pipeline = Pipeline(steps, breakpointers, pruners)

        fitted_mixture = pipeline.fit(sample_data, initial_mixture)

        assert call_log == ["pruner1", "pruner2"]
        assert fitted_mixture.n_components == 1

    def test_runs_pruners_after_steps(self, initial_mixture, sample_data):
        call_log = []
        steps = [MockStepA(call_log), MockStepB(call_log)]
        pruners = [RemovingFirstComponentPruner("pruner", call_log)]
        breakpointers = [StopOnCheck(call_number=1, call_log=call_log)]

        pipeline = Pipeline(steps, breakpointers, pruners)
        pipeline.fit(sample_data, initial_mixture)

        expected_order = ["A", "B", "pruner", "check_breakpointer"]

        assert call_log == expected_order

    def test_records_iteration_history(self, initial_mixture, sample_data):
        steps = [MockSingleStep()]
        breakpointers = [StopOnCheck(call_number=2)]

        pipeline = Pipeline(steps, breakpointers)

        assert len(pipeline.history) == 0

        fitted_mixture = pipeline.fit(sample_data, initial_mixture)

        assert len(pipeline.history) > 0
        assert isinstance(pipeline.history[0], IterationRecord)
        assert pipeline.history[0].iteration == 0
        assert pipeline.history[0].mixture == fitted_mixture

    def test_records_at_configured_frequency(self, initial_mixture, sample_data):
        steps = [MockSingleStep()]

        breakpointers = [StopOnCheck(call_number=4)]

        pipeline = Pipeline(steps, breakpointers, once_in_iterations=2)

        fitted_mixture = pipeline.fit(sample_data, initial_mixture)

        EXPECTED_LEN_OF_HISTORY = 2
        EXPECTED_ITERATIONS = [0, 2]

        assert len(pipeline.history) == EXPECTED_LEN_OF_HISTORY
        assert pipeline.history[0].iteration == EXPECTED_ITERATIONS[0]
        assert pipeline.history[1].iteration == EXPECTED_ITERATIONS[1]
        assert pipeline.history[1].mixture == fitted_mixture


# --- Breakpointer reset tests ---


class TestBreakpointerReset:
    """Pipeline.fit must reset every breakpointer before the first iteration of every fit."""

    def test_reset_is_called_once_before_the_first_check_of_every_fit(self, initial_mixture, sample_data):
        log: list[str] = []
        stopper = StopOnCheck(call_number=2)
        recording = RecordingBreakpointer(log)
        pipeline = Pipeline([MockSingleStep()], [recording, stopper])

        pipeline.fit(sample_data, initial_mixture)
        pipeline.fit(sample_data, initial_mixture)

        # Per fit: one reset, then one 'check' per iteration (2 iterations until the stopper fires).
        assert log == ["reset", "check", "check", "reset", "check", "check"]

    def test_reset_is_called_on_breakpointers_after_the_one_that_stopped(self, initial_mixture, sample_data):
        """any() short-circuits check(), but reset() must still reach every breakpointer."""
        log: list[str] = []
        pipeline = Pipeline(
            [MockSingleStep()],
            [StopOnCheck(call_number=1), RecordingBreakpointer(log)],
        )

        pipeline.fit(sample_data, initial_mixture)
        pipeline.fit(sample_data, initial_mixture)

        # The first breakpointer stops on the first check, so the recording one is never polled —
        # but it must still be reset once per fit.
        assert log == ["reset", "reset"]

    def test_breakpointer_state_does_not_leak_between_fits(self, initial_mixture, sample_data):
        """A stateful breakpointer must behave identically on repeated fits."""
        pipeline = Pipeline([MockSingleStep()], [StopOnCheck(call_number=3)])

        pipeline.fit(sample_data, initial_mixture)
        first_fit_iterations = len(pipeline.history)

        pipeline.fit(sample_data, initial_mixture)
        second_fit_iterations = len(pipeline.history) - first_fit_iterations

        assert second_fit_iterations == first_fit_iterations == 3  # noqa: PLR2004

    def test_breakpointer_without_reset_cannot_be_instantiated(self):
        class IncompleteBreakpointer(Breakpointer):
            def check(self, state: PipelineState) -> bool:
                return False

        with pytest.raises(TypeError):
            IncompleteBreakpointer()


# --- Step reset tests ---


class TestStepReset:
    """Pipeline.fit must reset every step before the first iteration of every fit."""

    def test_reset_is_called_once_before_the_first_run_of_every_fit(self, initial_mixture, sample_data):
        log: list[tuple[str, ...]] = []
        step = RecordingStep("A", log)
        pipeline = Pipeline([step], [StopOnCheck(call_number=2)])

        pipeline.fit(sample_data, initial_mixture)
        pipeline.fit(sample_data, initial_mixture)

        # Per fit: one reset before the first run, then one run per iteration (2 iterations each).
        assert log == [
            ("reset", "A"),
            ("run", "A"),
            ("run", "A"),
            ("reset", "A"),
            ("run", "A"),
            ("run", "A"),
        ]

    def test_reset_is_called_on_every_step(self, initial_mixture, sample_data):
        log: list[tuple[str, ...]] = []
        pipeline = Pipeline(
            [RecordingStep("A", log), RecordingStep("B", log)],
            [StopOnCheck(call_number=1)],
        )

        pipeline.fit(sample_data, initial_mixture)

        assert log == [("reset", "A"), ("reset", "B"), ("run", "A"), ("run", "B")]

    def test_step_without_reset_cannot_be_instantiated(self):
        class IncompleteStep(PipelineStep):
            @property
            def available_next_steps(self) -> list[type[PipelineStep]]:
                return [IncompleteStep]

            def run(self, state: PipelineState) -> PipelineState:
                return state

            def clear_after_prune(self, removed_components_indices: list[int]) -> None:
                pass

        with pytest.raises(TypeError):
            IncompleteStep()


# --- Bookkeeping of pruned components ---


class RecordingStep(PipelineStep):
    """A step that logs when it runs and which removed components it is told about."""

    def __init__(self, name: str, log: list[tuple[str, ...]]):
        self.name = name
        self.log = log

    @property
    def available_next_steps(self) -> list[type[PipelineStep]]:
        return [RecordingStep]

    def run(self, state: PipelineState) -> PipelineState:
        self.log.append(("run", self.name))
        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        self.log.append(("clear", self.name, str(removed_components_indices)))

    def reset(self) -> None:
        self.log.append(("reset", self.name))


class ResponsibilityStep(RecordingStep):
    """A step that sets H to uniform responsibilities for the current components."""

    def run(self, state: PipelineState) -> PipelineState:
        n_components = state.curr_mixture.n_components
        state.H = np.full((len(state.X), n_components), 1.0 / n_components)
        return super().run(state)


class ScriptedPruner(Pruner):
    """Remove the requested components and log calls.

    Requested indices must be sorted in ascending order.
    """

    def __init__(self, name: str, log: list[tuple[str, ...]], removals: list[list[int]]):
        self.name = name
        self.log = log
        self.removals = iter(removals)

    def prune(self, state: PipelineState) -> tuple[PipelineState, list[int]]:
        removed = next(self.removals, [])
        self.log.append(("prune", self.name, str(removed)))
        for index in sorted(removed, reverse=True):
            state.curr_mixture = state.curr_mixture.without_component(index)
        return state, removed


def exponential_mixture(n_components: int) -> MixtureModel:
    components = [family(FamilyName.EXPONENTIAL)(lambda_=float(i + 1)) for i in range(n_components)]
    return MixtureModel(components)


class TestPruning:
    """Tests that the steps learn about pruned components right away and exactly once."""

    def test_steps_are_notified_right_after_pruning(self, sample_data):
        log: list[tuple[str, ...]] = []
        steps = [RecordingStep("A", log), RecordingStep("B", log)]
        pipeline = Pipeline(steps, [StopOnCheck(call_number=2)], [ScriptedPruner("p", log, [[0], []])])

        pipeline.fit(sample_data, exponential_mixture(3))

        assert log == [
            ("reset", "A"),
            ("reset", "B"),
            ("run", "A"),
            ("run", "B"),
            ("prune", "p", "[0]"),
            ("clear", "A", "[0]"),
            ("clear", "B", "[0]"),
            ("run", "A"),
            ("run", "B"),
            ("prune", "p", "[]"),
        ]

    def test_each_removal_is_reported_once(self, sample_data):
        log: list[tuple[str, ...]] = []
        pruner = ScriptedPruner("p", log, [[0], [], [1], []])
        pipeline = Pipeline([RecordingStep("A", log)], [StopOnCheck(call_number=4)], [pruner])

        pipeline.fit(sample_data, exponential_mixture(4))

        assert [entry[2] for entry in log if entry[0] == "clear"] == ["[0]", "[1]"]

    def test_steps_are_not_notified_when_nothing_is_removed(self, sample_data):
        log: list[tuple[str, ...]] = []
        pipeline = Pipeline([RecordingStep("A", log)], [StopOnCheck(call_number=3)], [ScriptedPruner("p", log, [])])

        pipeline.fit(sample_data, exponential_mixture(2))

        assert not [entry for entry in log if entry[0] == "clear"]

    def test_each_pruner_reports_indices_relative_to_the_mixture_it_was_given(self, sample_data):
        """The second pruner sees the mixture without the component the first one removed."""
        log: list[tuple[str, ...]] = []
        pruners = [ScriptedPruner("p1", log, [[0]]), ScriptedPruner("p2", log, [[1]])]
        pipeline = Pipeline([RecordingStep("A", log)], [StopOnCheck(call_number=1)], pruners)

        fitted = pipeline.fit(sample_data, exponential_mixture(3))

        assert log == [
            ("reset", "A"),
            ("run", "A"),
            ("prune", "p1", "[0]"),
            ("clear", "A", "[0]"),
            ("prune", "p2", "[1]"),
            ("clear", "A", "[1]"),
        ]
        # The components had the rates 1, 2 and 3; the first pruner removed the rate 1, the second one the rate 3.
        remaining_rate = 2.0
        assert fitted.n_components == 1
        assert cc.all_parameters(fitted[0])["lambda_"] == remaining_rate

    def test_columns_of_h_of_pruned_components_are_dropped(self, sample_data):
        log: list[tuple[str, ...]] = []
        pipeline = Pipeline(
            [ResponsibilityStep("A", log)],
            [StopOnCheck(call_number=2)],
            [ScriptedPruner("p", log, [[0, 2], []])],
        )

        pipeline.fit(sample_data, exponential_mixture(4))

        first, second = pipeline.history[0], pipeline.history[1]
        assert first.mixture.n_components == 2  # noqa: PLR2004
        assert first.H.shape == (len(sample_data), 2)
        assert second.H.shape == (len(sample_data), 2)

    # --- Details found by mutation testing ---

    def test_notifies_steps_after_component_removal(self, initial_mixture, sample_data):
        call_log = []
        LEN_CALL_LOG = 2

        class TrackingStep(MockSingleStep):
            def clear_after_prune(self, removed_components_indices: list[int]) -> None:
                call_log.append(removed_components_indices)

        steps = [TrackingStep(), TrackingStep()]
        pruners = [RemovingFirstComponentPruner("pruner")]
        breakpointers = [StopOnCheck(call_number=1)]

        pipeline = Pipeline(steps, breakpointers, pruners)
        pipeline.fit(sample_data, initial_mixture)

        assert len(call_log) == LEN_CALL_LOG
        for removed_indices in call_log:
            assert removed_indices == [0]


class DataProbeStep(MockSingleStep):
    """A step that records the data it receives."""

    def __init__(self):
        super().__init__([])
        self.dtypes: list[np.dtype] = []

    def run(self, state: PipelineState) -> PipelineState:
        self.dtypes.append(state.X.dtype)
        return state


class PreviousMixtureProbeStep(MockSingleStep):
    """A step that records the previous mixture next to a copy of the current one, then changes the current one."""

    def __init__(self):
        super().__init__([])
        self.seen: list[tuple[MixtureModel | None, MixtureModel]] = []

    def run(self, state: PipelineState) -> PipelineState:
        self.seen.append((state.prev_mixture, state.curr_mixture))
        component = state.curr_mixture[0]
        state.curr_mixture = state.curr_mixture.with_component(
            0, cc.with_parameters(component, lambda_=cc.all_parameters(component)["lambda_"] + 1.0)
        )
        return state


class FailingStep(MockSingleStep):
    """A step that sets H and fails on the given run."""

    def __init__(self, failing_run: int):
        super().__init__([])
        self.failing_run = failing_run
        self.runs = 0
        self.error = ValueError("simulated failure")

    def run(self, state: PipelineState) -> PipelineState:
        self.runs += 1
        state.H = np.full((len(state.X), state.curr_mixture.n_components), 0.5)
        if self.runs == self.failing_run:
            state.error = self.error
        return state


class ErrorSettingPruner(Pruner):
    """A pruner that removes nothing but leaves an error in the state."""

    def __init__(self):
        self.error = ValueError("left by the pruner")

    def prune(self, state: PipelineState) -> tuple[PipelineState, list[int]]:
        state.error = self.error
        return state, []


class TestHistoryAndState:
    def test_converts_input_data_to_float64(self, initial_mixture):
        step = DataProbeStep()

        Pipeline([step], [StopOnCheck(call_number=2)]).fit([1, 2, 6, 7], initial_mixture)

        assert step.dtypes == [np.float64, np.float64]

    def test_sets_previous_mixture_at_iteration_start(self, initial_mixture):
        step = PreviousMixtureProbeStep()

        Pipeline([step], [StopOnCheck(call_number=3)]).fit(np.array([1.0, 2.0]), initial_mixture)

        for previous, current_at_start in step.seen:
            assert previous is not None
            assert previous == current_at_start
        assert step.seen[0][0] == initial_mixture
        assert step.seen[1][0] != step.seen[0][0]

    def test_a_step_error_in_the_first_iteration_is_recorded_in_the_history(self, initial_mixture, sample_data):
        step = FailingStep(failing_run=1)
        pipeline = Pipeline([step], [StopOnCheck(call_number=5)])

        with pytest.warns(RuntimeWarning):
            result = pipeline.fit(sample_data, initial_mixture)

        assert len(pipeline.history) == 1
        record = pipeline.history[0]
        assert record.iteration == 0
        assert record.mixture is result
        np.testing.assert_array_equal(record.X, sample_data)
        assert record.H is not None
        assert record.H.shape == (len(sample_data), 2)
        assert record.pruners_used is None
        assert record.error is step.error

    def test_a_step_error_in_a_later_iteration_is_added_to_the_last_record(self, initial_mixture, sample_data):
        step = FailingStep(failing_run=2)
        pipeline = Pipeline([step], [StopOnCheck(call_number=5)])

        with pytest.warns(RuntimeWarning):
            pipeline.fit(sample_data, initial_mixture)

        assert len(pipeline.history) == 1
        assert pipeline.history[0].error is step.error

    def test_the_record_of_an_iteration_holds_the_data_the_pruners_and_the_error(self, initial_mixture, sample_data):
        pruner = ErrorSettingPruner()
        pipeline = Pipeline([MockSingleStep([])], [StopOnCheck(call_number=1)], [pruner])

        pipeline.fit(sample_data, initial_mixture)

        record = pipeline.history[0]
        np.testing.assert_array_equal(record.X, sample_data)
        assert record.pruners_used == [pruner]
        assert record.error is pruner.error
