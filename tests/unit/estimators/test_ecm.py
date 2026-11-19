"""Tests for the ECM class, which acts as a facade for the Pipeline."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators import ECM
from pysatl_mpest.estimators.iterative import (
    Breakpointer,
    ExpectationStep,
    MaximizationStep,
    MaximizationStrategy,
    OptimizationBlock,
    PipelineState,
    Pruner,
    WeightsStep,
)
from pysatl_mpest.optimizers import Optimizer
from pytest_mock import MockerFixture

from tests.helpers.families import family

# --- Mocks and Fixtures for testing ECM ---


class StopOnSecondCheck(Breakpointer):
    """Return true on the second check; the mocked pipeline never calls it."""

    def __init__(self):
        self.called = False

    def reset(self) -> None:
        self.called = False

    def check(self, state: PipelineState) -> bool:
        if self.called:
            return True
        self.called = True
        return False


@pytest.fixture
def mock_optimizer(mocker: MockerFixture) -> Optimizer:
    return mocker.Mock(spec=Optimizer)


@pytest.fixture
def mock_breakpointers() -> list[Breakpointer]:
    return [StopOnSecondCheck()]


@pytest.fixture
def mock_pruners() -> list[Pruner]:
    return []


@pytest.fixture
def sample_mixture() -> MixtureModel:
    # The mean of the first component is fixed with a view, so only its free parameters are to be optimized.
    components = [
        family(FamilyName.NORMAL).view(mu=0.0)(sigma=1.0),
        family(FamilyName.EXPONENTIAL)(lambda_=1.0),
    ]
    return MixtureModel(components, weights=[0.4, 0.6])


@pytest.fixture
def two_normal_mixture() -> MixtureModel:
    # Both components have two free parameters, so parameter groups can split them.
    components = [
        family(FamilyName.NORMAL)(mu=0.0, sigma=1.0),
        family(FamilyName.NORMAL)(mu=5.0, sigma=2.0),
    ]
    return MixtureModel(components, weights=[0.4, 0.6])


@pytest.fixture
def sample_data() -> np.ndarray:
    return np.array([1, 2, 3, 11, 12, 13], dtype=np.float64)


# --- Test Cases ---


class TestInitialization:
    def test_copies_sequences_and_keeps_optimizer(
        self, mock_optimizer: Optimizer, mock_breakpointers: list[Breakpointer], mock_pruners: list[Pruner]
    ):
        ecm = ECM(breakpointers=mock_breakpointers, pruners=mock_pruners, optimizer=mock_optimizer)

        assert ecm.breakpointers is not mock_breakpointers
        assert ecm.breakpointers == mock_breakpointers
        assert ecm.pruners is not mock_pruners
        assert ecm.pruners == mock_pruners
        assert ecm.optimizer is mock_optimizer
        assert ecm._history is None


class TestPipeline:
    def test_returns_pipeline_result(
        self,
        mocker: MockerFixture,
        mock_optimizer: Optimizer,
        mock_breakpointers: list[Breakpointer],
        mock_pruners: list[Pruner],
        sample_mixture: MixtureModel,
        sample_data: np.ndarray,
    ):
        mocker.patch("pysatl_mpest.estimators.ecm.build_optimization_blocks", return_value=[])
        MockPipeline = mocker.patch("pysatl_mpest.estimators.ecm.Pipeline")
        mock_pipeline_instance = MockPipeline.return_value
        mock_pipeline_instance.fit.return_value = sample_mixture

        ecm = ECM(breakpointers=mock_breakpointers, pruners=mock_pruners, optimizer=mock_optimizer)

        result = ecm.fit(sample_data, sample_mixture)

        MockPipeline.assert_called_once()
        mock_pipeline_instance.fit.assert_called_once_with(sample_data, sample_mixture)
        assert result is sample_mixture

    def test_uses_shared_builder_with_q_strategy_for_every_component(
        self,
        mocker: MockerFixture,
        mock_optimizer: Optimizer,
        mock_breakpointers: list[Breakpointer],
        mock_pruners: list[Pruner],
        two_normal_mixture: MixtureModel,
        sample_data: np.ndarray,
    ):
        groups = {0: [{"mu"}, {"sigma"}]}
        blocks = [
            OptimizationBlock(0, {"mu"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(0, {"sigma"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(1, None, MaximizationStrategy.QFUNCTION),
        ]
        builder = mocker.patch("pysatl_mpest.estimators.ecm.build_optimization_blocks", return_value=blocks)
        pipeline = mocker.patch("pysatl_mpest.estimators.ecm.Pipeline")
        ecm = ECM(breakpointers=mock_breakpointers, pruners=mock_pruners, optimizer=mock_optimizer)

        ecm.fit(sample_data, two_normal_mixture, parameter_groups=groups)

        builder.assert_called_once_with(
            two_normal_mixture, groups, [MaximizationStrategy.QFUNCTION] * two_normal_mixture.n_components
        )
        maximization_step = pipeline.call_args.args[0][2]
        assert maximization_step.blocks == tuple(blocks)

    def test_builder_error_prevents_pipeline_creation(
        self,
        mocker: MockerFixture,
        mock_optimizer: Optimizer,
        mock_breakpointers: list[Breakpointer],
        mock_pruners: list[Pruner],
        sample_mixture: MixtureModel,
        sample_data: np.ndarray,
    ):
        mocker.patch("pysatl_mpest.estimators.ecm.build_optimization_blocks", side_effect=ValueError("invalid"))
        pipeline = mocker.patch("pysatl_mpest.estimators.ecm.Pipeline")
        ecm = ECM(breakpointers=mock_breakpointers, pruners=mock_pruners, optimizer=mock_optimizer)

        with pytest.raises(ValueError, match="invalid"):
            ecm.fit(sample_data, sample_mixture)

        pipeline.assert_not_called()

    def test_passes_steps_and_configuration_to_pipeline(
        self,
        mocker: MockerFixture,
        mock_optimizer: Optimizer,
        mock_breakpointers: list[Breakpointer],
        mock_pruners: list[Pruner],
        sample_mixture: MixtureModel,
        sample_data: np.ndarray,
    ):
        mocker.patch("pysatl_mpest.estimators.ecm.build_optimization_blocks", return_value=[])
        MockPipeline = mocker.patch("pysatl_mpest.estimators.ecm.Pipeline")
        ecm = ECM(breakpointers=mock_breakpointers, pruners=mock_pruners, optimizer=mock_optimizer)
        record_frequency = 5
        num_of_steps = 3

        ecm.fit(sample_data, sample_mixture, once_in_iterations=record_frequency)

        args, _ = MockPipeline.call_args
        pipeline_steps, pipeline_breakpointers, pipeline_pruners, pipeline_record_freq = (
            args[0],
            args[1],
            args[2],
            args[3],
        )

        assert pipeline_breakpointers == mock_breakpointers
        assert pipeline_pruners == mock_pruners
        assert pipeline_record_freq == record_frequency

        assert len(pipeline_steps) == num_of_steps
        assert isinstance(pipeline_steps[0], ExpectationStep)
        assert isinstance(pipeline_steps[1], WeightsStep)
        assert isinstance(pipeline_steps[2], MaximizationStep)

        m_step = pipeline_steps[2]
        assert m_step.optimizer is mock_optimizer


class TestParameterGroups:
    """Public behavior of the ``parameter_groups`` fit argument."""

    def test_parameter_groups_is_keyword_only(
        self,
        mock_optimizer: Optimizer,
        mock_breakpointers: list[Breakpointer],
        mock_pruners: list[Pruner],
        two_normal_mixture: MixtureModel,
        sample_data: np.ndarray,
    ):
        ecm = ECM(breakpointers=mock_breakpointers, pruners=mock_pruners, optimizer=mock_optimizer)

        with pytest.raises(TypeError):
            ecm.fit(sample_data, two_normal_mixture, 1, {0: [{"mu"}, {"sigma"}]})


class TestHistory:
    def test_is_unavailable_before_fit_and_tracks_pipeline_history(
        self,
        mocker: MockerFixture,
        mock_optimizer: Optimizer,
        mock_pruners: list[Pruner],
        sample_mixture: MixtureModel,
        sample_data: np.ndarray,
    ):
        breakpointers = [StopOnSecondCheck()]
        mocker.patch("pysatl_mpest.estimators.ecm.build_optimization_blocks", return_value=[])
        MockPipeline = mocker.patch("pysatl_mpest.estimators.ecm.Pipeline")
        ecm = ECM(breakpointers=breakpointers, pruners=mock_pruners, optimizer=mock_optimizer)

        with pytest.raises(AttributeError, match="History is not available. Call the 'fit' method first."):
            _ = ecm.history

        mock_pipeline_instance_1 = MockPipeline.return_value
        mock_pipeline_instance_1.history = "first_history_object"

        ecm.fit(sample_data, sample_mixture)

        assert ecm.history == "first_history_object"
        first_history_id = id(ecm.history)

        mock_pipeline_instance_2 = MockPipeline.return_value
        mock_pipeline_instance_2.history = "second_history_object"

        ecm.fit(sample_data, sample_mixture)

        assert ecm.history == "second_history_object"
        second_history_id = id(ecm.history)

        assert first_history_id != second_history_id
