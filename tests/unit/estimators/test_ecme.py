"""Tests for the ECME class, which acts as a facade for the Pipeline."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Literal

import numpy as np
import pytest
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators import ECME
from pysatl_mpest.estimators.iterative import (
    Breakpointer,
    ExpectationStep,
    MaximizationStep,
    MaximizationStrategy,
    OptimizationBlock,
    PipelineState,
    WeightsStep,
)
from pysatl_mpest.optimizers import Optimizer
from pytest_mock import MockerFixture

from tests.helpers.families import family

Q = MaximizationStrategy.QFUNCTION
ODL = MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD


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
def ecme(mock_optimizer: Optimizer, mock_breakpointers: list[Breakpointer]) -> ECME:
    return ECME(breakpointers=mock_breakpointers, pruners=[], optimizer=mock_optimizer, default_strategy="odl")


@pytest.fixture
def three_normal_mixture() -> MixtureModel:
    # All components have ``mu`` and ``sigma`` free, so parameter groups can split them.
    components = [
        family(FamilyName.NORMAL)(mu=0.0, sigma=1.0),
        family(FamilyName.NORMAL)(mu=5.0, sigma=2.0),
        family(FamilyName.NORMAL)(mu=-5.0, sigma=1.5),
    ]
    return MixtureModel(components)


@pytest.fixture
def sample_data() -> np.ndarray:
    return np.array([1, 2, 3, 11, 12, 13], dtype=np.float64)


class TestInitialization:
    def test_copies_sequences_and_keeps_optimizer(
        self, mock_optimizer: Optimizer, mock_breakpointers: list[Breakpointer]
    ):
        pruners = []
        ecm_e = ECME(
            breakpointers=mock_breakpointers, pruners=pruners, optimizer=mock_optimizer, default_strategy="q-func"
        )

        assert ecm_e.breakpointers is not mock_breakpointers
        assert ecm_e.breakpointers == mock_breakpointers
        assert ecm_e.pruners is not pruners
        assert ecm_e.optimizer is mock_optimizer
        assert ecm_e.default_strategy_name == "q-func"
        assert ecm_e._history is None

    def test_rejects_an_unknown_default_strategy(self, mock_optimizer: Optimizer):
        with pytest.raises(ValueError, match="Unknown default_strategy"):
            ECME(breakpointers=[StopOnSecondCheck()], pruners=[], optimizer=mock_optimizer, default_strategy="bad")


class TestStrategyIndices:
    """Index arguments of ``fit`` select strategies and reject invalid indices."""

    @pytest.mark.parametrize(
        ("q_indices", "odl_indices", "message"),
        [(-1, None, "negative"), (None, 3, "out of bounds"), (1, 1, "specified for both")],
    )
    def test_invalid_indices_are_rejected_before_building_blocks(
        self,
        mocker: MockerFixture,
        ecme: ECME,
        three_normal_mixture: MixtureModel,
        sample_data: np.ndarray,
        q_indices: int | None,
        odl_indices: int | None,
        message: str,
    ):
        builder = mocker.patch("pysatl_mpest.estimators.ecme.build_optimization_blocks")

        with pytest.raises(ValueError, match=message):
            ecme.fit(sample_data, three_normal_mixture, q_indices, odl_indices, parameter_groups={0: []})

        builder.assert_not_called()

    @pytest.mark.parametrize("q_indices", [-1, [-1], [-2, 0]], ids=["int", "list", "list-with-negative"])
    def test_negative_q_indices_raise_value_error(
        self,
        mocker: MockerFixture,
        ecme: ECME,
        three_normal_mixture: MixtureModel,
        sample_data: np.ndarray,
        q_indices,
    ):
        MockPipeline = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")

        with pytest.raises(ValueError, match="negative"):
            ecme.fit(sample_data, three_normal_mixture, q_indices_raw=q_indices)

        MockPipeline.assert_not_called()

    @pytest.mark.parametrize("odl_indices", [-1, [-1]], ids=["int", "list"])
    def test_negative_odl_indices_raise_value_error(
        self,
        mocker: MockerFixture,
        ecme: ECME,
        three_normal_mixture: MixtureModel,
        sample_data: np.ndarray,
        odl_indices,
    ):
        MockPipeline = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")

        with pytest.raises(ValueError, match="negative"):
            ecme.fit(sample_data, three_normal_mixture, odl_indices_raw=odl_indices)

        MockPipeline.assert_not_called()

    @pytest.mark.parametrize("indices", [3, [0, 5]], ids=["at-bound", "list"])
    def test_out_of_bounds_indices_raise_value_error(
        self,
        mocker: MockerFixture,
        ecme: ECME,
        three_normal_mixture: MixtureModel,
        sample_data: np.ndarray,
        indices,
    ):
        MockPipeline = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")

        with pytest.raises(ValueError, match="out of bounds"):
            ecme.fit(sample_data, three_normal_mixture, q_indices_raw=indices)

        with pytest.raises(ValueError, match="out of bounds"):
            ecme.fit(sample_data, three_normal_mixture, odl_indices_raw=indices)

        MockPipeline.assert_not_called()

    def test_overlapping_strategy_indices_raise_value_error(
        self, mocker: MockerFixture, ecme: ECME, three_normal_mixture: MixtureModel, sample_data: np.ndarray
    ):
        MockPipeline = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")

        with pytest.raises(ValueError, match="specified for both"):
            ecme.fit(sample_data, three_normal_mixture, q_indices_raw=[0, 1], odl_indices_raw=1)

        MockPipeline.assert_not_called()


class TestPipeline:
    @pytest.mark.parametrize(
        ("default", "q_indices", "odl_indices", "strategies"),
        [
            ("odl", [1], [0], [ODL, Q, ODL]),
            ("q-func", [0], [1], [Q, ODL, Q]),
        ],
    )
    def test_passes_component_strategies_and_groups_to_builder(
        self,
        mocker: MockerFixture,
        mock_optimizer: Optimizer,
        mock_breakpointers: list[Breakpointer],
        three_normal_mixture: MixtureModel,
        sample_data: np.ndarray,
        default: Literal["odl", "q-func"],
        q_indices: list[int],
        odl_indices: list[int],
        strategies: list[MaximizationStrategy],
    ):
        ecm_e = ECME(mock_breakpointers, [], mock_optimizer, default_strategy=default)
        builder = mocker.patch("pysatl_mpest.estimators.ecme.build_optimization_blocks", return_value=[])
        mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")
        groups = {1: [{"mu"}, {"sigma"}]}

        ecm_e.fit(sample_data, three_normal_mixture, q_indices, odl_indices, parameter_groups=groups)

        builder.assert_called_once_with(three_normal_mixture, groups, strategies)

    def test_returns_pipeline_result(
        self,
        mocker: MockerFixture,
        ecme: ECME,
        three_normal_mixture: MixtureModel,
        sample_data: np.ndarray,
    ):
        mocker.patch("pysatl_mpest.estimators.ecme.build_optimization_blocks", return_value=[])
        MockPipeline = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")
        mock_pipeline_instance = MockPipeline.return_value
        mock_pipeline_instance.fit.return_value = three_normal_mixture

        result = ecme.fit(sample_data, three_normal_mixture)

        MockPipeline.assert_called_once()
        mock_pipeline_instance.fit.assert_called_once_with(sample_data, three_normal_mixture)
        assert result is three_normal_mixture

    def test_passes_steps_and_configuration_to_pipeline(
        self,
        mocker: MockerFixture,
        ecme: ECME,
        mock_optimizer: Optimizer,
        mock_breakpointers: list[Breakpointer],
        three_normal_mixture: MixtureModel,
        sample_data: np.ndarray,
    ):
        mocker.patch("pysatl_mpest.estimators.ecme.build_optimization_blocks", return_value=[])
        MockPipeline = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")
        record_frequency = 5
        num_of_steps = 3

        ecme.fit(sample_data, three_normal_mixture, once_in_iterations=record_frequency)

        args = MockPipeline.call_args[0]
        pipeline_steps, pipeline_breakpointers, pipeline_pruners, pipeline_record_freq = (
            args[0],
            args[1],
            args[2],
            args[3],
        )

        assert pipeline_breakpointers == mock_breakpointers
        assert pipeline_pruners == []
        assert pipeline_record_freq == record_frequency

        assert len(pipeline_steps) == num_of_steps
        assert isinstance(pipeline_steps[0], ExpectationStep)
        assert isinstance(pipeline_steps[1], WeightsStep)
        assert isinstance(pipeline_steps[2], MaximizationStep)
        assert pipeline_steps[2].optimizer is mock_optimizer


class TestParameterGroups:
    """Blocks built from ``parameter_groups`` and validation of the argument."""

    def test_groups_split_chains_for_q_and_odl_components(
        self, mocker: MockerFixture, ecme: ECME, three_normal_mixture: MixtureModel, sample_data: np.ndarray
    ):
        groups = {0: [{"mu"}, {"sigma"}], 1: [{"mu"}, {"sigma"}]}
        builder_blocks = [
            OptimizationBlock(0, {"mu"}, ODL),
            OptimizationBlock(0, {"sigma"}, ODL),
            OptimizationBlock(1, {"mu"}, Q),
            OptimizationBlock(1, {"sigma"}, Q),
            OptimizationBlock(2, None, ODL),
        ]
        builder = mocker.patch("pysatl_mpest.estimators.ecme.build_optimization_blocks", return_value=builder_blocks)
        pipeline = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")

        ecme.fit(sample_data, three_normal_mixture, q_indices_raw=1, parameter_groups=groups)

        builder.assert_called_once_with(three_normal_mixture, groups, [ODL, Q, ODL])
        blocks = pipeline.call_args[0][0][2].blocks
        assert [(block.component_id, block.params_to_optimize, block.maximization_strategy) for block in blocks] == [
            (1, {"mu"}, Q),
            (1, {"sigma"}, Q),
            (0, {"mu"}, ODL),
            (0, {"sigma"}, ODL),
            (2, None, ODL),
        ]

    def test_builder_error_prevents_pipeline_creation(
        self, mocker: MockerFixture, ecme: ECME, three_normal_mixture: MixtureModel, sample_data: np.ndarray
    ):
        mocker.patch("pysatl_mpest.estimators.ecme.build_optimization_blocks", side_effect=ValueError("invalid groups"))
        pipeline = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")

        with pytest.raises(ValueError, match="invalid groups"):
            ecme.fit(sample_data, three_normal_mixture, parameter_groups={0: []})

        pipeline.assert_not_called()

    def test_parameter_groups_is_keyword_only(
        self, ecme: ECME, three_normal_mixture: MixtureModel, sample_data: np.ndarray
    ):
        with pytest.raises(TypeError):
            ecme.fit(sample_data, three_normal_mixture, None, None, 1, {0: [{"mu"}, {"sigma"}]})


class TestHistory:
    def test_raises_attribute_error_before_fit(self, ecme: ECME):
        with pytest.raises(AttributeError, match="Call the 'fit' method first"):
            _ = ecme.history

    def test_returns_pipeline_history_after_fit(
        self, mocker: MockerFixture, ecme: ECME, three_normal_mixture: MixtureModel, sample_data: np.ndarray
    ):
        mocker.patch("pysatl_mpest.estimators.ecme.build_optimization_blocks", return_value=[])
        mock_pipeline_cls = mocker.patch("pysatl_mpest.estimators.ecme.Pipeline")
        mock_pipeline_instance = mock_pipeline_cls.return_value

        expected_history = mocker.Mock()
        mock_pipeline_instance.history = expected_history

        ecme.fit(sample_data, three_normal_mixture)

        assert ecme.history is expected_history
