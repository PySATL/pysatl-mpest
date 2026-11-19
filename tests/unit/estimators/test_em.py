"""Tests for the standalone EM estimator facade."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from unittest.mock import create_autospec

import numpy as np
import pytest
from numpy.typing import NDArray
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators.em import EM
from pysatl_mpest.estimators.iterative import (
    Breakpointer,
    MaximizationStrategy,
    OptimizationBlock,
    Pruner,
)
from pysatl_mpest.estimators.iterative._iteration_history import IterationsHistory
from pysatl_mpest.optimizers import Optimizer
from pytest_mock import MockerFixture

from tests.helpers.families import family


@pytest.fixture
def mock_optimizer() -> Optimizer:
    return create_autospec(Optimizer, instance=True)


@pytest.fixture
def mock_breakpointers() -> list[Breakpointer]:
    return [create_autospec(Breakpointer, instance=True)]


@pytest.fixture
def mock_pruners() -> list[Pruner]:
    return [create_autospec(Pruner, instance=True)]


@pytest.fixture
def sample_mixture() -> MixtureModel:
    components = [
        family(FamilyName.NORMAL)(mu=0.0, sigma=1.0),
        family(FamilyName.EXPONENTIAL)(lambda_=1.0),
    ]
    return MixtureModel(components, weights=[0.4, 0.6])


@pytest.fixture
def sample_data() -> NDArray[np.float64]:
    return np.array([1.0, 2.0, 3.0, 11.0, 12.0, 13.0])


class TestInitialization:
    def test_copies_sequences_and_keeps_optimizer(
        self, mock_optimizer: Optimizer, mock_breakpointers: list[Breakpointer], mock_pruners: list[Pruner]
    ) -> None:
        em = EM(breakpointers=mock_breakpointers, pruners=mock_pruners, optimizer=mock_optimizer)

        assert em.breakpointers is not mock_breakpointers
        assert em.breakpointers == mock_breakpointers
        assert em.pruners is not mock_pruners
        assert em.pruners == mock_pruners
        assert em.optimizer is mock_optimizer
        assert em._history is None


class TestPipeline:
    def test_returns_result(
        self,
        mocker: MockerFixture,
        mock_optimizer: Optimizer,
        mock_breakpointers: list[Breakpointer],
        mock_pruners: list[Pruner],
        sample_mixture: MixtureModel,
        sample_data: NDArray[np.float64],
    ) -> None:
        mocker.patch("pysatl_mpest.estimators.em.ExpectationStep", autospec=True)
        mocker.patch("pysatl_mpest.estimators.em.MaximizationStep", autospec=True)
        MockPipeline = mocker.patch("pysatl_mpest.estimators.em.Pipeline", autospec=True)
        mock_pipeline_instance = MockPipeline.return_value
        mock_pipeline_instance.fit.return_value = sample_mixture
        mock_pipeline_instance.history = IterationsHistory(once_in_iterations=1)

        em = EM(breakpointers=mock_breakpointers, pruners=mock_pruners, optimizer=mock_optimizer)

        result = em.fit(sample_data, sample_mixture)

        MockPipeline.assert_called_once()
        mock_pipeline_instance.fit.assert_called_once_with(sample_data, sample_mixture)
        assert result is sample_mixture

    @pytest.mark.parametrize(
        ("fit_kwargs", "expected_frequency"),
        [({}, 1), ({"once_in_iterations": 5}, 5)],
        ids=["default_frequency", "explicit_frequency"],
    )
    def test_passes_steps_configuration_and_frequency(
        self,
        mocker: MockerFixture,
        mock_optimizer: Optimizer,
        mock_breakpointers: list[Breakpointer],
        mock_pruners: list[Pruner],
        sample_mixture: MixtureModel,
        sample_data: NDArray[np.float64],
        fit_kwargs: dict[str, int],
        expected_frequency: int,
    ) -> None:
        MockExpectationStep = mocker.patch("pysatl_mpest.estimators.em.ExpectationStep", autospec=True)
        MockWeightsStep = mocker.patch("pysatl_mpest.estimators.em.WeightsStep", autospec=True)
        MockMaximizationStep = mocker.patch("pysatl_mpest.estimators.em.MaximizationStep", autospec=True)
        MockPipeline = mocker.patch("pysatl_mpest.estimators.em.Pipeline", autospec=True)
        MockPipeline.return_value.history = IterationsHistory(once_in_iterations=1)

        em = EM(breakpointers=mock_breakpointers, pruners=mock_pruners, optimizer=mock_optimizer)
        em.fit(sample_data, sample_mixture, **fit_kwargs)

        MockExpectationStep.assert_called_once_with()
        MockWeightsStep.assert_called_once_with()
        MockMaximizationStep.assert_called_once()

        args, _ = MockPipeline.call_args
        pipeline_steps, pipeline_breakpointers, pipeline_pruners, pipeline_record_freq = args

        assert pipeline_steps == [
            MockExpectationStep.return_value,
            MockWeightsStep.return_value,
            MockMaximizationStep.return_value,
        ]
        assert pipeline_breakpointers == mock_breakpointers
        assert pipeline_pruners == mock_pruners
        assert pipeline_record_freq == expected_frequency

    def test_creates_q_function_block_per_component(
        self,
        mocker: MockerFixture,
        mock_optimizer: Optimizer,
        mock_breakpointers: list[Breakpointer],
        mock_pruners: list[Pruner],
        sample_mixture: MixtureModel,
        sample_data: NDArray[np.float64],
    ) -> None:
        mocker.patch("pysatl_mpest.estimators.em.ExpectationStep", autospec=True)
        MockMaximizationStep = mocker.patch("pysatl_mpest.estimators.em.MaximizationStep", autospec=True)
        MockPipeline = mocker.patch("pysatl_mpest.estimators.em.Pipeline", autospec=True)
        MockPipeline.return_value.history = IterationsHistory(once_in_iterations=1)

        em = EM(breakpointers=mock_breakpointers, pruners=mock_pruners, optimizer=mock_optimizer)
        em.fit(sample_data, sample_mixture)

        m_step_args, _ = MockMaximizationStep.call_args
        blocks, optimizer_arg = m_step_args

        assert optimizer_arg is mock_optimizer
        assert len(blocks) == sample_mixture.n_components

        for i, block in enumerate(blocks):
            assert isinstance(block, OptimizationBlock)
            assert block.component_id == i
            assert block.params_to_optimize is None
            assert block.maximization_strategy is MaximizationStrategy.QFUNCTION


class TestHistory:
    def test_is_unavailable_before_fit_and_exposes_recorded_iterations(
        self,
        mocker: MockerFixture,
        mock_optimizer: Optimizer,
        mock_pruners: list[Pruner],
        sample_mixture: MixtureModel,
        sample_data: NDArray[np.float64],
    ) -> None:
        mocker.patch("pysatl_mpest.estimators.em.ExpectationStep", autospec=True)
        mocker.patch("pysatl_mpest.estimators.em.MaximizationStep", autospec=True)
        MockPipeline = mocker.patch("pysatl_mpest.estimators.em.Pipeline", autospec=True)

        breakpointers = [create_autospec(Breakpointer, instance=True)]
        em = EM(breakpointers=breakpointers, pruners=mock_pruners, optimizer=mock_optimizer)

        with pytest.raises(AttributeError, match="History is not available. Call the 'fit' method first."):
            _ = em.history

        pipeline_history = IterationsHistory(once_in_iterations=1)
        mock_pipeline_instance = MockPipeline.return_value
        mock_pipeline_instance.history = pipeline_history

        em.fit(sample_data, sample_mixture)

        assert em.history is pipeline_history
