"""Tests for MaximizationStep"""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest
from numpy.testing import assert_allclose
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.core import _components as cc
from pysatl_mpest.estimators.iterative import (
    ExpectationStep,
    MaximizationStep,
    MaximizationStrategy,
    OptimizationBlock,
    PipelineState,
    PipelineStep,
)
from pysatl_mpest.optimizers import Optimizer
from pytest_mock import MockerFixture

from tests.helpers.families import family
from tests.helpers.pipeline import make_state

type RecordedCall = tuple[ParametricFamilyDistribution, PipelineState, OptimizationBlock, Optimizer]


@pytest.fixture
def mock_optimizer(mocker: MockerFixture) -> Optimizer:
    """Fixture to create a mock Optimizer."""

    return mocker.MagicMock(spec=Optimizer)


@pytest.fixture
def real_state() -> PipelineState:
    """Create a state with real Normal components and unequal responsibility sums."""
    normal = family(FamilyName.NORMAL)
    mixture = MixtureModel([normal(mu=0.0, sigma=1.0), normal(mu=5.0, sigma=2.0)])
    return PipelineState(
        X=np.array([1.0, 2.0, 3.0, 4.0]),
        H=np.array([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.2, 0.8]]),
        prev_mixture=None,
        curr_mixture=mixture,
        error=None,
    )


@pytest.fixture
def normal_mixture() -> MixtureModel:
    normal = family(FamilyName.NORMAL)
    return MixtureModel([normal(mu=0.0, sigma=1.0), normal(mu=5.0, sigma=1.0)], weights=[0.5, 0.5])


def uniform_responsibilities(mixture: MixtureModel) -> np.ndarray:
    """Create equal responsibilities for every observation and component."""
    return np.full((4, mixture.n_components), 1.0 / mixture.n_components)


def make_scripted_step(
    mocker: MockerFixture, blocks: list[OptimizationBlock], *updates: dict[str, float]
) -> MaximizationStep:
    """Build a step whose strategy returns scripted parameters for each block."""
    returned = iter(updates)

    def scripted(
        component: ParametricFamilyDistribution, state: PipelineState, block: OptimizationBlock, optimizer: Optimizer
    ) -> tuple[int, dict[str, float]]:
        return block.component_id, next(returned)

    mocker.patch.object(MaximizationStep, "_strategies", {MaximizationStrategy.QFUNCTION: scripted})
    return MaximizationStep(blocks, mocker.MagicMock(spec=Optimizer))


def make_recording_step(
    mocker: MockerFixture, blocks: list[OptimizationBlock], *updates: dict[str, float]
) -> tuple[MaximizationStep, list[RecordedCall]]:
    """Build a step whose strategy records its arguments and returns scripted parameters for each block."""
    returned = iter(updates)
    calls: list[RecordedCall] = []

    def scripted(
        component: ParametricFamilyDistribution, state: PipelineState, block: OptimizationBlock, optimizer: Optimizer
    ) -> tuple[int, dict[str, float]]:
        calls.append((component, state, block, optimizer))
        return block.component_id, next(returned)

    mocker.patch.object(MaximizationStep, "_strategies", {MaximizationStrategy.QFUNCTION: scripted})
    return MaximizationStep(blocks, mocker.MagicMock(spec=Optimizer)), calls


class TestConstruction:
    def test_stores_blocks_and_optimizer(self, mock_optimizer: Optimizer):
        blocks = [
            OptimizationBlock(
                component_id=0,
                params_to_optimize={"loc", "rate"},
                maximization_strategy=MaximizationStrategy.QFUNCTION,
            )
        ]

        step = MaximizationStep(blocks=blocks, optimizer=mock_optimizer)

        assert step.blocks == tuple(blocks)
        assert isinstance(step.blocks, tuple)
        assert step.optimizer == mock_optimizer
        assert isinstance(step, PipelineStep)

    def test_allows_expectation_as_the_next_step(self, mock_optimizer: Optimizer):
        step = MaximizationStep(blocks=[], optimizer=mock_optimizer)

        next_steps = step.available_next_steps

        assert len(next_steps) == 1
        assert next_steps[0] == ExpectationStep


class TestRun:
    def test_missing_responsibilities_set_error(self, mock_optimizer: Optimizer, real_state: PipelineState):
        state = real_state

        state.H = None
        step = MaximizationStep(blocks=[], optimizer=mock_optimizer)

        result_state = step.run(state)

        assert result_state is state
        assert isinstance(result_state.error, ValueError)
        assert str(result_state.error) == "Responsibility matrix H is not computed."

    def test_strategy_result_updates_component(
        self,
        mocker: MockerFixture,
        normal_mixture: MixtureModel,
    ):
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(normal_mixture), normal_mixture)
        block = OptimizationBlock(0, {"mu"}, MaximizationStrategy.QFUNCTION)
        step, calls = make_recording_step(mocker, [block], {"mu": 1.5})
        original = normal_mixture[0]

        step.run(state)

        component, seen_state, seen_block, seen_optimizer = calls[0]
        assert component is original
        assert seen_block is block
        assert seen_optimizer is step.optimizer
        assert seen_state is not state
        assert seen_state.curr_mixture is not state.curr_mixture
        # The strategy sees the wave's snapshot, which is not modified by the block's result.
        assert cc.all_parameters(seen_state.curr_mixture[0]) == {"mu": 0.0, "sigma": 1.0}
        assert cc.all_parameters(state.curr_mixture[0]) == {"mu": 1.5, "sigma": 1.0}
        # The mixture passed in the state is left unchanged.
        assert normal_mixture[0] is original
        assert cc.all_parameters(normal_mixture[0]) == {"mu": 0.0, "sigma": 1.0}

    def test_blocks_of_one_wave_share_one_snapshot(self, mocker: MockerFixture, normal_mixture: MixtureModel) -> None:
        """Blocks of one wave receive the same ``state.curr_mixture`` object: the wave's snapshot."""
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(normal_mixture), normal_mixture)
        blocks = [
            OptimizationBlock(0, {"mu"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(1, {"mu"}, MaximizationStrategy.QFUNCTION),
        ]
        originals = (normal_mixture[0], normal_mixture[1])
        step, calls = make_recording_step(mocker, blocks, {"mu": 9.0}, {"mu": 8.0})

        step.run(state)

        first_state, second_state = calls[0][1], calls[1][1]
        assert first_state.curr_mixture is second_state.curr_mixture
        assert first_state.curr_mixture is not state.curr_mixture
        assert calls[0][0] is originals[0]
        assert calls[1][0] is originals[1]
        # The snapshot is not modified: the first block's result went to the working copy.
        assert cc.all_parameters(second_state.curr_mixture[0])["mu"] == 0.0

    def test_next_wave_sees_the_previous_waves_update(self, mocker: MockerFixture, normal_mixture: MixtureModel):
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(normal_mixture), normal_mixture)
        blocks = [
            OptimizationBlock(0, {"mu"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(0, {"sigma"}, MaximizationStrategy.QFUNCTION),
        ]
        new_mu = 2.0
        step, calls = make_recording_step(mocker, blocks, {"mu": new_mu}, {"sigma": 3.0})

        step.run(state)

        second_component, second_state, second_block, _ = calls[1]
        assert second_block is blocks[1]
        assert cc.all_parameters(second_component)["mu"] == new_mu
        assert cc.all_parameters(second_state.curr_mixture[0])["mu"] == new_mu
        assert second_state.H is state.H
        assert second_state.X is state.X
        # Each wave computes from its own snapshot of the working mixture.
        assert calls[0][1].curr_mixture is not second_state.curr_mixture
        assert cc.all_parameters(calls[0][1].curr_mixture[0])["mu"] == 0.0
        assert cc.all_parameters(state.curr_mixture[0]) == {"mu": new_mu, "sigma": 3.0}
        # The mixture passed in the state is left unchanged.
        assert cc.all_parameters(normal_mixture[0]) == {"mu": 0.0, "sigma": 1.0}

    def test_same_wave_odl_block_reads_the_snapshot(self, mocker: MockerFixture, normal_mixture: MixtureModel) -> None:
        """In [ODL0, Q1] both blocks form one wave, so ODL reads component 1 before the wave's updates."""
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(normal_mixture), normal_mixture)
        new_mu = 9.0
        seen_mu: list[float] = []
        seen_mixtures: list[MixtureModel] = []

        def odl_probe(
            component: ParametricFamilyDistribution,
            state: PipelineState,
            block: OptimizationBlock,
            optimizer: Optimizer,
        ) -> tuple[int, dict[str, float]]:
            seen_mixtures.append(state.curr_mixture)
            seen_mu.append(cc.all_parameters(state.curr_mixture[1])["mu"])
            return block.component_id, {}

        def q_scripted(
            component: ParametricFamilyDistribution,
            state: PipelineState,
            block: OptimizationBlock,
            optimizer: Optimizer,
        ) -> tuple[int, dict[str, float]]:
            seen_mixtures.append(state.curr_mixture)
            return block.component_id, {"mu": new_mu}

        mocker.patch.object(
            MaximizationStep,
            "_strategies",
            {
                MaximizationStrategy.QFUNCTION: q_scripted,
                MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD: odl_probe,
            },
        )
        step = MaximizationStep(
            [
                OptimizationBlock(0, None, MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD),
                OptimizationBlock(1, {"mu"}, MaximizationStrategy.QFUNCTION),
            ],
            mocker.MagicMock(spec=Optimizer),
        )

        step.run(state)

        assert seen_mixtures[0] is seen_mixtures[1]
        assert seen_mu == [5.0]
        assert cc.all_parameters(state.curr_mixture[1])["mu"] == new_mu
        assert cc.all_parameters(normal_mixture[1]) == {"mu": 5.0, "sigma": 1.0}

    def test_odl_like_strategy_sees_earlier_block_updates(
        self, mocker: MockerFixture, normal_mixture: MixtureModel
    ) -> None:
        """A strategy reading ``state.curr_mixture`` observes components updated by earlier blocks."""
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(normal_mixture), normal_mixture)
        seen_mu: list[float] = []

        def q_scripted(
            component: ParametricFamilyDistribution,
            state: PipelineState,
            block: OptimizationBlock,
            optimizer: Optimizer,
        ) -> tuple[int, dict[str, float]]:
            return block.component_id, {"mu": 9.0}

        def odl_probe(
            component: ParametricFamilyDistribution,
            state: PipelineState,
            block: OptimizationBlock,
            optimizer: Optimizer,
        ) -> tuple[int, dict[str, float]]:
            seen_mu.append(cc.all_parameters(state.curr_mixture[0])["mu"])
            return block.component_id, {}

        mocker.patch.object(
            MaximizationStep,
            "_strategies",
            {
                MaximizationStrategy.QFUNCTION: q_scripted,
                MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD: odl_probe,
            },
        )
        step = MaximizationStep(
            [
                OptimizationBlock(0, {"mu"}, MaximizationStrategy.QFUNCTION),
                OptimizationBlock(1, None, MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD),
            ],
            mocker.MagicMock(spec=Optimizer),
        )

        step.run(state)

        assert seen_mu == [9.0]

    def test_does_not_update_weights(self, mocker: MockerFixture, normal_mixture: MixtureModel) -> None:
        H = np.array([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.2, 0.8]])
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), H, normal_mixture)
        log_weights_before = normal_mixture.log_weights.copy()

        make_scripted_step(
            mocker, [OptimizationBlock(0, None, MaximizationStrategy.QFUNCTION)], {"mu": 1.0, "sigma": 2.0}
        ).run(state)

        assert state.curr_mixture is not normal_mixture
        assert_allclose(state.curr_mixture.weights, normal_mixture.weights, atol=1e-12)
        np.testing.assert_array_equal(state.curr_mixture.log_weights, log_weights_before)
        np.testing.assert_array_equal(normal_mixture.log_weights, log_weights_before)

    def test_replaces_curr_mixture_with_a_new_object(self, mocker: MockerFixture, normal_mixture: MixtureModel) -> None:
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(normal_mixture), normal_mixture)
        mixture_before = state.curr_mixture

        make_scripted_step(
            mocker, [OptimizationBlock(0, None, MaximizationStrategy.QFUNCTION)], {"mu": 1.0, "sigma": 2.0}
        ).run(state)

        assert state.curr_mixture is not mixture_before
        assert state.curr_mixture is not normal_mixture
        assert state.curr_mixture[1] is normal_mixture[1]
        assert cc.all_parameters(state.curr_mixture[0]) == {"mu": 1.0, "sigma": 2.0}
        assert cc.all_parameters(normal_mixture[0]) == {"mu": 0.0, "sigma": 1.0}

    def test_out_of_range_component_is_rejected(self, mock_optimizer: Optimizer, real_state: PipelineState) -> None:
        block = OptimizationBlock(real_state.curr_mixture.n_components, None, MaximizationStrategy.QFUNCTION)
        step = MaximizationStep([block], mock_optimizer)

        with pytest.raises(IndexError, match="index out of range"):
            step.run(real_state)

        assert real_state.error is None

    def test_strategy_failure_leaves_state_mixture_untouched(
        self, mock_optimizer: Optimizer, real_state: PipelineState, mocker: MockerFixture
    ) -> None:
        state = real_state
        mixture = state.curr_mixture
        original_components = list(mixture)
        original_log_weights = mixture.log_weights
        blocks = [OptimizationBlock(index, {"mu"}, MaximizationStrategy.QFUNCTION) for index in range(2)]
        step = MaximizationStep(blocks, mock_optimizer)
        error = RuntimeError("Optimization diverged")
        strategy = mocker.Mock(side_effect=[(0, {"mu": 10.0}), error])
        mocker.patch.object(MaximizationStep, "_strategies", {MaximizationStrategy.QFUNCTION: strategy})
        update = mocker.spy(step, "_update_components_params")

        with pytest.raises(RuntimeError, match="Optimization diverged") as caught:
            step.run(state)

        assert caught.value is error
        assert strategy.call_count == len(blocks)
        # The first block's result went into the working copy, which is discarded on failure.
        assert update.call_count == 1
        assert state.curr_mixture is mixture
        assert all(current is original for current, original in zip(mixture, original_components, strict=True))
        assert mixture.log_weights is original_log_weights
        assert state.error is None

    def test_failure_in_a_later_wave_leaves_state_mixture_untouched(
        self, mock_optimizer: Optimizer, real_state: PipelineState, mocker: MockerFixture
    ) -> None:
        """A failure in a later wave discards the results of all earlier waves."""
        state = real_state
        mixture = state.curr_mixture
        original_components = list(mixture)
        original_log_weights = mixture.log_weights
        blocks = [
            OptimizationBlock(0, {"mu"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(0, {"sigma"}, MaximizationStrategy.QFUNCTION),
        ]
        step = MaximizationStep(blocks, mock_optimizer)
        error = RuntimeError("Optimization diverged")
        strategy = mocker.Mock(side_effect=[(0, {"mu": 10.0}), error])
        mocker.patch.object(MaximizationStep, "_strategies", {MaximizationStrategy.QFUNCTION: strategy})

        with pytest.raises(RuntimeError, match="Optimization diverged") as caught:
            step.run(state)

        assert caught.value is error
        assert state.curr_mixture is mixture
        assert all(current is original for current, original in zip(mixture, original_components, strict=True))
        assert mixture.log_weights is original_log_weights
        assert state.error is None

    def test_run_after_prune_plans_waves_for_current_blocks(
        self, mocker: MockerFixture, mock_optimizer: Optimizer
    ) -> None:
        """After pruning, the next run replans waves for the surviving blocks."""
        normal = family(FamilyName.NORMAL)
        mixture = MixtureModel([normal(mu=0.0, sigma=1.0), normal(mu=5.0, sigma=1.0), normal(mu=-5.0, sigma=1.0)])
        pruned = mixture.without_component(0)
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(pruned), pruned)
        blocks = [
            OptimizationBlock(0, {"mu"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(1, {"mu"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(1, {"sigma"}, MaximizationStrategy.QFUNCTION),
        ]
        step, calls = make_recording_step(mocker, blocks, {"mu": 1.0}, {"sigma": 3.0})

        step.clear_after_prune([0])
        step.run(state)

        # Surviving blocks [Q0{mu}, Q0{sigma}] form a chain of two waves.
        assert [call[2].params_to_optimize for call in calls] == [{"mu"}, {"sigma"}]
        assert calls[0][1].curr_mixture is not calls[1][1].curr_mixture
        assert cc.all_parameters(calls[1][1].curr_mixture[0])["mu"] == 1.0
        assert cc.all_parameters(state.curr_mixture[0]) == {"mu": 1.0, "sigma": 3.0}
        assert cc.all_parameters(state.curr_mixture[1]) == {"mu": -5.0, "sigma": 1.0}
        # The mixture passed in the state is left unchanged.
        assert cc.all_parameters(pruned[0]) == {"mu": 5.0, "sigma": 1.0}
        assert mixture.n_components == 3  # noqa: PLR2004

    def test_strategy_parameters_replace_component(self, mocker: MockerFixture, normal_mixture: MixtureModel) -> None:
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(normal_mixture), normal_mixture)
        original, untouched = normal_mixture[0], normal_mixture[1]

        make_scripted_step(
            mocker, [OptimizationBlock(0, None, MaximizationStrategy.QFUNCTION)], {"mu": 1.0, "sigma": 2.0}
        ).run(state)

        assert cc.all_parameters(state.curr_mixture[0]) == {"mu": 1.0, "sigma": 2.0}
        assert state.curr_mixture[1] is untouched
        assert cc.all_parameters(original) == {"mu": 0.0, "sigma": 1.0}
        # The mixture passed in the state is left unchanged.
        assert normal_mixture[0] is original
        assert cc.all_parameters(normal_mixture[0]) == {"mu": 0.0, "sigma": 1.0}

    def test_combines_updates_to_one_component(self, mocker: MockerFixture, normal_mixture: MixtureModel) -> None:
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(normal_mixture), normal_mixture)
        step = make_scripted_step(
            mocker,
            [
                OptimizationBlock(0, {"mu"}, MaximizationStrategy.QFUNCTION),
                OptimizationBlock(0, {"sigma"}, MaximizationStrategy.QFUNCTION),
            ],
            {"mu": 2.0},
            {"sigma": 3.0},
        )

        step.run(state)

        assert cc.all_parameters(state.curr_mixture[0]) == {"mu": 2.0, "sigma": 3.0}

    def test_preserves_view_parameters(self, mocker: MockerFixture) -> None:
        mixture = MixtureModel([family(FamilyName.NORMAL).view(mu=1.0)(sigma=2.0)])
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(mixture), mixture)

        make_scripted_step(
            mocker, [OptimizationBlock(0, {"sigma"}, MaximizationStrategy.QFUNCTION)], {"sigma": 3.0}
        ).run(state)

        assert cc.fixed_parameters(state.curr_mixture[0]) == {"mu": 1.0}
        assert cc.free_parameters(state.curr_mixture[0]) == {"sigma": 3.0}

    def test_rejects_fixed_parameter_updates(self, mocker: MockerFixture) -> None:
        component = family(FamilyName.NORMAL).view(mu=1.0)(sigma=2.0)
        mixture = MixtureModel([component])
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(mixture), mixture)

        with pytest.raises(ValueError, match=r"Cannot update \['mu'\]"):
            make_scripted_step(mocker, [OptimizationBlock(0, None, MaximizationStrategy.QFUNCTION)], {"mu": 5.0}).run(
                state
            )

        assert state.curr_mixture is mixture
        assert mixture[0] is component

    @pytest.mark.parametrize(
        "error",
        [NotImplementedError("strategy unavailable"), ValueError("invalid strategy parameters")],
        ids=["not-implemented", "value-error"],
    )
    def test_strategy_errors_propagate_without_setting_state_error(
        self, mocker: MockerFixture, mock_optimizer: Optimizer, real_state: PipelineState, error: Exception
    ) -> None:
        strategy = mocker.Mock(side_effect=error)
        mocker.patch.object(MaximizationStep, "_strategies", {MaximizationStrategy.QFUNCTION: strategy})
        step = MaximizationStep([OptimizationBlock(0, None, MaximizationStrategy.QFUNCTION)], mock_optimizer)

        with pytest.raises(type(error), match=str(error)):
            step.run(real_state)

        assert real_state.error is None


class TestClearAfterPrune:
    """After pruning, ``run`` applies the surviving blocks to the renumbered components."""

    def test_removes_pruned_blocks(self, mocker: MockerFixture):
        normal = family(FamilyName.NORMAL)
        mixture = MixtureModel([normal(mu=0.0, sigma=1.0), normal(mu=5.0, sigma=1.0), normal(mu=-5.0, sigma=1.0)])
        pruned = mixture.without_component(1)
        blocks = [
            OptimizationBlock(0, {"loc"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(1, {"rate"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(2, {"loc", "rate"}, MaximizationStrategy.QFUNCTION),
        ]
        step, calls = make_recording_step(mocker, blocks, {"mu": 1.0}, {"mu": 2.0})
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(pruned), pruned)

        step.clear_after_prune([1])
        step.run(state)

        # The blocks of components 0 and 2 survive, renumbered to 0 and 1.
        assert [call[2].component_id for call in calls] == [0, 1]
        assert calls[0][0] is pruned[0]
        assert calls[1][0] is pruned[1]
        assert cc.all_parameters(state.curr_mixture[0]) == {"mu": 1.0, "sigma": 1.0}
        assert cc.all_parameters(state.curr_mixture[1]) == {"mu": 2.0, "sigma": 1.0}

    def test_preserves_parameters_of_surviving_blocks(self, mocker: MockerFixture):
        normal = family(FamilyName.NORMAL)
        mixture = MixtureModel([normal(mu=0.0, sigma=1.0), normal(mu=5.0, sigma=1.0), normal(mu=-5.0, sigma=1.0)])
        pruned = mixture.without_component(1)
        blocks = [
            OptimizationBlock(0, {"loc"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(1, {"rate", "scale"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(2, {"loc", "rate"}, MaximizationStrategy.QFUNCTION),
        ]
        step, calls = make_recording_step(mocker, blocks, {"mu": 1.0}, {"mu": 2.0})
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(pruned), pruned)

        step.clear_after_prune([1])
        step.run(state)

        assert [call[2].params_to_optimize for call in calls] == [{"loc"}, {"loc", "rate"}]

    def test_shifts_unoptimized_components(self, mocker: MockerFixture) -> None:
        normal = family(FamilyName.NORMAL)
        mixture = MixtureModel([normal(mu=float(index), sigma=1.0) for index in range(5)])
        pruned = mixture.without_component(1)
        blocks = [
            OptimizationBlock(0, {"mu"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(4, {"sigma"}, MaximizationStrategy.QFUNCTION),
        ]
        step, calls = make_recording_step(mocker, blocks, {"mu": 10.0}, {"sigma": 3.0})
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(pruned), pruned)

        step.clear_after_prune([1])
        step.run(state)

        # The block of component 4 shifts to component 3 of the pruned mixture.
        assert [call[2].component_id for call in calls] == [0, 3]
        assert calls[1][0] is pruned[3]
        assert cc.all_parameters(state.curr_mixture[0]) == {"mu": 10.0, "sigma": 1.0}
        assert cc.all_parameters(state.curr_mixture[3]) == {"mu": 4.0, "sigma": 3.0}

    def test_shifts_each_block_for_one_component(self, mocker: MockerFixture) -> None:
        normal = family(FamilyName.NORMAL)
        mixture = MixtureModel([normal(mu=0.0, sigma=1.0), normal(mu=5.0, sigma=1.0), normal(mu=-5.0, sigma=1.0)])
        pruned = mixture.without_component(1)
        blocks = [
            OptimizationBlock(0, {"mu"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(0, {"sigma"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(2, {"rate"}, MaximizationStrategy.QFUNCTION),
        ]
        # The independent blocks share the first wave, so the call order is
        # [block mu of 0, block rate of 2 (renumbered to 1), block sigma of 0].
        step, calls = make_recording_step(mocker, blocks, {"mu": 1.0}, {"mu": 3.0}, {"sigma": 2.0})
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(pruned), pruned)

        step.clear_after_prune([1])
        step.run(state)

        assert [call[2].component_id for call in calls] == [0, 1, 0]
        assert [call[2].params_to_optimize for call in calls] == [{"mu"}, {"rate"}, {"sigma"}]
        assert cc.all_parameters(state.curr_mixture[0]) == {"mu": 1.0, "sigma": 2.0}
        assert cc.all_parameters(state.curr_mixture[1]) == {"mu": 3.0, "sigma": 1.0}

    def test_handles_duplicate_indices_and_multiple_removals(self, mocker: MockerFixture) -> None:
        normal = family(FamilyName.NORMAL)
        mixture = MixtureModel([normal(mu=float(index), sigma=1.0) for index in range(6)])
        pruned = mixture.without_component(3).without_component(1)
        blocks = [OptimizationBlock(index, {"mu"}, MaximizationStrategy.QFUNCTION) for index in [0, 2, 5]]
        step, calls = make_recording_step(mocker, blocks, {"mu": 10.0}, {"mu": 20.0}, {"mu": 50.0})
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(pruned), pruned)

        step.clear_after_prune([3, 1, 3])
        step.run(state)

        assert [call[2].component_id for call in calls] == [0, 1, 3]
        assert calls[2][0] is pruned[3]  # originally component 5

    def test_leaves_caller_blocks_unchanged(self, mocker: MockerFixture) -> None:
        normal = family(FamilyName.NORMAL)
        mixture = MixtureModel([normal(mu=0.0, sigma=1.0), normal(mu=5.0, sigma=1.0), normal(mu=-5.0, sigma=1.0)])
        pruned = mixture.without_component(1)
        blocks = [
            OptimizationBlock(0, {"mu"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(1, None, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(2, {"sigma"}, MaximizationStrategy.QFUNCTION),
        ]
        step, calls = make_recording_step(mocker, blocks, {"mu": 1.0}, {"sigma": 3.0})
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(pruned), pruned)

        step.clear_after_prune([1])
        step.run(state)

        # The caller's blocks are not mutated; the applied surviving blocks are updated copies.
        assert [block.component_id for block in blocks] == [0, 1, 2]
        assert step.blocks == tuple(blocks)
        for call, original in zip(calls, [blocks[0], blocks[2]], strict=True):
            surviving = call[2]
            assert surviving is not original
            assert surviving.params_to_optimize is original.params_to_optimize
            assert surviving.maximization_strategy is original.maximization_strategy

    def test_handles_empty_removal(self, mocker: MockerFixture):
        normal = family(FamilyName.NORMAL)
        mixture = MixtureModel([normal(mu=0.0, sigma=1.0), normal(mu=5.0, sigma=1.0)])
        blocks = [
            OptimizationBlock(0, {"loc"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(1, {"rate"}, MaximizationStrategy.QFUNCTION),
        ]
        step, calls = make_recording_step(mocker, blocks, {"mu": 1.0}, {"mu": 2.0})
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(mixture), mixture)

        step.clear_after_prune([])
        step.run(state)

        # An empty removal leaves the working list untouched: the original blocks are applied.
        assert [call[2].component_id for call in calls] == [0, 1]
        assert calls[0][2] is blocks[0]
        assert calls[1][2] is blocks[1]

    def test_handles_no_blocks(self, mocker: MockerFixture):
        normal = family(FamilyName.NORMAL)
        mixture = MixtureModel([normal(mu=0.0, sigma=1.0), normal(mu=5.0, sigma=1.0)])
        step, calls = make_recording_step(mocker, [])
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(mixture), mixture)

        step.clear_after_prune([0, 1])
        step.run(state)

        assert calls == []
        assert all(new is old for new, old in zip(state.curr_mixture, mixture, strict=True))

    def test_handles_multiple_removals(self, mocker: MockerFixture):
        normal = family(FamilyName.NORMAL)
        mixture = MixtureModel([normal(mu=float(index), sigma=1.0) for index in range(4)])
        pruned = mixture.without_component(3).without_component(1)
        blocks = [
            OptimizationBlock(0, {"loc"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(1, {"rate"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(2, {"scale"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(3, {"loc", "rate"}, MaximizationStrategy.QFUNCTION),
        ]
        step, calls = make_recording_step(mocker, blocks, {"mu": 1.0}, {"mu": 2.0})
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(pruned), pruned)

        step.clear_after_prune([1, 3])
        step.run(state)

        # The blocks of components 0 and 2 survive, renumbered to 0 and 1.
        assert [call[2].component_id for call in calls] == [0, 1]
        assert [call[2].params_to_optimize for call in calls] == [{"loc"}, {"scale"}]

    def test_shifts_blocks_after_removed_component(self, mocker: MockerFixture):
        normal = family(FamilyName.NORMAL)
        mixture = MixtureModel([normal(mu=float(index), sigma=1.0) for index in range(4)])
        pruned = mixture.without_component(0)
        blocks = [OptimizationBlock(i, {"mu"}, MaximizationStrategy.QFUNCTION) for i in range(4)]
        step, calls = make_recording_step(mocker, blocks, {"mu": 10.0}, {"mu": 20.0}, {"mu": 30.0})
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(pruned), pruned)

        step.clear_after_prune([0])
        step.run(state)

        assert [call[2].component_id for call in calls] == [0, 1, 2]
        assert calls[0][0] is pruned[0]  # originally component 1


class TestReset:
    """``reset`` restores the working block list from the constructor ``blocks``."""

    def test_blocks_keeps_the_constructor_blocks_after_pruning(self, mock_optimizer: Optimizer):
        blocks = [
            OptimizationBlock(0, {"mu"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(1, {"sigma"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(2, {"mu"}, MaximizationStrategy.QFUNCTION),
        ]
        step = MaximizationStep(blocks, mock_optimizer)

        step.clear_after_prune([1])

        assert step.blocks == tuple(blocks)
        assert all(current is original for current, original in zip(step.blocks, blocks, strict=True))

    def test_reset_restores_the_constructor_blocks_for_the_next_run(self, mocker: MockerFixture):
        normal = family(FamilyName.NORMAL)
        mixture = MixtureModel([normal(mu=0.0, sigma=1.0), normal(mu=5.0, sigma=1.0), normal(mu=-5.0, sigma=1.0)])
        blocks = [
            OptimizationBlock(0, {"mu"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(2, {"mu"}, MaximizationStrategy.QFUNCTION),
        ]
        step, calls = make_recording_step(mocker, blocks, {"mu": 1.0}, {"mu": 2.0})
        state = make_state(np.array([-1.0, 0.0, 0.5, 2.0]), uniform_responsibilities(mixture), mixture)

        step.clear_after_prune([1])
        step.reset()
        step.run(state)

        # The run applies the original blocks on the original mixture again.
        assert [call[2].component_id for call in calls] == [0, 2]
        assert calls[0][2] is blocks[0]
        assert calls[1][2] is blocks[1]
        assert cc.all_parameters(state.curr_mixture[0]) == {"mu": 1.0, "sigma": 1.0}
        assert cc.all_parameters(state.curr_mixture[2]) == {"mu": 2.0, "sigma": 1.0}
