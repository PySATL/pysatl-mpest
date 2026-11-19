"""Integration tests for MaximizationStep with real strategies and components."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.core import _components as cc
from pysatl_mpest.estimators.iterative import (
    ExpectationStep,
    MaximizationStep,
    MaximizationStrategy,
    OptimizationBlock,
    PipelineState,
    WeightsStep,
)
from pysatl_mpest.estimators.iterative._strategies import (
    StrategyFunction,
    moments_strategy,
    observed_data_likelihood_strategy,
    q_function_strategy,
)
from pysatl_mpest.optimizers import Optimizer, ScipyNelderMead

from tests.helpers.families import family

_STRATEGY_FUNCTIONS: dict[MaximizationStrategy, StrategyFunction] = {
    MaximizationStrategy.QFUNCTION: q_function_strategy,
    MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD: observed_data_likelihood_strategy,
    MaximizationStrategy.MOMENTS: moments_strategy,
}


def run_sequential_reference(
    state: PipelineState, blocks: list[OptimizationBlock], optimizer: Optimizer
) -> MixtureModel:
    """Apply the blocks to the components of ``state.curr_mixture`` one by one in list order.

    Reproduces the sequential stage-1 semantics independently of
    :class:`MaximizationStep`: each block's result is applied to the working
    components immediately, so a later block sees the updates of all earlier ones.
    """
    components = list(state.curr_mixture.components)
    for block in blocks:
        working = state.curr_mixture.with_components(components)
        block_state = replace(state, curr_mixture=working)
        strategy = _STRATEGY_FUNCTIONS[block.maximization_strategy]
        component_id, params = strategy(working[block.component_id], block_state, block, optimizer)
        components[component_id] = cc.with_parameters(components[component_id], **params)
    return state.curr_mixture.with_components(components)


def test_moments_reject_unsupported_normal_parametrization() -> None:
    component = family(FamilyName.NORMAL)(parametrization_name="meanVar", mu=0.0, var=1.0)
    mixture = MixtureModel([component])
    state = PipelineState(np.array([1.0, 2.0]), np.ones((2, 1)), None, mixture, None)
    step = MaximizationStep([OptimizationBlock(0, None, MaximizationStrategy.MOMENTS)], ScipyNelderMead())

    with pytest.raises(NotImplementedError, match="Moments strategy is not implemented for the 'meanVar'"):
        step.run(state)

    assert state.error is None


def test_q_function_rejects_unknown_parameter() -> None:
    mixture = MixtureModel([family(FamilyName.NORMAL)(mu=0.0, sigma=1.0)])
    state = PipelineState(np.array([1.0, 2.0, 3.0]), np.ones((3, 1)), None, mixture, None)
    step = MaximizationStep(
        [OptimizationBlock(0, {"invalid_param"}, MaximizationStrategy.QFUNCTION)], ScipyNelderMead()
    )

    with pytest.raises(ValueError, match="invalid_param.*are not parameters"):
        step.run(state)

    assert state.error is None


def test_view_component_density_matches_plain_component() -> None:
    x = np.array([-1.0, 0.0, 0.5, 2.0])
    mixture = MixtureModel([family(FamilyName.NORMAL).view(mu=1.0)(sigma=2.0)])
    state = PipelineState(x, np.full((len(x), 1), 1.0), None, mixture, None)
    step = MaximizationStep([OptimizationBlock(0, {"sigma"}, MaximizationStrategy.QFUNCTION)], ScipyNelderMead())

    step.run(state)

    expected = family(FamilyName.NORMAL)(mu=1.0, sigma=1.25)
    assert cc.fixed_parameters(state.curr_mixture[0]) == {"mu": 1.0}
    assert cc.free_parameters(state.curr_mixture[0]) == {"sigma": 1.25}
    assert_allclose(cc.lpdf(state.curr_mixture[0], x), cc.lpdf(expected, x))


def test_run_refreshes_previously_cached_mixture_density() -> None:
    x = np.array([-1.0, 0.0, 0.5, 2.0])
    normal = family(FamilyName.NORMAL)
    mixture = MixtureModel([normal(mu=0.0, sigma=1.0), normal(mu=5.0, sigma=1.0)], weights=[0.5, 0.5])
    pdf_before = mixture.pdf(x)
    state = PipelineState(x, np.full((len(x), 2), 0.5), None, mixture, None)
    step = MaximizationStep([OptimizationBlock(0, {"mu", "sigma"}, MaximizationStrategy.QFUNCTION)], ScipyNelderMead())

    step.run(state)

    expected = MixtureModel(
        [normal(mu=0.375, sigma=np.sqrt(4.6875 / 4)), normal(mu=5.0, sigma=1.0)],
        weights=[0.5, 0.5],
    )
    assert_allclose(state.curr_mixture.pdf(x), expected.pdf(x))
    # The mixture passed in the state is left unchanged.
    assert_allclose(mixture.pdf(x), pdf_before)


def test_sequential_blocks_match_a_joint_block() -> None:
    """Blocks {mu}, {sigma} of one component give the same result as the joint block {mu, sigma}."""
    normal = family(FamilyName.NORMAL)
    truth = MixtureModel([normal(mu=0.0, sigma=1.0), normal(mu=6.0, sigma=1.5)], weights=[0.4, 0.6])
    x = np.asarray(truth.generate(size=2000, random_state=np.random.default_rng(42)), dtype=np.float64)
    start = MixtureModel([normal(mu=1.0, sigma=2.0), normal(mu=4.0, sigma=2.0)], weights=[0.5, 0.5])

    def run_m_step(blocks: list[OptimizationBlock]) -> MixtureModel:
        state = PipelineState(x, None, None, start, None)
        ExpectationStep().run(state)
        WeightsStep().run(state)
        MaximizationStep(blocks, ScipyNelderMead()).run(state)
        return state.curr_mixture

    split = run_m_step(
        [
            OptimizationBlock(0, {"mu"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(0, {"sigma"}, MaximizationStrategy.QFUNCTION),
        ]
    )
    joint = run_m_step([OptimizationBlock(0, {"mu", "sigma"}, MaximizationStrategy.QFUNCTION)])

    split_params = cc.all_parameters(split[0])
    joint_params = cc.all_parameters(joint[0])
    assert split_params.keys() == joint_params.keys()
    for name in joint_params:
        assert_allclose(split_params[name], joint_params[name], atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "blocks",
    [
        [
            OptimizationBlock(0, None, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(1, None, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(2, None, MaximizationStrategy.QFUNCTION),
        ],
        [
            OptimizationBlock(0, {"mu"}, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(0, {"sigma"}, MaximizationStrategy.QFUNCTION),
        ],
        [
            OptimizationBlock(0, None, MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD),
            OptimizationBlock(1, None, MaximizationStrategy.QFUNCTION),
        ],
        [
            OptimizationBlock(0, None, MaximizationStrategy.QFUNCTION),
            OptimizationBlock(1, None, MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD),
            OptimizationBlock(2, None, MaximizationStrategy.QFUNCTION),
        ],
    ],
    ids=["em-list", "same-component-chain", "odl-then-q", "q-odl-q"],
)
def test_wave_execution_matches_sequential_reference(blocks: list[OptimizationBlock]) -> None:
    """Wave execution gives exactly the sequential list-order result."""
    normal = family(FamilyName.NORMAL)
    truth = MixtureModel(
        [normal(mu=0.0, sigma=1.0), normal(mu=6.0, sigma=1.5), normal(mu=-5.0, sigma=0.8)],
        weights=[0.4, 0.35, 0.25],
    )
    x = np.asarray(truth.generate(size=2000, random_state=np.random.default_rng(42)), dtype=np.float64)
    start = MixtureModel(
        [normal(mu=1.0, sigma=2.0), normal(mu=4.0, sigma=2.0), normal(mu=-3.0, sigma=1.5)],
        weights=[1.0 / 3.0] * 3,
    )
    state = PipelineState(x, None, None, start, None)
    ExpectationStep().run(state)
    WeightsStep().run(state)

    optimizer = ScipyNelderMead()
    expected = run_sequential_reference(state, blocks, optimizer)
    MaximizationStep(blocks, optimizer).run(state)

    for index in range(expected.n_components):
        expected_params = cc.all_parameters(expected[index])
        actual_params = cc.all_parameters(state.curr_mixture[index])
        assert actual_params.keys() == expected_params.keys()
        for name in expected_params:
            assert_allclose(actual_params[name], expected_params[name], atol=1e-12, rtol=0)
