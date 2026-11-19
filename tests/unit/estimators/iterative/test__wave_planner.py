"""Tests for the private M-step wave planner."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock
from pysatl_mpest.estimators.iterative._wave_planner import STRATEGY_ACCESS, plan_waves

Q = MaximizationStrategy.QFUNCTION
M = MaximizationStrategy.MOMENTS
ODL = MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD


def make_block(component_id: int, strategy: MaximizationStrategy) -> OptimizationBlock:
    """Create an optimization block without parameter restrictions."""

    return OptimizationBlock(component_id=component_id, params_to_optimize=None, maximization_strategy=strategy)


def test_plan_waves_empty_list() -> None:
    """An empty block list produces no waves."""

    assert plan_waves([]) == []


def test_plan_waves_same_component_q_blocks() -> None:
    """Two Q blocks on the same component form two waves."""

    blocks = [make_block(0, Q), make_block(0, Q)]

    assert plan_waves(blocks) == [[0], [1]]


def test_plan_waves_different_component_q_blocks() -> None:
    """Q blocks on different components share one wave."""

    blocks = [make_block(0, Q), make_block(1, Q)]

    assert plan_waves(blocks) == [[0, 1]]


def test_plan_waves_odl_after_q_blocks() -> None:
    """An ODL block lands in a wave later than all preceding blocks."""

    blocks = [make_block(0, Q), make_block(1, Q), make_block(2, ODL)]

    assert plan_waves(blocks) == [[0, 1], [2]]


def test_plan_waves_two_odl_blocks() -> None:
    """A second ODL block lands in a wave later than the first one."""

    blocks = [make_block(0, ODL), make_block(1, ODL)]

    assert plan_waves(blocks) == [[0], [1]]


def test_plan_waves_em_list_single_wave() -> None:
    """An EM-style list with one Q block per component forms a single wave."""

    blocks = [make_block(0, Q), make_block(1, Q), make_block(2, Q)]

    assert plan_waves(blocks) == [[0, 1, 2]]


def test_plan_waves_ecm_mu_sigma_interleaved() -> None:
    """ECM blocks for two components alternate between two waves."""

    blocks = [make_block(0, Q), make_block(0, Q), make_block(1, Q), make_block(1, Q)]

    assert plan_waves(blocks) == [[0, 2], [1, 3]]


def test_plan_waves_odl_first_then_other_component() -> None:
    """An ODL block followed by a Q block on another component shares one wave."""

    blocks = [make_block(0, ODL), make_block(1, Q)]

    assert plan_waves(blocks) == [[0, 1]]


def test_plan_waves_q_odl_q() -> None:
    """A Q block after an ODL block joins the ODL wave but not the first Q wave."""

    blocks = [make_block(0, Q), make_block(1, ODL), make_block(2, Q)]

    assert plan_waves(blocks) == [[0], [1, 2]]


def test_plan_waves_odl_then_q_same_component() -> None:
    """A Q block on the component an earlier ODL wrote forms a later wave."""

    blocks = [make_block(0, ODL), make_block(0, Q)]

    assert plan_waves(blocks) == [[0], [1]]


def test_plan_waves_moments_behave_like_q() -> None:
    """Moments blocks follow the same dependency rules as Q blocks."""

    blocks = [make_block(0, M), make_block(0, M), make_block(1, M)]

    assert plan_waves(blocks) == [[0, 2], [1]]


def test_strategy_access_covers_all_strategies() -> None:
    """STRATEGY_ACCESS declares an access kind for every MaximizationStrategy."""

    assert set(STRATEGY_ACCESS) == set(MaximizationStrategy)
