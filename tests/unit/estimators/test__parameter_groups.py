"""
Unit tests for shared parameter-group validation and optimization block construction.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Mapping, Sequence

import pytest
from pysatl_core.types import FamilyName
from pysatl_mpest.core import MixtureModel, _components
from pysatl_mpest.estimators._parameter_groups import build_optimization_blocks
from pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock
from pytest_mock import MockerFixture

from tests.helpers.families import family


@pytest.fixture
def two_component_mixture() -> MixtureModel:
    """A mixture whose component 0 has free parameters ``mu`` and ``sigma`` and component 1 has ``lambda_``."""
    return MixtureModel(
        [
            family(FamilyName.NORMAL)(mu=0.0, sigma=1.0),
            family(FamilyName.EXPONENTIAL)(lambda_=1.0),
        ],
        weights=[0.4, 0.6],
    )


@pytest.fixture
def partially_fixed_mixture() -> MixtureModel:
    """A mixture whose component 0 has ``mu`` fixed by a view and only ``sigma`` free."""
    return MixtureModel(
        [
            family(FamilyName.NORMAL).view(mu=0.0)(sigma=1.0),
            family(FamilyName.EXPONENTIAL)(lambda_=1.0),
        ],
        weights=[0.4, 0.6],
    )


class TestBuildOptimizationBlocks:
    @pytest.mark.parametrize("groups", [None, {}])
    def test_no_groups_create_joint_blocks(
        self,
        two_component_mixture: MixtureModel,
        groups: Mapping[int, Sequence[set[str]]] | None,
        mocker: MockerFixture,
    ):
        strategies = [MaximizationStrategy.QFUNCTION, MaximizationStrategy.MOMENTS]
        free_parameters = mocker.spy(_components, "free_parameters")
        fixed_parameters = mocker.spy(_components, "fixed_parameters")

        blocks = build_optimization_blocks(two_component_mixture, groups, strategies)

        assert blocks == [OptimizationBlock(0, None, strategies[0]), OptimizationBlock(1, None, strategies[1])]
        free_parameters.assert_not_called()
        fixed_parameters.assert_not_called()

    @pytest.mark.parametrize(
        ("groups", "expected"),
        [
            ({0: [{"sigma"}, {"mu"}]}, [(0, {"sigma"}), (0, {"mu"}), (1, None)]),
            ({1: [{"lambda_"}]}, [(0, None), (1, {"lambda_"})]),
            ({1: [{"lambda_"}], 0: [{"sigma"}, {"mu"}]}, [(0, {"sigma"}), (0, {"mu"}), (1, {"lambda_"})]),
        ],
        ids=["first", "second", "reverse-insertion"],
    )
    def test_groups_preserve_order_strategies_and_copy_sets(
        self,
        two_component_mixture: MixtureModel,
        groups: Mapping[int, Sequence[set[str]]],
        expected: Sequence[tuple[int, set[str] | None]],
    ):
        strategies = [MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD, MaximizationStrategy.QFUNCTION]

        blocks = build_optimization_blocks(two_component_mixture, groups, strategies)

        assert blocks == [OptimizationBlock(index, params, strategies[index]) for index, params in expected]
        explicit_groups = [group for index in sorted(groups) for group in groups[index]]
        explicit_blocks = [block for block in blocks if block.params_to_optimize is not None]
        for block, group in zip(explicit_blocks, explicit_groups, strict=True):
            assert block.params_to_optimize is not group
            assert block.params_to_optimize is not None
            block.params_to_optimize.add("other")
            assert "other" not in group

    @pytest.mark.parametrize("strategies", [[], [MaximizationStrategy.QFUNCTION] * 3])
    def test_strategy_count_must_match_components(
        self, two_component_mixture: MixtureModel, strategies: Sequence[MaximizationStrategy]
    ):
        with pytest.raises(ValueError, match="number of strategies"):
            build_optimization_blocks(two_component_mixture, None, strategies)


class TestParameterGroupValidation:
    def test_valid_partitions_are_accepted(self, two_component_mixture: MixtureModel):
        build_optimization_blocks(
            two_component_mixture, {0: [{"mu"}, {"sigma"}], 1: [{"lambda_"}]}, [MaximizationStrategy.QFUNCTION] * 2
        )
        build_optimization_blocks(two_component_mixture, {0: [{"mu", "sigma"}]}, [MaximizationStrategy.QFUNCTION] * 2)
        build_optimization_blocks(two_component_mixture, {1: [{"lambda_"}]}, [MaximizationStrategy.QFUNCTION] * 2)

    def test_negative_key_is_rejected(self, two_component_mixture: MixtureModel):
        with pytest.raises(ValueError, match="negative"):
            build_optimization_blocks(two_component_mixture, {-1: [{"mu"}]}, [MaximizationStrategy.QFUNCTION] * 2)

    @pytest.mark.parametrize("key", [2, 10], ids=["at-bound", "above-bound"])
    def test_out_of_bounds_key_is_rejected(self, two_component_mixture: MixtureModel, key: int):
        with pytest.raises(ValueError, match="out of bounds"):
            build_optimization_blocks(two_component_mixture, {key: [{"mu"}]}, [MaximizationStrategy.QFUNCTION] * 2)

    def test_empty_group_list_is_rejected(self, two_component_mixture: MixtureModel):
        with pytest.raises(ValueError, match="must not be empty"):
            build_optimization_blocks(two_component_mixture, {0: []}, [MaximizationStrategy.QFUNCTION] * 2)

    def test_empty_group_is_rejected(self, two_component_mixture: MixtureModel):
        with pytest.raises(ValueError, match="empty group"):
            build_optimization_blocks(two_component_mixture, {0: [{"mu"}, set()]}, [MaximizationStrategy.QFUNCTION] * 2)

    def test_overlapping_groups_are_rejected(self, two_component_mixture: MixtureModel):
        with pytest.raises(ValueError, match="overlap"):
            build_optimization_blocks(
                two_component_mixture, {0: [{"mu", "sigma"}, {"sigma"}]}, [MaximizationStrategy.QFUNCTION] * 2
            )

    def test_uncovered_parameters_are_rejected(self, two_component_mixture: MixtureModel):
        with pytest.raises(ValueError, match="not covered"):
            build_optimization_blocks(two_component_mixture, {0: [{"mu"}]}, [MaximizationStrategy.QFUNCTION] * 2)

    def test_unknown_parameters_are_rejected(self, two_component_mixture: MixtureModel):
        with pytest.raises(ValueError, match="are not parameters"):
            build_optimization_blocks(
                two_component_mixture, {0: [{"mu", "sigma", "x"}]}, [MaximizationStrategy.QFUNCTION] * 2
            )

    def test_fixed_parameters_are_rejected(self, partially_fixed_mixture: MixtureModel):
        with pytest.raises(ValueError, match="are fixed and cannot be optimized"):
            build_optimization_blocks(
                partially_fixed_mixture, {0: [{"mu"}, {"sigma"}]}, [MaximizationStrategy.QFUNCTION] * 2
            )

    def test_a_group_covering_only_free_parameters_of_a_view_is_accepted(self, partially_fixed_mixture: MixtureModel):
        build_optimization_blocks(
            partially_fixed_mixture, {0: [{"sigma"}], 1: [{"lambda_"}]}, [MaximizationStrategy.QFUNCTION] * 2
        )
