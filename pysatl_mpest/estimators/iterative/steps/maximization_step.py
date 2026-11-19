"""Maximization step for iterative estimation pipelines."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import bisect
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType
from typing import ClassVar

from pysatl_core.families import ParametricFamilyDistribution

from ....core import _components
from ....optimizers import Optimizer
from .._strategies import StrategyFunction, moments_strategy, observed_data_likelihood_strategy, q_function_strategy
from .._wave_planner import plan_waves
from ..pipeline_state import PipelineState
from ..pipeline_step import PipelineStep
from .block import MaximizationStrategy, OptimizationBlock


class MaximizationStep(PipelineStep):
    """Maximization step updating mixture component parameters.

    Applies :class:`OptimizationBlock` tasks with sequential list-order semantics,
    grouping independent blocks into waves. Updates component parameters only;
    use :class:`WeightsStep` to update mixture weights.

    Parameters
    ----------
    blocks : Sequence[OptimizationBlock]
        Optimization tasks specifying components, parameters, and strategies.
    optimizer : Optimizer
        Numerical optimizer for strategies requiring numerical search.

    Attributes
    ----------
    blocks : tuple[OptimizationBlock, ...]
        Optimization tasks as given to the constructor; never reassigned.
        Pruning renumbers a private working copy that :meth:`reset` restores.
    optimizer : Optimizer
        Numerical optimizer instance.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        run
        clear_after_prune
        reset
    """

    _strategies: ClassVar[Mapping[MaximizationStrategy, StrategyFunction]] = MappingProxyType(
        {
            MaximizationStrategy.QFUNCTION: q_function_strategy,
            MaximizationStrategy.OBSERVED_DATA_LIKELIHOOD: observed_data_likelihood_strategy,
            MaximizationStrategy.MOMENTS: moments_strategy,
        }
    )

    def __init__(self, blocks: Sequence[OptimizationBlock], optimizer: Optimizer):
        self.blocks: tuple[OptimizationBlock, ...] = tuple(blocks)
        self._working_blocks: list[OptimizationBlock] = list(self.blocks)
        self.optimizer = optimizer

    @property
    def available_next_steps(self) -> list[type[PipelineStep]]:
        """list[type[PipelineStep]]: Steps permitted to follow MaximizationStep."""

        from pysatl_mpest.estimators.iterative.steps.expectation_step import ExpectationStep

        return [ExpectationStep]

    def _update_components_params(
        self, component: ParametricFamilyDistribution, params: dict[str, float]
    ) -> ParametricFamilyDistribution:
        return _components.with_parameters(component, **params)

    def run(self, state: PipelineState) -> PipelineState:
        """Execute the M-step, updating component parameters.

        Applies the blocks of the working list, which reflects pruning done by
        :meth:`clear_after_prune` until the next :meth:`reset`. All blocks are
        applied atomically: the results are collected in a working list of
        components, and ``state.curr_mixture`` is replaced by a new mixture
        with the updated components and the same weights. The mixture
        passed in the state is not modified. Strategy exceptions propagate
        without changing ``state.curr_mixture``. Missing responsibilities set
        :attr:`~pysatl_mpest.estimators.iterative.pipeline_state.PipelineState.error` instead.

        Parameters
        ----------
        state : PipelineState
            Current state containing responsibility matrix H and current mixture.

        Returns
        -------
        PipelineState
            Updated state with a new mixture model.

        Raises
        ------
        ValueError
            If a block requests an unknown or fixed parameter.
        IndexError
            If a block's component index is out of bounds.
        NotImplementedError
            If a strategy is not implemented for the component's family.
        """

        if state.H is None:
            state.error = ValueError("Responsibility matrix H is not computed.")
            return state

        components = list(state.curr_mixture.components)

        for wave in plan_waves(self._working_blocks):
            snapshot = state.curr_mixture.with_components(components)
            wave_state = replace(state, curr_mixture=snapshot)
            for index in wave:
                block = self._working_blocks[index]
                strategy = self._strategies[block.maximization_strategy]
                component_id, new_params = strategy(snapshot[block.component_id], wave_state, block, self.optimizer)
                components[component_id] = self._update_components_params(components[component_id], new_params)

        state.curr_mixture = state.curr_mixture.with_components(components)

        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        """Update the working optimization blocks after component pruning.

        Removes blocks associated with pruned components and shifts component_id
        for remaining blocks by the number of removed components before them.
        Surviving blocks are replaced with updated copies, so the block objects
        given to the constructor are not modified. The ``blocks`` attribute is
        left unchanged; the renumbering applies to the working list until the
        next :meth:`reset`.

        Parameters
        ----------
        removed_components_indices : list[int]
            Indices of components that were removed.
        """

        if not removed_components_indices or not self._working_blocks:
            return

        removed = sorted(set(removed_components_indices))
        self._working_blocks = [
            replace(block, component_id=block.component_id - bisect.bisect_left(removed, block.component_id))
            for block in self._working_blocks
            if block.component_id not in removed
        ]

    def reset(self) -> None:
        """Restore the working block list to the configured ``blocks``.

        :meth:`Pipeline.fit` calls this method before the first iteration, so
        pruning changes of a previous fit do not carry over.
        """
        self._working_blocks = list(self.blocks)
