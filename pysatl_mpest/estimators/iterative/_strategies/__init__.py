"""Maximization step strategies for updating mixture component parameters.

A strategy is called as ``strategy(component, state, block, optimizer)`` for the component
``block.component_id`` and returns ``(component_id, new_params)``. ``new_params`` maps the names of the
free parameters to their new values; parameters it omits keep their values, so an empty dict means
the component is left unchanged.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Callable
from typing import Any

from pysatl_core.families import ParametricFamilyDistribution

from ....optimizers import Optimizer
from ..pipeline_state import PipelineState
from ..steps.block import OptimizationBlock
from .moments import moments_strategy as _moments_strategy
from .observed_data_likelihood import observed_data_likelihood_strategy as _observed_data_likelihood_strategy
from .q_function import q_function_strategy as _q_function_strategy

type StrategyFunction = Callable[
    [ParametricFamilyDistribution, PipelineState, OptimizationBlock, Optimizer],
    tuple[int, dict[str, float]],
]

q_function_strategy: StrategyFunction = _q_function_strategy
observed_data_likelihood_strategy: StrategyFunction = _observed_data_likelihood_strategy
moments_strategy: StrategyFunction = _moments_strategy

__all__ = ["StrategyFunction", "moments_strategy", "observed_data_likelihood_strategy", "q_function_strategy"]
