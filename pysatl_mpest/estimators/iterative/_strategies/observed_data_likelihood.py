"""Observed data log-likelihood maximization strategy."""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
from pysatl_core.families import ParametricFamilyDistribution
from scipy.special import logsumexp

from ....core import _components
from ....optimizers import Optimizer
from ..pipeline_state import PipelineState
from ..steps import OptimizationBlock

# Unsupported observations contribute a finite penalty instead of freezing the whole objective.
LOG_FLOOR = -1e4


def observed_data_likelihood_strategy(
    component: ParametricFamilyDistribution,
    state: PipelineState,
    block: OptimizationBlock,
    optimizer: Optimizer,
) -> tuple[int, dict[str, float]]:
    """Maximize observed data log-likelihood for a component with fixed background.

    The mixture log-density of each observation is bounded below by ``LOG_FLOOR`` to assign a finite
    penalty to unsupported observations instead of freezing the objective. Parameter
    constraint violations or non-finite objectives evaluate to ``+inf``. Only
    parameters specified by ``block`` are updated, preserving fixed view parameters.
    Posterior responsibilities (``state.H``) are not used.

    Raises
    ------
    ValueError
        If ``block`` specifies parameters that are not free parameters of the component.
    """

    X = state.X
    component_id = block.component_id

    params_to_optimize = block.params_to_update(component)
    if not params_to_optimize:
        return component_id, {}

    mixture = state.curr_mixture
    log_weights = mixture.log_weights

    # Log of the contribution of the other components: the target component is masked out.
    joint = mixture.component_lpdf_matrix(X) + log_weights
    joint[:, component_id] = -np.inf
    log_background = logsumexp(joint, axis=1)

    log_target_weight = log_weights[component_id]
    current = _components.free_parameters(component)

    def target(vector_params: list[float]) -> float:
        try:
            candidate = _components.with_parameters(component, **dict(zip(params_to_optimize, vector_params)))
        except ValueError:
            return np.inf
        log_mixture = np.logaddexp(log_background, log_target_weight + _components.lpdf(candidate, X))
        bounded_log_mixture = np.maximum(log_mixture, LOG_FLOOR)
        value = float(-np.sum(bounded_log_mixture))
        return value if np.isfinite(value) else np.inf

    new_params = optimizer.minimize(target, [current[name] for name in params_to_optimize])
    return component_id, {name: float(value) for name, value in zip(params_to_optimize, new_params)}
