"""Provides strategies for the Maximization-step based on the Q-function.

This module implements the logic for updating component parameters by
maximizing the Q-function (the expected complete-data log-likelihood).
It dispatches on the family and parametrization of the component to provide a generic,
optimization-based approach for any component, as well as specialized, more efficient
analytical solutions for specific ones: the `Normal` family with the ``meanStd``
parametrization and the `Exponential` family with the ``rate`` parametrization.
"""

__author__ = "Danil Totmyanin, Maksim Pastukhov, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
from pysatl_core.families import ParametricFamilyDistribution
from pysatl_core.types import FamilyName

from ....core import _components
from ....optimizers import Optimizer
from ..pipeline_state import PipelineState
from ..steps import OptimizationBlock
from ._dispatch import FamilyDispatch

NUMERICAL_TOLERANCE = 1e-9


# ------------------------
# Base Q-function strategy
# ------------------------


def _numerical_q_function(
    component: ParametricFamilyDistribution,
    state: PipelineState,
    block: OptimizationBlock,
    optimizer: Optimizer,
) -> tuple[int, dict[str, float]]:
    """Default M-step strategy maximizing Q-function via numerical optimization.

    Target evaluates to +inf where parameters violate parametrization constraints.
    """

    if state.H is None:
        raise ValueError("Responsibility matrix H is not computed.")

    X, H_j = state.X, state.H[:, block.component_id]
    component_id = block.component_id

    params_to_optimize = block.params_to_update(component)
    if not params_to_optimize:
        return component_id, {}

    current = _components.free_parameters(component)

    def target(vector_params: list[float]) -> float:
        try:
            candidate = _components.with_parameters(component, **dict(zip(params_to_optimize, vector_params)))
        except ValueError:
            return np.inf
        lpdf_values = _components.lpdf(candidate, X)
        # A point with zero responsibility must not contribute even where the density is zero:
        # 0 * (-inf) is NaN.
        safe_lpdf = np.where(H_j == 0, 0.0, lpdf_values)
        return float(-np.dot(H_j, safe_lpdf))

    new_params = optimizer.minimize(target, [current[name] for name in params_to_optimize])
    return component_id, {name: float(value) for name, value in zip(params_to_optimize, new_params)}


def _exponential_rate(
    component: ParametricFamilyDistribution, state: PipelineState, block: OptimizationBlock, optimizer: Optimizer
) -> tuple[int, dict[str, float]]:
    """Analytical Q-function maximizer for Exponential distribution in rate parametrization.

    If total responsibility or the weighted mean is <= NUMERICAL_TOLERANCE, parameter is unchanged.
    """

    if state.H is None:
        raise ValueError("Responsibility matrix H is not computed.")

    X = state.X
    H_j = state.H[:, block.component_id]

    params_to_optimize = block.params_to_update(component)
    new_params: dict[str, float] = {}

    N_j = np.sum(H_j)

    # If the component has negligible responsibility, do not update its parameters.
    if N_j <= NUMERICAL_TOLERANCE:
        return block.component_id, {}

    # Update lambda (rate) if it's in the optimization block
    if "lambda_" in params_to_optimize:
        weighted_sum_X = np.dot(H_j, X)
        denominator = weighted_sum_X / N_j

        if denominator > NUMERICAL_TOLERANCE:
            new_params["lambda_"] = float(1.0 / denominator)
        else:
            # If the weighted average is too close to zero,
            # leave rate unchanged to avoid infinity.
            new_params["lambda_"] = _components.free_parameters(component)["lambda_"]

    return block.component_id, new_params


# ----------------------------
# Normal distribution strategy
# ----------------------------


def _normal_mean_std(
    component: ParametricFamilyDistribution, state: PipelineState, block: OptimizationBlock, optimizer: Optimizer
) -> tuple[int, dict[str, float]]:
    """Analytical Q-function maximizer for Normal distribution in meanStd parametrization.

    If total responsibility is <= NUMERICAL_TOLERANCE, parameters are unchanged.
    If weighted variance is <= NUMERICAL_TOLERANCE, sigma is left unchanged.
    """

    if state.H is None:
        raise ValueError("Responsibility matrix H is not computed.")

    X = state.X
    H_j = state.H[:, block.component_id]

    params_to_optimize = block.params_to_update(component)
    new_params: dict[str, float] = {}

    N_j = np.sum(H_j)

    # If the component has negligible responsibility, do not update its parameters.
    if N_j <= NUMERICAL_TOLERANCE:
        return block.component_id, {}

    current = _components.all_parameters(component)

    # Update mean (mu) if it's in the optimization block
    if "mu" in params_to_optimize:
        weighted_sum_X = np.dot(H_j, X)
        new_params["mu"] = float(weighted_sum_X / N_j)

    # Update std (sigma) if it's in the optimization block
    if "sigma" in params_to_optimize:
        # Use the newly computed mean if available, otherwise use the existing one.
        mu = new_params.get("mu", current["mu"])

        # Calculate the weighted variance
        weighted_sum_sq_diff = np.dot(H_j, (X - mu) ** 2)

        new_variance = weighted_sum_sq_diff / N_j

        if new_variance > NUMERICAL_TOLERANCE:
            new_params["sigma"] = float(np.sqrt(new_variance))
        else:
            # If variance is too small, it can lead to instability.
            # Keep the old sigma to prevent it from collapsing to zero.
            new_params["sigma"] = current["sigma"]

    return block.component_id, new_params


# --------
# Dispatch
# --------

# The implementations are plain functions and are registered by a call, not by a decorator:
# the mutation testing tool does not mutate decorated functions.
q_function_strategy = FamilyDispatch(
    _numerical_q_function,
    name="q_function_strategy",
    doc="""M-step strategy based on maximizing the Q-function.

Selects the implementation by the family and parametrization of the component.
Analytical solutions are registered for the ``Normal`` family with the
``meanStd`` parametrization and the ``Exponential`` family with the ``rate``
parametrization. For any other pair the parameters requested in ``block`` are
maximized numerically by ``optimizer`` over the expected complete-data
log-likelihood; points violating parametrization constraints are rejected
with an infinite target value.

Parameters
----------
component : ParametricFamilyDistribution
    The mixture component to update.
state : PipelineState
    The pipeline state holding the data ``X`` and the responsibilities ``H``.
block : OptimizationBlock
    The block describing which parameters of ``component`` to update.
optimizer : Optimizer
    The optimizer used by the numerical fallback.

Returns
-------
tuple[int, dict[str, float]]
    The component id and a mapping of updated parameter names to new values.
""",
)
q_function_strategy.register(FamilyName.EXPONENTIAL, "rate")(_exponential_rate)
q_function_strategy.register(FamilyName.NORMAL, "meanStd")(_normal_mean_std)
