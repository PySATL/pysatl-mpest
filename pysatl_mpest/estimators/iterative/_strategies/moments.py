"""Provides strategies for the Maximization-step based on the Method of Moments.

This module implements the logic for updating component parameters by
equating theoretical moments of the distribution to the empirical weighted
moments of the data. It dispatches on the family and parametrization of the component to provide a generic
interface that can be specialized for specific distribution types.
"""

__author__ = "Aleksandra Ri"
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
# Base Moments strategy
# ------------------------


def _not_implemented(
    component: ParametricFamilyDistribution,
    state: PipelineState,
    block: OptimizationBlock,
    optimizer: Optimizer,
) -> tuple[int, dict[str, float]]:
    raise NotImplementedError(
        f"Moments strategy is not implemented for the '{component.parametrization_name}' "
        f"parametrization of the '{component.family_name}' family."
    )


def _exponential_rate(
    component: ParametricFamilyDistribution, state: PipelineState, block: OptimizationBlock, optimizer: Optimizer
) -> tuple[int, dict[str, float]]:
    """Update Exponential rate parameter using weighted first moment.

    If total component responsibility is within NUMERICAL_TOLERANCE of zero, or the weighted
    mean is <= NUMERICAL_TOLERANCE, the parameter is left unchanged.
    """

    if state.H is None:
        raise ValueError("Responsibility matrix H is not computed.")

    X = state.X
    H_j = state.H[:, block.component_id]

    params_to_optimize = block.params_to_update(component)
    new_params: dict[str, float] = {}

    N_j = np.sum(H_j)

    # If the component has negligible responsibility, do not update its parameters.
    if np.isclose(N_j, 0.0, atol=NUMERICAL_TOLERANCE):
        return block.component_id, {}

    weighted_sum_X = np.dot(H_j, X)

    m1 = weighted_sum_X / N_j

    # Update lambda (rate) if it's in the optimization block
    if "lambda_" in params_to_optimize:
        if m1 > NUMERICAL_TOLERANCE:
            new_params["lambda_"] = float(1.0 / m1)
        else:
            new_params["lambda_"] = _components.free_parameters(component)["lambda_"]

    return block.component_id, new_params


# ---------------------------------
# Normal distribution strategy
# ---------------------------------


def _normal_mean_std(
    component: ParametricFamilyDistribution, state: PipelineState, block: OptimizationBlock, optimizer: Optimizer
) -> tuple[int, dict[str, float]]:
    """Update Normal parameters (mu, sigma) using weighted moments.

    If total responsibility is within NUMERICAL_TOLERANCE, parameters are not updated.
    If weighted variance is <= NUMERICAL_TOLERANCE, sigma is left unchanged to prevent
    collapse to zero.
    """

    if state.H is None:
        raise ValueError("Responsibility matrix H is not computed.")

    X = state.X
    H_j = state.H[:, block.component_id]

    params_to_optimize = block.params_to_update(component)
    new_params: dict[str, float] = {}

    N_j = np.sum(H_j)

    # If the component has negligible responsibility, do not update its parameters.
    if np.isclose(N_j, 0.0, atol=NUMERICAL_TOLERANCE):
        return block.component_id, {}

    current = _components.all_parameters(component)

    if "mu" in params_to_optimize:
        new_loc = float(np.average(X, weights=H_j))
        new_params["mu"] = new_loc
    else:
        new_loc = current["mu"]

    if "sigma" in params_to_optimize:
        new_variance = np.average((X - new_loc) ** 2, weights=H_j)

        if new_variance > NUMERICAL_TOLERANCE:
            new_params["sigma"] = float(np.sqrt(new_variance))
        else:
            # The variance is degenerate: keep the old sigma to prevent it from collapsing to zero.
            new_params["sigma"] = current["sigma"]

    return block.component_id, new_params


# --------
# Dispatch
# --------

# The implementations are plain functions and are registered by a call, not by a decorator:
# the mutation testing tool does not mutate decorated functions.
moments_strategy = FamilyDispatch(
    _not_implemented,
    name="moments_strategy",
    doc="""M-step strategy based on the Method of Moments.

Selects the implementation by the family and parametrization of the component.
Analytical moment matching is registered for the ``Normal`` family with the
``meanStd`` parametrization and the ``Exponential`` family with the ``rate``
parametrization. No fallback exists for other pairs.

Parameters
----------
component : ParametricFamilyDistribution
    The mixture component to update.
state : PipelineState
    The pipeline state holding the data ``X`` and the responsibilities ``H``.
block : OptimizationBlock
    The block describing which parameters of ``component`` to update.
optimizer : Optimizer
    The optimizer passed to the strategy; unused by the analytical
    implementations.

Returns
-------
tuple[int, dict[str, float]]
    The component id and a mapping of updated parameter names to new values.

Raises
------
NotImplementedError
    If nothing is registered for the family and parametrization of the
    component.
""",
)
moments_strategy.register(FamilyName.EXPONENTIAL, "rate")(_exponential_rate)
moments_strategy.register(FamilyName.NORMAL, "meanStd")(_normal_mean_std)
