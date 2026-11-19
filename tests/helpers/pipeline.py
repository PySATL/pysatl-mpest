"""
Helpers for building pipeline states and optimization blocks in tests.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from pysatl_mpest.core import MixtureModel
from pysatl_mpest.estimators.iterative import MaximizationStrategy, OptimizationBlock, PipelineState


def make_state(X: np.ndarray, H: np.ndarray | None, mixture: MixtureModel) -> PipelineState:
    """
    Build a pipeline state with the given data and responsibilities.

    Parameters
    ----------
    X : np.ndarray
        The data sample.
    H : np.ndarray or None
        The responsibility matrix, or ``None`` if it has not been computed.
    mixture : MixtureModel
        The current mixture.

    Returns
    -------
    PipelineState
        The state with no previous mixture and no error.
    """
    return PipelineState(X=X, H=H, prev_mixture=None, curr_mixture=mixture, error=None)


def q_block(component_id: int = 0, params: set[str] | None = None) -> OptimizationBlock:
    """
    Build a block that maximizes the Q-function.

    Parameters
    ----------
    component_id : int, default=0
        The index of the component.
    params : set of str or None, default=None
        The parameters to optimize; ``None`` means all free parameters.

    Returns
    -------
    OptimizationBlock
        The block.
    """
    return OptimizationBlock(component_id, params, MaximizationStrategy.QFUNCTION)
