"""
Reference weighted moments for tests of the M-step strategies.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from pysatl_mpest.estimators.iterative import PipelineState


def weighted_moments(state: PipelineState, component_id: int = 0, mu: float | None = None) -> tuple[float, float]:
    """
    Weighted mean and weighted standard deviation of the data, with the responsibilities as weights.

    Parameters
    ----------
    state : PipelineState
        The state holding the data ``X`` and the responsibility matrix ``H``.
    component_id : int, default=0
        The column of ``H`` used as weights.
    mu : float or None, default=None
        The centre the spread is measured around; the weighted mean if ``None``.

    Returns
    -------
    tuple of float
        The weighted mean and the weighted standard deviation around ``mu``.
    """
    assert state.H is not None
    weights = state.H[:, component_id]
    mean = np.dot(weights, state.X) / weights.sum()
    centre = mean if mu is None else mu
    return float(mean), float(np.sqrt(np.dot(weights, (state.X - centre) ** 2) / weights.sum()))
