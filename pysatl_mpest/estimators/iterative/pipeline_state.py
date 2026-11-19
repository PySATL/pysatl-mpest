"""State container for iterative estimation pipelines."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from dataclasses import dataclass

from ...core import MixtureModel
from ...typings import MultivariateFloatArray, UnivariateFloatArray


@dataclass
class PipelineState:
    """State container passed between pipeline steps and pruners.

    Parameters
    ----------
    X : UnivariateFloatArray
        Input data sample.
    H : MultivariateFloatArray | None
        Responsibility matrix (posterior probabilities), where ``H[i, j]`` is
        the probability that observation `i` belongs to component `j`.
    prev_mixture : MixtureModel | None
        Mixture model snapshot from previous iteration for convergence evaluation.
    curr_mixture : MixtureModel
        Current mixture model being optimized.
    error : Exception | None
        Step failure indicator signaling premature termination.
    """

    X: UnivariateFloatArray
    H: MultivariateFloatArray | None
    prev_mixture: MixtureModel | None
    curr_mixture: MixtureModel
    error: Exception | None
