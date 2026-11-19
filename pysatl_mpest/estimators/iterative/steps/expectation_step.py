"""Expectation step for iterative estimation pipelines."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
from scipy.special import logsumexp

from ..pipeline_state import PipelineState
from ..pipeline_step import PipelineStep


class ExpectationStep(PipelineStep):
    """Expectation step computing component responsibility matrix H.

    Supports soft (probabilistic) or hard (winner-takes-all) assignment.
    Points outside the support of all components receive zero responsibilities.

    Parameters
    ----------
    is_soft : bool, optional
        If True (default), computes posterior probabilities.
        If False, assigns 1.0 to the most likely component and 0.0 otherwise.

    Attributes
    ----------
    is_soft : bool
        Flag indicating soft or hard assignment.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        run
        clear_after_prune
        reset
    """

    def __init__(self, is_soft: bool = True):
        self.is_soft = is_soft

    @property
    def available_next_steps(self) -> list[type[PipelineStep]]:
        """list[type[PipelineStep]]: Steps permitted to follow ExpectationStep."""

        from pysatl_mpest.estimators.iterative.steps.maximization_step import MaximizationStep
        from pysatl_mpest.estimators.iterative.steps.weights_step import WeightsStep

        return [WeightsStep, MaximizationStep]

    def run(self, state: PipelineState) -> PipelineState:
        """Compute responsibility matrix H and store it in pipeline state.

        The state is modified in place. Rows of points outside the support of every component are zeros,
        so they do not sum to 1.

        Parameters
        ----------
        state : PipelineState
            Current state containing observations X and mixture model.

        Returns
        -------
        PipelineState
            Updated state with computed responsibility matrix H.
        """

        X, mixture = state.X, state.curr_mixture

        log_weighted_likelihoods = mixture.component_lpdf_matrix(X) + mixture.log_weights
        log_denominator = logsumexp(log_weighted_likelihoods, axis=1, keepdims=True)

        # A point with zero density under every component gives -inf - (-inf) = NaN below;
        # such a point has no responsibilities, so its row of H is zeros.
        with np.errstate(invalid="ignore"):
            log_H = log_weighted_likelihoods - log_denominator

        H_soft = np.exp(log_H)
        H_soft[np.isnan(H_soft)] = 0.0

        if not self.is_soft:
            H_hard = np.zeros_like(H_soft, dtype=np.float64)

            has_mass = np.any(H_soft > 0, axis=1)
            max_indices = np.argmax(H_soft, axis=1)
            H_hard[np.flatnonzero(has_mass), max_indices[has_mass]] = 1.0

            state.H = H_hard
        else:
            state.H = H_soft

        return state

    def clear_after_prune(self, removed_components_indices: list[int]) -> None:
        pass

    def reset(self) -> None:
        """Return the step to its post-constructor state (no per-fit state)."""
        pass
