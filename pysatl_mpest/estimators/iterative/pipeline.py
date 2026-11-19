"""Configurable iterative pipeline estimator for mixture models."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from ...core import MixtureModel
from ..base_estimator import BaseEstimator
from ._iteration_history import IterationRecord, IterationsHistory
from .breakpointer import Breakpointer
from .pipeline_state import PipelineState
from .pipeline_step import PipelineStep
from .pruner import Pruner


class Pipeline(BaseEstimator):
    """Iterative mixture model parameter estimator.

    Sequentially executes configured processing steps, applies pruners,
    and checks stopping criteria on each iteration until convergence.

    Steps and breakpointers may keep state between iterations. They are reused by every call of
    :meth:`fit`, which calls :meth:`PipelineStep.reset` and :meth:`Breakpointer.reset` on each of them
    before the first iteration, so no per-fit state carries over between fits. Pruners are reused as is.
    ``history`` is not cleared between calls, so records of repeated fits accumulate.

    Steps and pruners must replace ``state.curr_mixture`` with a new mixture object instead of
    modifying it in place, because ``prev_mixture`` and the history records hold references to it.

    Parameters
    ----------
    steps : Sequence[PipelineStep]
        Ordered sequence of steps executed in each iteration.
    breakpointers : Sequence[Breakpointer]
        Stopping criteria defining termination conditions. Must not be empty.
    pruners : Sequence[Pruner] | None, optional
        Component pruning strategies applied after step execution. Defaults to None.
    once_in_iterations : int, optional
        Iteration snapshot recording frequency in history. Defaults to 1.

    Attributes
    ----------
    steps : list[PipelineStep]
        Ordered list of iteration steps.
    breakpointers : list[Breakpointer]
        Configured stopping criteria.
    pruners : list[Pruner]
        Configured component pruners.
    history : IterationsHistory
        Recorded iteration snapshots.

    Raises
    ------
    ValueError
        If ``steps`` is empty or cannot form a closed loop (each step, including
        the last transitioning to the first, must permit the next in ``available_next_steps``),
        or if ``breakpointers`` is empty.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        fit
    """

    def __init__(
        self,
        steps: Sequence[PipelineStep],
        breakpointers: Sequence[Breakpointer],
        pruners: Sequence[Pruner] | None = None,
        once_in_iterations: int = 1,
    ):
        self._validate_steps(list(steps))

        if not breakpointers:
            raise ValueError(
                "The 'breakpointers' list cannot be empty. "
                "At least one stopping criterion must be provided to prevent an infinite loop."
            )

        self.breakpointers = list(breakpointers)
        self.pruners = list(pruners) if pruners else []
        self.steps = list(steps)
        self.history = IterationsHistory(once_in_iterations)

    def _validate_steps(self, steps: list[PipelineStep]):
        if not steps:
            raise ValueError("The 'steps' list cannot be empty for a Pipeline.")

        for i in range(-1, len(steps) - 1, 1):
            curr_step, next_step = steps[i], steps[i + 1]
            available_steps = tuple(curr_step.available_next_steps)

            if not isinstance(next_step, available_steps):
                raise ValueError(
                    f"Wrong pipeline configuration. Step '{curr_step}' have"
                    f"available next steps:'{curr_step.available_next_steps}', but got '{next_step}'"
                )

    def fit(self, X: ArrayLike, mixture: MixtureModel) -> MixtureModel:
        """Fit mixture model parameters using the iterative pipeline.

        Pruned components are immediately removed from responsibilities and notified
        to steps via :meth:`~PipelineStep.clear_after_prune`.

        Parameters
        ----------
        X : ArrayLike
            Input data sample.
        mixture : MixtureModel
            Initial mixture model to fit (not modified in-place).

        Returns
        -------
        MixtureModel
            Fitted mixture model.

        Raises
        ------
        ValueError
            If a maximization block requests an unknown or fixed parameter.
        IndexError
            If a block's component index is out of bounds.
        NotImplementedError
            If a maximization strategy is not implemented for the component's family.

        Warns
        -----
        RuntimeWarning
            If a step sets :attr:`PipelineState.error`, stopping fitting prematurely
            and returning the current mixture.
        """

        X = np.asarray(X, dtype=np.float64)
        state = PipelineState(X, None, None, mixture, None)

        for breakpointer in self.breakpointers:
            breakpointer.reset()

        for step in self.steps:
            step.reset()

        while True:
            state.prev_mixture = state.curr_mixture

            for step in self.steps:
                result_state = step.run(state)
                if result_state.error:
                    if len(self.history) > 0:
                        self.history[-1].error = result_state.error
                    else:
                        self.history.save_record(
                            IterationRecord(
                                self.history._counter,
                                result_state.curr_mixture,
                                result_state.X,
                                result_state.H,
                                None,
                                result_state.error,
                            )
                        )

                    warnings.warn(
                        f"Pipeline fitting stopped prematurely due to an error in step "
                        f"'{step.__class__.__name__}': {state.error}",
                        RuntimeWarning,
                    )
                    return result_state.curr_mixture
                state = result_state

            for pruner in self.pruners:
                state, removed_components_indices = pruner.prune(state)
                if len(removed_components_indices) != 0:
                    self._discard_pruned_components(state, removed_components_indices)

            self.history.save_record(
                IterationRecord(self.history._counter, state.curr_mixture, state.X, state.H, self.pruners, state.error)
            )

            if any(breakpointer.check(state) for breakpointer in self.breakpointers):
                break

        return state.curr_mixture

    def _discard_pruned_components(self, state: PipelineState, removed_components_indices: list[int]) -> None:
        if state.H is not None:
            state.H = np.delete(state.H, removed_components_indices, axis=1)
        for step in self.steps:
            step.clear_after_prune(removed_components_indices)
