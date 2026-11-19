"""Abstract base class for pipeline stopping criteria."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

from .pipeline_state import PipelineState


class Breakpointer(ABC):
    """Abstract base class for pipeline stopping conditions.

    Evaluated after pruning in each iteration; implementations must account for
    differing component counts between ``state.prev_mixture`` and ``state.curr_mixture``.

    ``check`` may keep state between calls and is expected to be called once per iteration. The pipeline stops
    at the first breakpointer that returns True, so the ones after it are not called in that iteration.

    Any state accumulated by ``check`` lives only within a single ``Pipeline.fit`` call: ``Pipeline.fit``
    invokes :meth:`reset` on every breakpointer before the first iteration, so reusing the estimator
    across fits starts from a clean state.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        check
        reset
    """

    @abstractmethod
    def check(self, state: PipelineState) -> bool:
        """Evaluate whether pipeline execution should terminate.

        Parameters
        ----------
        state : PipelineState
            Current state after step execution and pruning.

        Returns
        -------
        bool
            True if stopping condition is satisfied, False otherwise.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset all state accumulated by ``check`` between calls.

        Called by :meth:`Pipeline.fit` before the first iteration, so that
        every ``fit`` call on the same estimator starts from a clean state.
        Implementations without state may leave the body empty.
        """
