"""Stopping strategy based on maximum iteration count."""

__author__ = "Danil Totmyanin, Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from ..breakpointer import Breakpointer
from ..pipeline_state import PipelineState


class StepBreakpointer(Breakpointer):
    """Stops the pipeline after a fixed number of iterations.

    The counter is reset by :meth:`reset`, which :meth:`Pipeline.fit` calls before every fitting run,
    so each ``fit`` starts counting from zero.

    Parameters
    ----------
    max_steps : int
        Maximum number of iterations before stopping. Must be >= 1.

    Attributes
    ----------
    max_steps : int
        Maximum iteration limit.

    Raises
    ------
    ValueError
        If ``max_steps`` is <= 0.

    Methods
    -------
    .. autosummary::
        :toctree: generated/

        check
        reset
    """

    def __init__(self, max_steps: int):
        self._validate(max_steps)

        self.max_steps = max_steps
        self._current_step = 0

    def _validate(self, max_steps: int):
        if max_steps <= 0:
            raise ValueError("The maximum number of steps must be greater than or equal to 1")

    def check(self, state: PipelineState) -> bool:
        """Evaluate if the iteration limit has been reached."""

        self._current_step += 1
        return self._current_step >= self.max_steps

    def reset(self) -> None:
        """Reset the iteration counter to zero."""
        self._current_step = 0
