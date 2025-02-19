"""Module which contains EM breakpointer by step count"""

from mpest.core.mixture_distribution import MixtureDistribution
from mpest.em.breakpointers.unionable_breakpointer import AUnionableBreakpointer


class StepCountBreakpointer(AUnionableBreakpointer):
    """Class which represents EM breakpointer by step count"""

    def __init__(self, max_step: int | None = 16) -> None:
        self._max_step = max_step

    @property
    def max_step(self):
        """Max step getter"""
        return self._max_step

    @property
    def name(self):
        return f"StepCountBreakpointer(max_step={self.max_step})"

    def is_over(
        self,
        step: int,
        previous_step: MixtureDistribution | None,
        current_step: MixtureDistribution,
    ) -> bool:
        return (self.max_step is not None) and (step >= self.max_step)
