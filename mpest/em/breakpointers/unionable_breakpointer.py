"""Module which contains classes to reproduce unionable EM breakpointers"""

from abc import ABC

from mpest.core.mixture_distribution import MixtureDistribution
from mpest.em import EM


class UnionBreakpointer(EM.ABreakpointer):
    """Class which represents union of breakpointers"""

    def __init__(self, breakpointers: list[EM.ABreakpointer] | None) -> None:
        super().__init__()
        self._breakpointers: list[EM.ABreakpointer] = [] if breakpointers is None else breakpointers

    @property
    def name(self):
        return " + ".join(breakpointer.name for breakpointer in self._breakpointers)

    def __add__(self, additional: "UnionBreakpointer | EM.ABreakpointer"):
        if isinstance(additional, UnionBreakpointer):
            return UnionBreakpointer(self._breakpointers + additional._breakpointers)
        return UnionBreakpointer([*self._breakpointers, additional])

    def __radd__(self, additional: "UnionBreakpointer | EM.ABreakpointer"):
        return self + additional

    @staticmethod
    def union(
        first: "UnionBreakpointer | EM.ABreakpointer",
        second: "UnionBreakpointer | EM.ABreakpointer",
    ) -> "UnionBreakpointer":
        """Unions two EM breakpointers"""
        if isinstance(first, UnionBreakpointer):
            return first + second
        if isinstance(second, UnionBreakpointer):
            return first + second
        return UnionBreakpointer([first, second])

    def is_over(
        self,
        step: int,
        previous_step: MixtureDistribution | None,
        current_step: MixtureDistribution,
    ) -> bool:
        return any(breakpointer.is_over(step, previous_step, current_step) for breakpointer in self._breakpointers)


class AUnionableBreakpointer(EM.ABreakpointer, ABC):
    """Abstract class which can be used to make any EM breakpointer unionable"""

    def union(self, another: EM.ABreakpointer):
        """Method which allows that class to union with another EM breakpointer"""
        return UnionBreakpointer.union(self, another)

    def __add__(self, another: EM.ABreakpointer):
        return self.union(another)

    def __radd__(self, another: EM.ABreakpointer):
        return self.union(another)
