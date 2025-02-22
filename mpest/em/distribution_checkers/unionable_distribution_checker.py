"""Module which contains classes to reproduce unionable distribution checkers"""

from abc import ABC

from mpest.core.mixture_distribution import DistributionInMixture
from mpest.em import EM


class UnionDistributionChecker(EM.ADistributionChecker):
    """Class which represents union of distribution checkers"""

    def __init__(self, breakpointers: list[EM.ADistributionChecker] | None) -> None:
        super().__init__()
        self._distribution_checkers: list[EM.ADistributionChecker] = [] if breakpointers is None else breakpointers

    @property
    def name(self):
        return " + ".join(distribution_checker.name for distribution_checker in self._distribution_checkers)

    def __add__(
        self,
        additional: "UnionDistributionChecker | EM.ADistributionChecker",
    ):
        if isinstance(additional, UnionDistributionChecker):
            return UnionDistributionChecker(self._distribution_checkers + additional._distribution_checkers)
        return UnionDistributionChecker([*self._distribution_checkers, additional])

    def __radd__(
        self,
        additional: "UnionDistributionChecker | EM.ADistributionChecker",
    ):
        return self + additional

    @staticmethod
    def union(
        first: "UnionDistributionChecker | EM.ADistributionChecker",
        second: "UnionDistributionChecker | EM.ADistributionChecker",
    ):
        """Unions two distribution checkers"""
        if isinstance(first, UnionDistributionChecker):
            return first + second
        if isinstance(second, UnionDistributionChecker):
            return first + second
        return UnionDistributionChecker([first, second])

    def is_alive(
        self,
        step: int,
        distribution: DistributionInMixture,
    ) -> bool:
        for distribution_checker in self._distribution_checkers:
            if not distribution_checker.is_alive(step, distribution):
                return False
        return True


class AUnionableDistributionChecker(EM.ADistributionChecker, ABC):
    """Abstract class which can be used to make any distribution checker unionable"""

    def union(self, additional: EM.ADistributionChecker):
        """Method which allows that class to union with another distribution checker"""
        return UnionDistributionChecker.union(self, additional)

    def __add__(self, additional: EM.ADistributionChecker):
        return self.union(additional)

    def __radd__(self, additional: EM.ADistributionChecker):
        return self.union(additional)
