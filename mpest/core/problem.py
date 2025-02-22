"""Module which represents problem, which can be solved by using this lib."""

from abc import ABC, abstractmethod

from mpest.annotations import Samples
from mpest.core.mixture_distribution import MixtureDistribution
from mpest.utils import ResultWithError


class Problem:
    """
    Class which represents the parameter estimation of mixture distribution problem.

    Described by samples and the initial approximation.
    Initial approximation is an mixture distribution.

    """

    def __init__(
        self,
        samples: Samples,
        distributions: MixtureDistribution,
    ) -> None:
        self._samples = samples
        self._distributions = distributions

    @property
    def samples(self):
        """Samples getter"""
        return self._samples

    @property
    def distributions(self):
        """Distributions getter"""
        return self._distributions


Result = ResultWithError[MixtureDistribution]


class ASolver(ABC):
    """
    Abstract class which represents solver for
    the parameter estimation of mixture distribution problem.
    """

    @abstractmethod
    def solve(self, problem: Problem) -> Result:
        """
        Method which solve the parameter estimation
        of mixture distribution problem.
        """
