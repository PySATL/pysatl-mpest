"""Module that represents method class"""

from typing import Generic, TypeVar

from mpest.core.mixture_distribution import MixtureDistribution
from mpest.core.problem import Problem
from mpest.em.methods.abstract_steps import AExpectation, AMaximization
from mpest.utils import ResultWithError

T = TypeVar("T")


class Method(Generic[T]):
    """
    Class that performs E and M steps.
    """

    def __init__(self, e_step: AExpectation[T], m_step: AMaximization[T]):
        """
        Object constructor.

        :param e_step: AExpectation object which performs E step
        :param e_step: AMaximization object which performs M step
        """

        self.e_step = e_step
        self.m_step = m_step

    def step(self, problem: Problem) -> ResultWithError[MixtureDistribution]:
        """
        Function that performs E and M steps

        :param problem: Object of Problem class which contains samples and mixture.
        """

        e_result = self.e_step.step(problem)
        return self.m_step.step(e_result)
