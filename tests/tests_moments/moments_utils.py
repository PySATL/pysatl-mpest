"""TODO"""

from mpest.core.problem import Problem, Result
from mpest.em import EM
from mpest.em.breakpointers import ParamDifferBreakpointer, StepCountBreakpointer
from mpest.em.distribution_checkers import (
    FiniteChecker,
    PriorProbabilityThresholdChecker,
)
from mpest.em.methods.moments_method import MomentsMStep
from mpest.em.methods.likelihood_method import BayesEStep
from mpest.em.methods.method import Method


def run_test(problem: Problem, deviation: float) -> Result:
    """TODO"""
    method = Method(BayesEStep(), MomentsMStep())
    em_algo = EM(
        StepCountBreakpointer() + ParamDifferBreakpointer(deviation=deviation),
        FiniteChecker() + PriorProbabilityThresholdChecker(),
        method,
    )

    return em_algo.solve(problem=problem)
