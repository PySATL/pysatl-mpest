"""TODO"""

from mpest.em import EM
from mpest.em.breakpointers import ParamDifferBreakpointer, StepCountBreakpointer
from mpest.em.distribution_checkers import (
    FiniteChecker,
    PriorProbabilityThresholdChecker,
)
from mpest.em.methods.l_moments_method import IndicatorEStep, LMomentsMStep
from mpest.em.methods.method import Method
from mpest.problem import Problem, Result


def run_test(problem: Problem, deviation: float) -> Result:
    """TODO"""
    method = Method(IndicatorEStep(), LMomentsMStep())
    em_algo = EM(
        StepCountBreakpointer() + ParamDifferBreakpointer(deviation=deviation),
        FiniteChecker() + PriorProbabilityThresholdChecker(),
        method,
    )

    return em_algo.solve(problem=problem)
