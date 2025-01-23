"""Estimators for estimating parameters in second stage of experiment"""
import random
from abc import abstractmethod
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor

from tqdm import tqdm

from experimental_env.utils import OrderedProblem, choose_best_mle
from mpest import Problem
from mpest.em import EM
from mpest.em.methods.l_moments_method import IndicatorEStep, LMomentsMStep
from mpest.em.methods.likelihood_method import BayesEStep, LikelihoodMStep
from mpest.em.methods.method import Method
from mpest.optimizers import ALL_OPTIMIZERS
from mpest.utils import ANamed, Factory, ResultWithLog

METHODS = {
    "Likelihood": [
        [Factory(BayesEStep), Factory(LikelihoodMStep, optimizer)]
        for optimizer in ALL_OPTIMIZERS
    ],
    "L-moments": [Factory(IndicatorEStep), Factory(LMomentsMStep)],
}


class AEstimator(ANamed):
    """
    An abstract class to describe the estimator.
    Implements the second stage of the experiment, evaluating the parameters of the mixture.
    """

    @abstractmethod
    def estimate(
        self, problems: list[Problem], cpu_count: int, seed: int
    ) -> list[ResultWithLog]:
        """
        The process of estimating the parameters of the mixture
        """


class LikelihoodEstimator(AEstimator):
    """
    An estimator using the maximum likelihood method with various optimizers.
    During the estimating process, the result with the best approximation is returned.
    """

    def __init__(
        self, brkpointer: EM.ABreakpointer, dst_checker: EM.ADistributionChecker
    ):
        """
        Class constructor
        """
        self._brkpointer = brkpointer
        self._dst_checker = dst_checker

    @property
    def name(self):
        return "MLE-EM"

    def _helper(self, problem: OrderedProblem):
        """
        Helper function for multiprocessed estimation
        """
        methods = [
            Method(step[0].construct(), step[1].construct())
            for step in METHODS["Likelihood"]
        ]
        ems = [EM(self._brkpointer, self._dst_checker, method) for method in methods]
        results = [em.solve_logged(problem, True, True, True) for em in ems]
        return choose_best_mle(problem.distributions, results), problem.number

    def estimate(
        self, problems: list[Problem], cpu_count: int, seed: int = 42
    ) -> list[ResultWithLog]:
        output = {}
        random.seed(seed)
        print("Starting Likelihood estimation")
        ordered_problem = [
            OrderedProblem(problem.samples, problem.distributions, i)
            for i, problem in enumerate(problems)
        ]
        with tqdm(total=len(problems)) as pbar:
            with ProcessPoolExecutor(max_workers=cpu_count) as executor:
                fs = [
                    executor.submit(self._helper, problem)
                    for problem in ordered_problem
                ]

                for f in as_completed(fs):
                    pbar.update()
                    res = f.result()
                    output[res[1]] = res[0]
        return [res for num, res in sorted(output.items())]


class LMomentsEstimator(AEstimator):
    """
    An estimator using the L-moments method.
    """

    def __init__(self, brkpointer, dst_checker):
        self._brkpointer = brkpointer
        self._dst_checker = dst_checker

    @property
    def name(self):
        return "LM-EM"

    def _helper(self, problem: OrderedProblem):
        """
        Helper function for multiprocessed estimation
        """
        steps = METHODS["L-moments"]
        new_method = Method(steps[0].construct(), steps[1].construct())
        em_factory = Factory(EM, self._brkpointer, self._dst_checker, new_method)

        return (
            em_factory.construct().solve_logged(problem, True, True, True),
            problem.number,
        )

    def estimate(
        self, problems: list[Problem], cpu_count: int, seed: int = 42
    ) -> list[ResultWithLog]:
        output = {}
        random.seed(seed)
        print("Starting L-moments estimation")
        ordered_problem = [
            OrderedProblem(problem.samples, problem.distributions, i)
            for i, problem in enumerate(problems)
        ]
        with tqdm(total=len(problems)) as pbar:
            with ProcessPoolExecutor(max_workers=cpu_count) as executor:
                fs = [
                    executor.submit(self._helper, problem)
                    for problem in ordered_problem
                ]
                for f in as_completed(fs):
                    pbar.update()
                    res = f.result()
                    output[res[1]] = res[0]

        return [res for num, res in sorted(output.items())]
