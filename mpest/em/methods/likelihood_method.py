"""The module in which the maximum likelihood method is presented"""

from functools import partial

import numpy as np

from mpest.core.distribution import Distribution
from mpest.core.mixture_distribution import MixtureDistribution
from mpest.core.problem import Problem, Result
from mpest.em.methods.abstract_steps import AExpectation, AMaximization
from mpest.exceptions import SampleError
from mpest.models import AModel, AModelDifferentiable
from mpest.optimizers import AOptimizerJacobian, TOptimizer
from mpest.utils import ResultWithError

EResult = tuple[list[float], np.ndarray, Problem] | ResultWithError[MixtureDistribution]


class BayesEStep(AExpectation[EResult]):
    """
    Class which represents Bayesian method for calculating matrix for M step in likelihood method
    """

    def step(self, problem: Problem) -> EResult:
        """
        A function that performs E step

        :param problem: Object of class Problem, which contains samples and mixture.
        :return: Return active_samples, matrix with probabilities and problem.
        """
        samples = problem.samples
        mixture = problem.distributions
        p_xij = []
        active_samples = []
        for x in samples:
            p = np.array([d.model.pdf(x, d.params) for d in mixture])
            if np.any(p):
                p_xij.append(p)
                active_samples.append(x)

        if not active_samples:
            error = SampleError("None of the elements in the sample is correct for this mixture")
            return ResultWithError(mixture, error)

        # h[j, i] contains probability of X_i to be a part of distribution j
        m = len(p_xij)
        k = len(mixture)
        h = np.zeros([k, m], dtype=float)
        curr_w = np.array([d.prior_probability for d in mixture])
        for i, p in enumerate(p_xij):
            wp = curr_w * p
            swp = np.sum(wp)

            if not swp:
                return ResultWithError(mixture, ZeroDivisionError())

            h[:, i] = wp / swp

        return active_samples, h, problem


# class ML(AExpectation[EResult]):
#     """
#     Class which represents ML method for calculating matrix for M step in likelihood method
#     """
#
#     def step(self, problem: Problem) -> EResult:
#         """
#         A function that performs E step
#
#         :param problem: Object of class Problem, which contains samples and mixture.
#         """


class LikelihoodMStep(AMaximization[EResult]):
    """
    Class which calculate new params using logarithm od likelihood function

    :param optimizer: The optimizer that is used in the step
    """

    def __init__(self, optimizer: TOptimizer):
        """
        Object constructor

        :param optimizer: The optimizer that is used in the step
        """
        self.optimizer = optimizer

    def step(self, e_result: EResult) -> Result:
        """
        A function that performs E step

        :param e_result: A tuple containing the arguments obtained from step E:
        active_samples, matrix with probabilities and problem.
        """

        if isinstance(e_result, ResultWithError):
            return e_result

        samples, h, problem = e_result
        optimizer = self.optimizer

        m = len(h[0])
        mixture = problem.distributions

        new_w = np.sum(h, axis=1) / m
        new_distributions: list[Distribution] = []
        for j, ch in enumerate(h[:]):
            d = mixture[j]

            def log_likelihood(params, ch, model: AModel):
                return -np.sum(ch * [model.lpdf(x, params) for x in samples])

            def jacobian(params, ch, model: AModelDifferentiable):
                return -np.sum(
                    ch * np.swapaxes([model.ld_params(x, params) for x in samples], 0, 1),
                    axis=1,
                )

            # maximizing log of likelihood function for every active distribution
            if isinstance(optimizer, AOptimizerJacobian):
                if not isinstance(d.model, AModelDifferentiable):
                    raise TypeError

                new_params = optimizer.minimize(
                    partial(log_likelihood, ch=ch, model=d.model),
                    d.params,
                    jacobian=partial(jacobian, ch=ch, model=d.model),
                )
            else:
                new_params = optimizer.minimize(
                    func=partial(log_likelihood, ch=ch, model=d.model),
                    params=d.params,
                )

            new_distributions.append(Distribution(d.model, new_params))
        return ResultWithError(MixtureDistribution.from_distributions(new_distributions, new_w))
