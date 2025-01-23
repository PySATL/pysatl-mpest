""" The module in which the L moments method is presented """

import json
from math import ceil

import numpy as np

from mpest import Samples
from mpest.distribution import Distribution
from mpest.em.methods.abstract_steps import AExpectation, AMaximization
from mpest.exceptions import EStepError, MStepError
from mpest.mixture_distribution import MixtureDistribution
from mpest.problem import Problem, Result
from mpest.utils import ResultWithError, find_file

EResult = tuple[Problem, list[float], np.ndarray] | ResultWithError[MixtureDistribution]


class IndicatorEStep(AExpectation[EResult]):
    """
    Class which represents method for performing E step in L moments method.
    """

    def __init__(self):
        """
        Object constructor
        """

        self.indicators: np.ndarray

    def calc_indicators(self, problem: Problem) -> None:
        """
        A function that recalculates the matrix with indicators.

        :param problem: Object of class Problem, which contains samples and mixture.
        """

        samples, mixture = np.sort(problem.samples), problem.distributions
        priors = np.array([dist.prior_probability for dist in mixture])
        k, m = len(mixture), len(samples)

        z = np.zeros((k, m), dtype=float)

        pdf_values = np.zeros((k, m), dtype=float)
        for j, d_j in enumerate(mixture):
            pdf_values[j] = [d_j.model.pdf(samples[i], d_j.params) for i in range(m)]

        denominators = np.sum(priors[:, np.newaxis] * pdf_values, axis=0)
        if 0.0 in denominators:
            self.indicators = np.ndarray([])
            return None

        for j in range(k):
            numerators = priors[j] * pdf_values[j]
            z[j] = numerators / denominators

        self.indicators = z
        return None

    def update_priors(self, problem: Problem) -> list[float]:
        """
        A function that recalculates the list with prior probabilities.

        :param problem: Object of class Problem, which contains samples and mixture.
        :return: List with prior probabilities
        """

        k, m = len(problem.distributions), len(problem.samples)

        new_priors = []
        for j in range(k):
            p_j = np.sum([self.indicators[j][i] for i in range(m)])
            new_priors.append(p_j / m)

        return new_priors

    def step(self, problem: Problem) -> EResult:
        """
        A function that performs E step

        :param problem: Object of class Problem, which contains samples and mixture.
        :return: Tuple with problem, new_priors and indicators.
        """
        sorted_problem = Problem(np.sort(problem.samples), problem.distributions)

        if (
            any(d.model.name == "Gaussian" for d in sorted_problem.distributions)
            and sorted_problem.samples[0] < 0
            and not hasattr(self, "indicators")
        ):
            self.indicators = distribute_numbers_transposed(
                sorted_problem.samples, sorted_problem.distributions
            )
        else:
            self.calc_indicators(sorted_problem)

        if self.indicators.shape == ():
            error = EStepError("The indicators could not be calculated")
            return ResultWithError(sorted_problem.distributions, error)

        if np.isnan(self.indicators).any():
            return ResultWithError(sorted_problem.distributions, EStepError(""))

        new_priors = self.update_priors(sorted_problem)
        new_problem = Problem(
            sorted_problem.samples,
            MixtureDistribution.from_distributions(
                sorted_problem.distributions, new_priors
            ),
        )
        return new_problem, new_priors, self.indicators


class LMomentsMStep(AMaximization[EResult]):
    """
    Class which calculate new params using matrix with indicator from E step.
    """

    def __init__(self):
        with open(find_file("binoms.json", "/"), "r", encoding="utf-8") as f:
            self.binoms = json.load(f)

    def calculate_mr_j(
        self, r: int, j: int, samples: Samples, indicators: np.ndarray
    ) -> float:
        """
        A function that calculates the list of n-th moments of each distribution.

        :param r: Order of L-moment.
        :param j: The number of the distribution for which we count the L moment.
        :param samples: Ndarray with samples.
        :param indicators: Matrix with indicators

        :return: lj_r L-moment
        """

        n = len(samples)
        binoms = self.binoms
        mr_j = 0
        for k in range(r):
            b_num = np.sum(
                [
                    binoms[f"{round(np.sum(indicators[j][:i+1]))} {k}"]
                    * samples[i]
                    * indicators[j][i]
                    for i in range(k, n)
                ]
            )

            ind_sum = np.sum(indicators[j])
            b_den = ind_sum * binoms[f"{ceil(ind_sum)} {k}"]
            b_k = b_num / b_den
            p_rk = (
                (-1) ** (r - k - 1)
                * binoms[f"{r - 1} {k}"]
                * binoms[f"{r + k - 1} {k}"]
            )

            mr_j += p_rk * b_k
        return mr_j

    def step(self, e_result: EResult) -> Result:
        """
        A function that performs E step

        :param e_result: Tuple with problem, new_priors and indicators.
        """

        if isinstance(e_result, ResultWithError):
            return e_result

        problem, new_priors, indicators = e_result
        samples, mixture = problem.samples, problem.distributions

        max_params_count = max(len(d.params) for d in mixture)
        l_moments = np.zeros(shape=[len(mixture), max_params_count])
        for j, d in enumerate(mixture):
            for r in range(len(d.params)):
                l_moments[j][r] = self.calculate_mr_j(r + 1, j, samples, indicators)

        for i, d in enumerate(mixture):
            if d.model.name == "WeibullExp" and (l_moments[i][0] * l_moments[i][1] < 0):
                error = MStepError(
                    "The weibul distribution degenerated in the first step."
                )
                return ResultWithError(mixture.distributions, error)

        new_distributions = []

        for j, d in enumerate(mixture):
            new_params = d.model.calc_params(l_moments[j])
            new_d = Distribution(d.model, d.model.params_convert_to_model(new_params))
            new_distributions.append(new_d)

        new_mixture = MixtureDistribution.from_distributions(
            new_distributions, new_priors
        )
        return ResultWithError(new_mixture)


def distribute_numbers_transposed(numbers, mixture: MixtureDistribution):
    """
    A function that implements a heuristic for negative elements
    """

    # TODO: To complete the heuristic approach

    n = len(numbers)
    m = len(mixture)

    gaussian_indices = [
        i for i, dist in enumerate(mixture) if dist.model.name == "Gaussian"
    ]
    gaussian_index = gaussian_indices[0]

    result = np.zeros((m, n))

    for i, num in enumerate(numbers):
        if num < 0:
            result[gaussian_index, i] = 1
        else:
            uniform_probability = 1 / len(mixture)
            for j in range(len(mixture)):
                result[j, i] = uniform_probability

    return result
