"""The module in which the L moments method is presented"""

import json
from math import ceil

import numpy as np

from mpest import Samples
from mpest.core.distribution import Distribution
from mpest.core.mixture_distribution import MixtureDistribution
from mpest.core.problem import Problem, Result
from mpest.em.methods.abstract_steps import AMaximization
from mpest.exceptions import MStepError
from mpest.utils import ResultWithError, find_file

EResult = tuple[Problem, np.ndarray] | ResultWithError[MixtureDistribution]


class LMomentsMStep(AMaximization[EResult]):
    """
    Class which calculate new params using matrix with indicator from E step.
    """

    def __init__(self):
        with open(find_file("binoms.json", "/"), encoding="utf-8") as f:
            self.binoms = json.load(f)

    def calculate_mr_j(self, r: int, j: int, samples: Samples, indicators: np.ndarray) -> float:
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
                    binoms[f"{round(np.sum(indicators[j][: i + 1]))} {k}"] * samples[i] * indicators[j][i]
                    for i in range(k, n)
                ]
            )

            ind_sum = np.sum(indicators[j])
            b_den = ind_sum * binoms[f"{ceil(ind_sum)} {k}"]
            b_k = b_num / b_den
            p_rk = (-1) ** (r - k - 1) * binoms[f"{r - 1} {k}"] * binoms[f"{r + k - 1} {k}"]

            mr_j += p_rk * b_k
        return mr_j

    def step(self, e_result: EResult) -> Result:
        """
        A function that performs E step

        :param e_result: Tuple with problem, new_priors and indicators.
        """

        if isinstance(e_result, ResultWithError):
            return e_result

        problem, indicators = e_result

        samples = problem.samples

        mixture = problem.distributions

        new_priors = np.sum(indicators, axis=1) / len(samples)

        max_params_count = max(len(d.params) for d in mixture)
        l_moments = np.zeros(shape=[len(mixture), max_params_count])
        for j, d in enumerate(mixture):
            for r in range(len(d.params)):
                l_moments[j][r] = self.calculate_mr_j(r + 1, j, samples, indicators)

        for i, d in enumerate(mixture):
            if d.model.name == "WeibullExp" and (l_moments[i][0] * l_moments[i][1] < 0):
                error = MStepError("The weibul distribution degenerated in the first step.")
                return ResultWithError(mixture.distributions, error)

        new_distributions = []

        for j, d in enumerate(mixture):
            new_params = d.model.calc_params(l_moments[j])
            new_d = Distribution(d.model, d.model.params_convert_to_model(new_params))
            new_distributions.append(new_d)

        new_mixture = MixtureDistribution.from_distributions(new_distributions, new_priors)
        return ResultWithError(new_mixture)
