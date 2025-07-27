"""The module in which the moments method is presented"""

import numpy as np

from mpest import Samples
from mpest.core.distribution import Distribution
from mpest.core.mixture_distribution import MixtureDistribution
from mpest.core.problem import Problem, Result
from mpest.em.methods.abstract_steps import AMaximization
from mpest.exceptions import MStepError
from mpest.utils import ResultWithError

EResult = tuple[Problem, np.ndarray] | ResultWithError[MixtureDistribution]


class MomentsMStep(AMaximization[EResult]):
    """
    Class which calculate new params using matrix with indicator from E step.
    """

    def calc_order_moment_of_index_element(self, order: int, i: int, samples: Samples, indicators: np.ndarray) -> float:
        """
        A function that calculates the list of n-th moments of each distribution.

        :param order: Order of Moment.
        :param i: The number of the distribution for which we count the moment.
        :param samples: Ndarray with samples.
        :param indicators: Matrix with indicators

        :return:  order-Moment of index element.
        """

        sum_j_row_probabilities = np.sum(indicators[i])

        if sum_j_row_probabilities == 0:
            return 0

        moment_values = samples**order

        numerator = np.sum(moment_values * indicators[i])

        return numerator / sum_j_row_probabilities

    def step(self, e_result: EResult) -> Result:
        """
        A function that performs M step

        :param e_result: Tuple with problem, new_priors and indicators.
        """

        if isinstance(e_result, ResultWithError):
            return e_result

        problem, indicators = e_result

        samples = problem.samples

        mixture = problem.distributions

        new_priors = np.sum(indicators, axis=1) / len(samples)

        max_params_count = max(len(d.params) for d in mixture)
        moments = np.zeros(shape=[len(mixture), max_params_count])

        for j, d in enumerate(mixture):
            for r in range(len(d.params)):
                moments[j][r] = self.calc_order_moment_of_index_element(r + 1, j, samples, indicators)

        for i, d in enumerate(mixture):
            if d.model.name == "WeibullExp" and (moments[i][0] * moments[i][1] < 0):
                error = MStepError("The Weibull distribution degenerated in the first step.")
                return ResultWithError(mixture.distributions, error)

        new_distributions = []

        for j, d in enumerate(mixture):
            new_params = d.model.calc_moments_params(moments[j])
            new_d = Distribution(d.model, d.model.params_convert_to_model(new_params))
            new_distributions.append(new_d)

        new_mixture = MixtureDistribution.from_distributions(new_distributions, new_priors)
        return ResultWithError(new_mixture)
