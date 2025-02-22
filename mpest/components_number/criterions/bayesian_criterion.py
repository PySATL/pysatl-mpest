"""Module which contains Bayesian Information Criterion class"""

from math import log

from mpest.components_number.criterions.abstract_criterion import ACriterion
from mpest.mixture_distribution import DistributionInMixture
from mpest.types import Samples


class BIC(ACriterion):
    """
    BIC = k * Log(n) - 2 * Log(L)
    """

    @property
    def name(self):
        return "Bayesian_Infromation_Criterion"

    def estimate(
        self, distributions: list[DistributionInMixture], samples: Samples
    ) -> float:
        n = samples.size
        return self._k_parameters(distributions) * log(n) - 2 * self._loglikehood(
            distributions, samples
        )
