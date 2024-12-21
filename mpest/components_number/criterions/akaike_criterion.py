"""Module which contains Akaike Information Criterion class"""

from mpest.components_number.criterions.abstract_criterion import ACriterion
from mpest.mixture_distribution import DistributionInMixture
from mpest.types import Samples


class AIC(ACriterion):
    """
    AIC = 2k - 2Log(L)
    """

    @property
    def name(self):
        return "Akaike_Infromation_Criterion"

    def estimate(
        self, distributions: list[DistributionInMixture], samples: Samples
    ) -> float:
        return 2 * self._k_parameters(distributions) - 2 * self._loglikehood(
            distributions, samples
        )
