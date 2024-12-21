"""Module which contains abstract information criterion"""

from abc import ABC, abstractmethod
from math import log

from mpest.mixture_distribution import DistributionInMixture
from mpest.types import Samples
from mpest.utils import ANamed


class ACriterion(ANamed, ABC):
    """
    Abstract class which represents information criterion with likehood realisation
    """

    @staticmethod
    def _loglikehood(
        distributions: list[DistributionInMixture], samples: Samples
    ) -> float:
        """The function for calculating the likelihood of a mixture model for a sample"""

        llh = 0
        for x in samples:
            lh = sum(
                d.prior_probability * d.model.pdf(x, d.params) for d in distributions
            )
            llh += log(lh) if lh > 0 else float("-inf")
        return llh

    @staticmethod
    def _k_parameters(distributions: list[DistributionInMixture]) -> int:
        """The function for calculating the number of parameters mixture model"""
        k = len(distributions) - 1
        for d in distributions:
            k += len(d.params)
        return k

    @abstractmethod
    def estimate(
        self, distributions: list[DistributionInMixture], samples: Samples
    ) -> float:
        """The function for estimating information criterion for the mixture model"""
