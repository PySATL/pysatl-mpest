"""A module that provides various metrics for evaluating the quality of parameter estimation."""

from abc import ABC, abstractmethod
from itertools import permutations

import numpy as np
from scipy.integrate import quad

from mpest import MixtureDistribution
from mpest.utils import ANamed


class AMetric(ANamed, ABC):
    """
    Abstract class of metric
    """

    @abstractmethod
    def error(
        self,
        base_mixture: MixtureDistribution,
        result_mixture: MixtureDistribution,
    ) -> float:
        """
        A public function for calculating the error of pair of mixtures
        """


class SquaredError(AMetric):
    r"""
    The class that calculates the SquaredError.

    .. math::
       \text{err} = \int_{-\infty}^{+\infty} (f_1(x) - f_2(x))^2 \,dx

    """

    @property
    def name(self) -> str:
        return "SquaredError"

    def error(self, base_mixture, result_mixture):
        def integrand(x, base_mixture, result_mixture):
            return (base_mixture.pdf(x) - result_mixture.pdf(x)) ** 2

        output = quad(integrand, -np.inf, +np.inf, args=(base_mixture, result_mixture))
        return output[0]


class Parametric(AMetric):
    """
    A class that calculates the absolute difference in the parameters of mixtures.
     Does not use the difference of a prior probabilities.
    """

    @property
    def name(self) -> str:
        return "ParamsError"

    def error(self, base_mixture, result_mixture):
        base_p, res_p = ([d.params for d in ld] for ld in (base_mixture, result_mixture))

        param_diff = min(sum(sum(abs(x - y)) for x, y in zip(base_p, _res_p)) for _res_p in permutations(res_p))

        return param_diff


METRICS = {
    SquaredError().name: SquaredError,
    Parametric().name: Parametric,
}
