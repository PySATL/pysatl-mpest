"""Module which contains Gaussian model class"""

import numpy as np
from scipy.stats import norm

from mpest.annotations import Params, Samples
from mpest.models.abstract_model import AModelDifferentiable, AModelWithGenerator


class LMomentsParameterMixin:
    """
    A class representing functions for calculating distribution parameters for the first two L moments
    """

    def calc_mean(self, moments: list[float]) -> float:
        """
        The function for calculating the parameter mean for the Gaussian distribution
        """

        return moments[0]

    def calc_variance(self, moments: list[float]) -> float:
        """
        The function for calculating the parameter variance for the Gaussian distribution
        """

        m2 = moments[1]
        return m2 * np.sqrt(np.pi)


class GaussianModel(AModelDifferentiable, AModelWithGenerator, LMomentsParameterMixin):
    """
    f(x) = e^(-1/2 * ((x - m) / sd)^2) / (sd * sqrt(2pi))

    sd = e^(_sd)

    O = [m, _sd]
    """

    @property
    def name(self) -> str:
        return "Gaussian"

    def params_convert_to_model(self, params: Params) -> Params:
        return np.array([params[0], np.log(params[1])])

    def params_convert_from_model(self, params: Params) -> Params:
        return np.array([params[0], np.exp(params[1])])

    def generate(self, params: Params, size: int = 1, normalized: bool = True) -> Samples:
        if not normalized:
            return np.array(norm.rvs(loc=params[0], scale=params[1], size=size))

        c_params = self.params_convert_from_model(params)
        return np.array(norm.rvs(loc=c_params[0], scale=c_params[1], size=size))

    def pdf(self, x: float, params: Params) -> float:
        m, sd = params
        sd = np.exp(sd)
        return np.exp(-0.5 * (((x - m) / sd) ** 2)) / (sd * np.sqrt(2 * np.pi))

    def lpdf(self, x: float, params: Params) -> float:
        p = self.pdf(x, params)
        if p <= 0:
            return -np.inf
        return np.log(p)

    def ldm(self, x: float, params: Params) -> float:
        """Method which returns logarithm of derivative with respect to mean"""

        m, sd = params
        return (x - m) / (np.exp(2 * sd))

    def ldsd(self, x: float, params: Params) -> float:
        """Method which returns logarithm of derivative with respect to variance"""

        m, sd = params
        return ((x - m) ** 2) / np.exp(2 * sd) - 1

    def ld_params(self, x: float, params: Params) -> np.ndarray:
        return np.array([self.ldm(x, params), self.ldsd(x, params)])

    def calc_params(self, moments: list[float]) -> np.ndarray:
        """
        The function for calculating params using L moments
        """

        return np.array([self.calc_mean(moments), self.calc_variance(moments)])
