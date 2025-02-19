"""Module which contains Exponential model class"""

import numpy as np
from scipy.stats import expon

from mpest.annotations import Params, Samples
from mpest.models.abstract_model import AModelDifferentiable, AModelWithGenerator


class LMomentsParameterMixin:
    """
    A class representing functions for calculating distribution parameters for the first two L moments
    """

    def calc_lambda(self, moments: list[float]):
        """
        The function for calculating the parameter lambda for the Exponential distribution
        """

        return 1 / moments[0]


class ExponentialModel(AModelDifferentiable, AModelWithGenerator, LMomentsParameterMixin):
    """
    f(x) = lm * e^(-lx)

    lm = e^(_lm)

    O = [_lm]
    """

    @property
    def name(self) -> str:
        return "Exponential"

    def params_convert_to_model(self, params):
        return np.log(params)

    def params_convert_from_model(self, params):
        return np.exp(params)

    def generate(self, params: Params, size: int = 1, normalized: bool = True) -> Samples:
        if not normalized:
            return np.array(expon.rvs(scale=1 / params[0], size=size))

        c_params = self.params_convert_from_model(params)
        return np.array(expon.rvs(scale=1 / c_params[0], size=size))

    def pdf(self, x: float, params: Params) -> float:
        if x < 0:
            return 0
        (lm,) = params
        return np.exp(lm - np.exp(lm) * x)

    def lpdf(self, x: float, params: Params) -> float:
        if x < 0:
            return -np.inf
        (lm,) = params
        return lm - np.exp(lm) * x

    def ldl(self, x: float, params: Params) -> float:
        """Method which returns logarithm of derivative with respect to parameter lm"""

        if x < 0:
            return -np.inf
        (lm,) = params
        return 1 - np.exp(lm) * x

    def ld_params(self, x: float, params: Params) -> np.ndarray:
        return np.array([self.ldl(x, params)])

    def calc_params(self, moments: list[float]):
        """
        The function for calculating params using L moments
        """
        return np.array([self.calc_lambda(moments)])
