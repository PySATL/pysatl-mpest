"""Module which contains Weibull model class"""

import math

import numpy as np
from scipy.stats import weibull_min

from mpest.annotations import Params, Samples
from mpest.models.abstract_model import AModelDifferentiable, AModelWithGenerator


class LMomentsParameterMixin:
    """
    A class representing functions for calculating distribution parameters for the first two L moments
    """

    def calc_k(self, moments: list[float]) -> float:
        """
        The function for calculating the parameter k for the Weibull distribution
        """

        m1, m2 = moments[0], moments[1]
        return -np.log(2) / np.log(1 - (m2 / m1))

    def calc_lambda(self, moments: list[float]) -> float:
        """
        The function for calculating the parameter lambda for the Weibull distribution
        """

        m1 = moments[0]
        k = self.calc_k(moments)
        return m1 / math.gamma(1 + 1 / k)


class WeibullModelExp(AModelDifferentiable, AModelWithGenerator, LMomentsParameterMixin):
    """
    f(x) = (k / lm) * (x / lm)^(k - 1) / e^((x / lm)^k)

    k = e^(_k)

    lm = e^(_lm)

    O = [_k, _lm]
    """

    @property
    def name(self) -> str:
        return "WeibullExp"

    def params_convert_to_model(self, params: Params) -> Params:
        return np.log(params)

    def params_convert_from_model(self, params: Params) -> Params:
        return np.exp(params)

    def generate(self, params: Params, size: int = 1, normalized: bool = True) -> Samples:
        if not normalized:
            return np.array(weibull_min.rvs(params[0], loc=0, scale=params[1], size=size))

        c_params = self.params_convert_from_model(params)
        return np.array(weibull_min.rvs(c_params[0], loc=0, scale=c_params[1], size=size))

    def pdf(self, x: float, params: Params) -> float:
        if x < 0:
            return 0
        ek, elm = np.exp(params)
        xl = x / elm
        return (ek / elm) * (xl ** (ek - 1.0)) / np.exp(xl**ek)

    def lpdf(self, x: float, params: Params) -> float:
        if x < 0:
            return -np.inf
        k, lm = params
        ek, elm = np.exp(params)
        lx = np.log(x)
        return k - ((x / elm) ** ek) - ek * lm - lx + ek * lx

    def ldk(self, x: float, params: Params) -> float:
        """Method which returns logarithm of derivative with respect to k"""

        if x < 0:
            return -np.inf
        ek, elm = np.exp(params)
        xlm = x / elm
        return 1.0 - ek * ((xlm**ek) - 1.0) * np.log(xlm)

    def ldl(self, x: float, params: Params) -> float:
        """Method which returns logarithm of derivative with respect to lm"""

        if x < 0:
            return -np.inf
        ek, elm = np.exp(params)
        return ek * ((x / elm) ** ek - 1.0)

    def ld_params(self, x: float, params: Params) -> np.ndarray:
        return np.array([self.ldk(x, params), self.ldl(x, params)])

    def calc_params(self, moments: list[float]):
        """
        The function for calculating params using L moments
        """

        return np.array([self.calc_k(moments), self.calc_lambda(moments)])
