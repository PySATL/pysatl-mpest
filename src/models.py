from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import weibull_min, norm, expon

from utils import Samples, Params


class Model(ABC):
    @staticmethod
    @abstractmethod
    def name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def params_convert_to_model(params: Params) -> Params:
        pass

    @staticmethod
    @abstractmethod
    def params_convert_from_model(params: Params) -> Params:
        pass

    @staticmethod
    @abstractmethod
    def generate(params: Params, size: int = 1) -> Samples:
        pass

    @staticmethod
    @abstractmethod
    def p(x: float, params: Params) -> float:
        pass

    @staticmethod
    @abstractmethod
    def lp(x: float, params: Params) -> float:
        pass

    @staticmethod
    @abstractmethod
    def ld_params(x: float, params: Params) -> np.ndarray:
        pass


class WeibullModelExp(Model):
    """
    f(x) = (k / l) * (x / l)^(k - 1) / e^((x / l)^k)

    k = e^(_k)

    l = e^(_l)

    O = [_k, _l]
    """

    @staticmethod
    def name() -> str:
        return "WeibullExp"

    @staticmethod
    def params_convert_to_model(params: Params) -> Params:
        return np.log(params)

    @staticmethod
    def params_convert_from_model(params: Params) -> Params:
        return np.exp(params)

    @staticmethod
    def generate(params: Params, size: int = 1) -> Samples:
        return np.array(weibull_min.rvs(params[0], loc=0, scale=params[1], size=size))

    @staticmethod
    def p(x: float, params: Params) -> float:
        if x < 0:
            return 0
        ek, el = np.exp(params)
        xl = x / el
        return (ek / el) * (xl ** (ek - 1.0)) / np.exp(xl**ek)

    @staticmethod
    def lp(x: float, params: Params) -> float:
        if x < 0:
            return -np.inf
        k, l = params
        ek, el = np.exp(params)
        lx = np.log(x)
        return k - ((x / el) ** ek) - ek * l - lx + ek * lx

    @staticmethod
    def ldk(x: float, params: Params) -> float:
        if x < 0:
            return -np.inf
        ek, el = np.exp(params)
        xl = x / el
        return 1.0 - ek * ((xl**ek) - 1.0) * np.log(xl)

    @staticmethod
    def ldl(x: float, params: Params) -> float:
        if x < 0:
            return -np.inf
        ek, el = np.exp(params)
        return ek * ((x / el) ** ek - 1.0)

    @staticmethod
    def ld_params(x: float, params: Params) -> np.ndarray:
        return np.array(
            [WeibullModelExp.ldk(x, params), WeibullModelExp.ldl(x, params)]
        )


class GaussianModel(Model):
    """
    f(x) = e^(-1/2 * ((x - m) / sd)^2) / (sd * sqrt(2pi))

    sd = e^(_sd)

    O = [m, _sd]
    """

    @staticmethod
    def name() -> str:
        return "Gaussian"

    @staticmethod
    def params_convert_to_model(params: Params) -> Params:
        return np.array([params[0], np.log(params[1])])

    @staticmethod
    def params_convert_from_model(params: Params) -> Params:
        return np.array([params[0], np.exp(params[1])])

    @staticmethod
    def generate(params: Params, size: int = 1) -> Samples:
        return np.array(norm.rvs(loc=params[0], scale=params[1], size=size))

    @staticmethod
    def p(x: float, params: Params) -> float:
        m, sd = params
        sd = np.exp(sd)
        return np.exp(-0.5 * (((x - m) / sd) ** 2)) / (sd * np.sqrt(2 * np.pi))

    @staticmethod
    def lp(x: float, params: Params) -> float:
        p = GaussianModel.p(x, params)
        if p <= 0:
            return -np.inf
        return np.log(p)

    @staticmethod
    def ldm(x: float, params: Params) -> float:
        m, sd = params
        return (x - m) / (np.exp(2 * sd))

    @staticmethod
    def ldsd(x: float, params: Params) -> float:
        m, sd = params
        return ((x - m) ** 2) / np.exp(2 * sd) - 1

    @staticmethod
    def ld_params(x: float, params: Params) -> np.ndarray:
        return np.array([GaussianModel.ldm(x, params), GaussianModel.ldsd(x, params)])


class ExponentialModel(Model):
    """
    f(x) = l * e^(-lx)

    l = e^(_l)

    O = [_l]
    """

    @staticmethod
    def name() -> str:
        return "Exponential"

    @staticmethod
    def params_convert_to_model(params: Params) -> Params:
        return np.log(params)

    @staticmethod
    def params_convert_from_model(params: Params) -> Params:
        return np.exp(params)

    @staticmethod
    def generate(params: Params, size: int = 1) -> Samples:
        return np.array(expon.rvs(scale=1 / params[0], size=size))

    @staticmethod
    def p(x: float, params: Params) -> float:
        if x < 0:
            return 0
        (l,) = params
        return np.exp(l - np.exp(l) * x)

    @staticmethod
    def lp(x: float, params: Params) -> float:
        if x < 0:
            return -np.inf
        (l,) = params
        return l - np.exp(l) * x

    @staticmethod
    def ldl(x: float, params: Params) -> float:
        if x < 0:
            return -np.inf
        (l,) = params
        return 1 - np.exp(l) * x

    @staticmethod
    def ld_params(x: float, params: Params) -> np.ndarray:
        return np.array([ExponentialModel.ldl(x, params)])
