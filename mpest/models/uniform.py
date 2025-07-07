import numpy as np

from mpest.annotations import Params, Samples
from mpest.models import AModelDifferentiable, AModelWithGenerator


class LMomentsParameterMixin:
    def calc_alpha(self, moments: list[float]) -> float:
        return moments[0] - 3 * moments[1]

    def calc_beta(self, moments: list[float]) -> float:
        return moments[0] + 3 * moments[1]


class Uniform(AModelDifferentiable, AModelWithGenerator, LMomentsParameterMixin):
    @property
    def name(self) -> str:
        """Returns the name of the distribution.

        Returns:
            str: The name of the distribution "Uniform".
        """

        return "Uniform"

    def params_convert_to_model(self, params: Params) -> Params:
        return params

    def params_convert_from_model(self, params: Params) -> Params:
        return params

    def generate(self, params: Params, size: int = 1, normalized: bool = True) -> Samples:
        return np.random.uniform(params[0], params[1], size=size)

    def pdf(self, x: float, params: Params) -> float:
        a, b = params
        return 1 / (b - a) if a <= x <= b else 0

    def lpdf(self, x: float, params: Params) -> float:
        a, b = params
        if a <= x <= b:
            return -np.log(b - a)
        else:
            return -np.inf

    def lda(self, x: float, params: Params) -> float:
        a, b = params
        if a <= x <= b:
            return 1 / (b - a)
        else:
            return -np.inf

    def ldb(self, x: float, params: Params) -> float:
        a, b = params
        if a <= x <= b:
            return -1 / (b - a)
        else:
            return -np.inf

    def ld_params(self, x: float, params: Params) -> np.ndarray:
        """
        Method which returns logarithm of derivative with respect to params
        """

        return np.array([self.lda(x, params), self.ldb(x, params)])

    def calc_params(self, moments: list[float]) -> np.ndarray:
        return np.array([self.calc_alpha(moments), self.calc_beta(moments)])
