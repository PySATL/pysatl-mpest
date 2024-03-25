from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from scipy.optimize import minimize

from utils import Params


class Optimizer(ABC):
    @staticmethod
    @abstractmethod
    def name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def minimize(
        func: Callable[[Params], float],
        params: Params,
        jacobian: Callable[[Params], np.ndarray],
    ) -> Params:
        pass


class ScipyNewtonCG(Optimizer):
    @staticmethod
    def name():
        return "ScipyNewtonCG"

    @staticmethod
    def minimize(
        func: Callable[[Params], float],
        params: Params,
        jacobian: Callable[[Params], np.ndarray],
    ) -> Params:
        return minimize(func, params, jac=jacobian, method="Newton-CG").x
