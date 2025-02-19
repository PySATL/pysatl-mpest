"""Module which contains Nelder Mead optimizer"""

from collections.abc import Callable

from scipy.optimize import minimize

from mpest.optimizers.abstract_optimizer import AOptimizer
from mpest.types import Params


class ScipyNelderMead(AOptimizer):
    """Class which represents SciPy Nelder Mead optimizer"""

    @property
    def name(self):
        return "ScipyNelder-Mead"

    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
    ) -> Params:
        return minimize(func, params, method="Nelder-Mead").x
