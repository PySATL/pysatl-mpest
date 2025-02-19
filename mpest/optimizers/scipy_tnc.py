"""Module which contains truncated Newton (TNC) optimizer"""

from collections.abc import Callable

from scipy.optimize import minimize

from mpest.optimizers.abstract_optimizer import AOptimizer
from mpest.types import Params


class ScipyTNC(AOptimizer):
    """Class which represents SciPy truncated Newton (TNC) optimizer"""

    @property
    def name(self):
        return "ScipyTNC"

    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
    ) -> Params:
        return minimize(func, params, method="TNC").x
