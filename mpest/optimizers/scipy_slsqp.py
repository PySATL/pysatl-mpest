"""Module which contains Sequential Least Squares Programming (SLSQP) optimizer"""

from collections.abc import Callable

from scipy.optimize import minimize

from mpest.annotations import Params
from mpest.optimizers.abstract_optimizer import AOptimizer


class ScipySLSQP(AOptimizer):
    """Class which represents SciPy Sequential Least Squares Programming (SLSQP) optimizer"""

    @property
    def name(self):
        return "ScipySLSQP"

    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
    ) -> Params:
        return minimize(func, params, method="SLSQP").x
