"""Module which contains CG (Conjugate Gradient iteration) optimizer"""

from collections.abc import Callable

from scipy.optimize import minimize

from mpest.annotations import Params
from mpest.optimizers.abstract_optimizer import AOptimizer


class ScipyCG(AOptimizer):
    """Class which represents SciPy CG (Conjugate Gradient iteration) optimizer"""

    @property
    def name(self):
        return "ScipyCG"

    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
    ) -> Params:
        return minimize(func, params, method="CG").x
