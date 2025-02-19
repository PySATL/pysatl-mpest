"""Module which contains Newton-CG optimizer"""

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize

from mpest.annotations import Params
from mpest.optimizers.abstract_optimizer import AOptimizerJacobian


class ScipyNewtonCG(AOptimizerJacobian):
    """Class which represents SciPy Newton-CG optimizer"""

    @property
    def name(self):
        return "ScipyNewtonCG"

    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
        jacobian: Callable[[Params], np.ndarray],
    ) -> Params:
        return minimize(func, params, jac=jacobian, method="Newton-CG").x
