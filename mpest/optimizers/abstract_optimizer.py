"""Module which contains abstract optimizer classes"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from mpest.annotations import Params
from mpest.utils import ANamed


class AOptimizer(ANamed, ABC):
    """Abstract class which represents simple optimizer"""

    @abstractmethod
    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
    ) -> Params:
        """Optimization minimization method"""


class AOptimizerJacobian(ANamed, ABC):
    """Abstract class which represents gradient method optimizer"""

    @abstractmethod
    def minimize(
        self,
        func: Callable[[Params], float],
        params: Params,
        jacobian: Callable[[Params], np.ndarray],
    ) -> Params:
        """Optimization minimization method, which also needs jacobian for work"""


TOptimizer = AOptimizer | AOptimizerJacobian
