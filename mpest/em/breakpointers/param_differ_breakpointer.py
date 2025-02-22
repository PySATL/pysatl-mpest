"""Module which contains EM breakpointer by sum of difference between params"""

import numpy as np

from mpest.core.mixture_distribution import MixtureDistribution
from mpest.em.breakpointers.unionable_breakpointer import AUnionableBreakpointer


class ParamDifferBreakpointer(AUnionableBreakpointer):
    """Class which represents EM breakpointer by sum of difference between params"""

    def __init__(self, deviation: float = 0.01) -> None:
        self._deviation = deviation

    @property
    def deviation(self):
        """Max deviation getter"""
        return self._deviation

    @property
    def name(self):
        return f"ParamDifferBreakpointer(deviation={self.deviation})"

    def is_over(
        self,
        step: int,
        previous_step: MixtureDistribution | None,
        current_step: MixtureDistribution,
    ) -> bool:
        if previous_step is None:
            return False

        if len(previous_step) != len(current_step):
            return False

        for d_p, d_c in zip(previous_step, current_step):
            if np.any(np.abs(d_p.params - d_c.params) > self.deviation):
                return False
            d_pp = 0.0 if d_p.prior_probability is None else d_p.prior_probability
            d_cp = 0.0 if d_c.prior_probability is None else d_c.prior_probability
            if np.any(np.abs(d_pp - d_cp) > self.deviation):
                return False
        return True
