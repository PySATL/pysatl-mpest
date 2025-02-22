"""Module which contains distribution checker by finiteness of it's params"""

import numpy as np

from mpest.core.mixture_distribution import DistributionInMixture
from mpest.em.distribution_checkers.unionable_distribution_checker import (
    AUnionableDistributionChecker,
)


class FiniteChecker(AUnionableDistributionChecker):
    """Class which represents distribution checker by finiteness of it's params"""

    @property
    def name(self):
        return "FiniteChecker"

    def is_alive(
        self,
        step: int,
        distribution: DistributionInMixture,
    ) -> bool:
        if (distribution.prior_probability is not None) and not np.isfinite(distribution.prior_probability):
            return False
        return bool(np.all(np.isfinite(distribution.params)))
