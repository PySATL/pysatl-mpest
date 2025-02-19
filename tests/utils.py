"""TODO"""

from itertools import permutations

import numpy as np

from mpest.core.mixture_distribution import MixtureDistribution
from mpest.utils import ResultWithError


def check_for_params_error_tolerance(
    results: list[ResultWithError[MixtureDistribution]],
    base_mixture: MixtureDistribution,
    expected_error: float,
) -> bool:
    """A function to check for compliance with an expected parameters error"""

    def absolute_diff_params(a: MixtureDistribution, b: MixtureDistribution):
        """Metric which checks absolute differ of params of gotten distribution mixtures"""

        a_p, b_p = ([d.params for d in ld] for ld in (a, b))

        return min(sum(np.sum(np.abs(x - y)) for x, y in zip(a_p, _b_p)) for _b_p in permutations(b_p))

    for result in results:
        assert result.error is None
        actual_error = absolute_diff_params(result.result, base_mixture)
        print(actual_error)
        if actual_error <= expected_error:
            return True
    return False


def check_for_priors_error_tolerance(
    results: list[ResultWithError[MixtureDistribution]],
    base_mixture: MixtureDistribution,
    expected_error: float,
) -> bool:
    """A function to check for compliance with an expected prior probabilities error"""

    def absolute_diff_priors(a: MixtureDistribution, b: MixtureDistribution):
        """
        Metric which checks absolute differ of prior probabilities of gotten distribution mixtures
        """

        a_p, b_p = ([d.prior_probability for d in ld] for ld in (a, b))

        return min(sum(np.sum(np.abs(x - y)) for x, y in zip(a_p, _b_p)) for _b_p in permutations(b_p))

    for result in results:
        assert result.error is None
        actual_error = absolute_diff_priors(result.result, base_mixture)
        print(actual_error)
        if actual_error <= expected_error:
            return True
    return False
