"""Module which contain utility for testing methods of estimating the number of components"""

import numpy as np

from mpest import Distribution, MixtureDistribution
from mpest.models import AModelWithGenerator
from mpest.preprocessing.components_number import AComponentsNumber


def run_test(
    models: list[AModelWithGenerator],
    params: list[list[float]],
    prior_probabilities: list[float],
    size: int,
    method: AComponentsNumber,
) -> int:
    """Run a test scenario"""

    np.random.seed(42)

    base_mixture_distribution = MixtureDistribution.from_distributions(
        [Distribution(model, param) for model, param in zip(models, params)],
        prior_probabilities,
    )

    x = base_mixture_distribution.generate(size)
    result = method.estimate(x)
    return result
