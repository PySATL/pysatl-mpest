import numpy as np
from mpest.components_number.methods import AComponentsNumber
from mpest import Distribution, MixtureDistribution
from mpest.models import AModelWithGenerator


def run_test(models: list[AModelWithGenerator], params: list[list[float]],
             prior_probabilities: list[float], size: int, method: AComponentsNumber) -> int:

    np.random.seed(42)

    base_mixture_distribution = MixtureDistribution.from_distributions(
        [
            Distribution(model, param) for model, param in zip(models, params)
        ],
        prior_probabilities
    )

    x = base_mixture_distribution.generate(size)
    result = method.estimate(x)
    return result
