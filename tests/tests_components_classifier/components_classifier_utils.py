"""Module which contain utility for testing methods components classifier"""


import numpy as np

from mpest import Distribution, MixtureDistribution
from mpest.models import AModelWithGenerator, GaussianModel, WeibullModelExp, ExponentialModel
from mpest.preprocessing.components_classifier import AComponentsClassifier


def run_test(
    models: list[AModelWithGenerator],
    params: list[list[float]],
    prior_probabilities: list[float],
    size: int,
    method: AComponentsClassifier,
) -> int:
    """Run a test scenario"""

    np.random.seed(42)
    possible_models = [GaussianModel, WeibullModelExp, ExponentialModel]

    base_mixture_distribution = MixtureDistribution.from_distributions(
        [Distribution(model, param) for model, param in zip(models, params)],
        prior_probabilities,
    )

    x = base_mixture_distribution.generate(size)
    result = method.predict(x, len(models))

    counter = [[1 if isinstance(curr_model, model) else 0 for model in possible_models] for curr_model in result]
    return np.sum(np.array(counter), axis=0)
