"""Functions for experimental environment"""

import json
from random import uniform

from scipy.stats import dirichlet

from mpest import Distribution, MixtureDistribution
from mpest.models import AModel, ExponentialModel, GaussianModel, WeibullModelExp

DISTS = {
    ExponentialModel().name: ExponentialModel,
    GaussianModel().name: GaussianModel,
    WeibullModelExp().name: WeibullModelExp,
}


def create_mixture(models: list[type[AModel]]):
    """Function for generating random mixture"""
    dists = []
    priors = dirichlet.rvs([1 for _ in range(len(models))], 1)[0]
    for m in models:
        if m == ExponentialModel:
            params = [uniform(0.1, 5)]
        elif m == GaussianModel:
            params = [uniform(-5, 5), uniform(0.1, 5)]
        else:
            params = [uniform(0.1, 5), uniform(0.1, 5)]

        dists.append(Distribution.from_params(m, params))

    return MixtureDistribution.from_distributions(dists, priors)


def unpack_dataset(ds):
    """Function for unpack dataset"""
    mixture_output = []
    samples = []
    with open(ds, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        for data in dataset:
            current_dists = []
            current_prior = []
            mixture_dict = data[0]
            samples.append(data[1])
            for name, params in mixture_dict.items():
                for i in range(0, len(params) - 1, 2):
                    dist = Distribution.from_params(DISTS[name], params[i])
                    current_prior.append(params[i + 1])
                    current_dists.append(dist)

            mixture_output.append(
                MixtureDistribution.from_distributions(current_dists, current_prior)
            )
    return mixture_output, samples
