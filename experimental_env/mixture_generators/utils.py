""" A module containing parameter generators for mixture generators. """
from random import uniform

from mpest import Distribution
from mpest.models import AModel, ExponentialModel, GaussianModel


def generate_standart_params(models: list[type[AModel]]) -> list[Distribution]:
    """
    Generate standard parameters
    """
    dists = []
    for m in models:
        if m == ExponentialModel:
            params = [1.0]
        elif m == GaussianModel:
            params = [0.0, 1.0]
        else:
            params = [1.0, 1.5]

        dists.append(Distribution.from_params(m, params))

    return dists


def generate_uniform_params(models: list[type[AModel]]) -> list[Distribution]:
    """
    Generate parameters based on a uniform distribution
    """
    dists = []
    for m in models:
        if m == ExponentialModel:
            params = [uniform(0.1, 5.0)]
        elif m == GaussianModel:
            params = [uniform(-5.0, 5.0), uniform(0.1, 5.0)]
        else:
            params = [uniform(0.1, 5.0), uniform(0.1, 5.0)]

        dists.append(Distribution.from_params(m, params))

    return dists
