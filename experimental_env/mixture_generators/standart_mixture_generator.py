""" A module providing a class for generating mixtures with standard parameters. """

from experimental_env.mixture_generators.abstract_generator import AMixtureGenerator
from experimental_env.mixture_generators.utils import generate_standart_params


class StandartMixtureGenerator(AMixtureGenerator):
    """
    A class that generates a mixture with standard parameters.
    """

    def generate_dists(self, models):
        return generate_standart_params(models)

    def generate_priors(self, models):
        return [1 / len(models) for _ in range(len(models))]
