""" A module that provides a mixture generator class in which parameters are generated using a uniform distribution. """
from experimental_env.mixture_generators.abstract_generator import AMixtureGenerator
from experimental_env.mixture_generators.utils import generate_uniform_params


class RandomMixtureGenerator(AMixtureGenerator):
    """
    A class for generating a prior probabilities for init mixtures in experiment stage.
    Params generated randomly.
    """

    def generate_dists(self, models):
        return generate_uniform_params(models)

    def generate_priors(self, models):
        return [1 / len(models) for _ in range(len(models))]
