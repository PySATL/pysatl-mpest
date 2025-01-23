""" A module that provides a mixture generator in which parameters are generated in accordance with the requirements for the dataset. """
import numpy as np

from experimental_env.mixture_generators.abstract_generator import AMixtureGenerator
from experimental_env.mixture_generators.utils import generate_uniform_params


class DatasetMixtureGenerator(AMixtureGenerator):
    """
    A class for generating a prior probabilities for datasets.
    The alphas for the Dirichlet distribution are 5, which means that a prior probabilities will rarely be close to 0 or 1
    """

    def generate_dists(self, models):
        return generate_uniform_params(models)

    def generate_priors(self, models):
        return np.random.dirichlet(alpha=[5 for _ in range(len(models))], size=1)[0]
