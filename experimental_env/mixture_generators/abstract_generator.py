"""A module that provides an abstract class for generating a mixture."""

import random
from abc import ABC, abstractmethod

from mpest import Distribution, MixtureDistribution
from mpest.models import AModel


class AMixtureGenerator(ABC):
    """
    An abstract class for generating mixtures.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

    @abstractmethod
    def generate_priors(self, models: list[type[AModel]]) -> list[float | None]:
        """
        A method for choosing how a prior probabilities will be generated.
        The function is needed so that, with a uniform distribution, degenerate distributions are not generated.
        """

    @abstractmethod
    def generate_dists(self, models: list[type[AModel]]) -> list[Distribution]:
        """
        A method for choosing how a params will be generated.
        """

    def create_mixture(self, models: list[type[AModel]]) -> MixtureDistribution:
        """
        Function for generating random mixture
        """
        priors = self.generate_priors(models)
        dists = self.generate_dists(models)

        return MixtureDistribution.from_distributions(dists, priors)
