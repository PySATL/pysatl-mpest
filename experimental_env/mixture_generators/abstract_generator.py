""" A module that provides an abstract class for generating a mixture. """
from abc import ABC, abstractmethod

import numpy as np

from mpest import Distribution, MixtureDistribution
from mpest.models import AModel


class AMixtureGenerator(ABC):
    """
    An abstract class for generating mixtures.
    """

    @abstractmethod
    def generate_priors(self, models: list[type[AModel]]) -> list[float]:
        """
        A method for choosing how a prior probabilities will be generated.
        The function is needed so that, with a uniform distribution, degenerate distributions are not generated.
        """

    @abstractmethod
    def generate_dists(self, models: list[type[AModel]]) -> list[Distribution]:
        """
        A method for choosing how a params will be generated.
        """

    def create_mixture(
        self, models: list[type[AModel]], seed: int = 42
    ) -> MixtureDistribution:
        """
        Function for generating random mixture
        """
        np.random.seed(seed)
        priors = self.generate_priors(models)
        dists = self.generate_dists(models)

        return MixtureDistribution.from_distributions(dists, priors)
