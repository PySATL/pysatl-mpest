"""Module which contains abstract model classes, which describe distribution models"""

from abc import ABC, abstractmethod

import numpy as np

from mpest.annotations import Params, Samples
from mpest.utils import ANamed


class AModel(ANamed, ABC):
    """Abstract class which represents all methods needed from model for EM algorithm"""

    @abstractmethod
    def params_convert_to_model(self, params: Params) -> Params:
        """
        Method which converts generally accepted distribution params
        into local ones, for handling optimizers which needs continuous distributions
        """

    @abstractmethod
    def params_convert_from_model(self, params: Params) -> Params:
        """
        Method which converts local distribution params into
        generally accepted ones, for handling optimizers which needs continuous distributions
        """

    @abstractmethod
    def pdf(self, x: float, params: Params) -> float:
        """Probability density function"""

    @abstractmethod
    def lpdf(self, x: float, params: Params) -> float:
        """
        Logarithm of probability density function,
        for speed improvements if it possible
        """


class AModelDifferentiable(AModel, ABC):
    """Abstract class which extends AModel by adding derivatives"""

    @abstractmethod
    def ld_params(self, x: float, params: Params) -> np.ndarray:
        """
        Method which returns list of logarithms of partial derivatives
        with respects to all it's params
        """


class AModelWithGenerator(AModel, ABC):
    """
    Abstract class which extends AModel by adding ability to generate samples
    by given params
    """

    @abstractmethod
    def generate(self, params: Params, size: int = 1, normalized: bool = False) -> Samples:
        """
        Method which generates samples by given params
        :param size: sample size

        :param normalize: by default normalized=False which means params are not normalized,
         normalized=True if params normalized

        :return: Samples with elements that belong to the distribution density of this model
        """
