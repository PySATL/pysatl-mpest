"""Module which represents distribution."""

from abc import ABC, abstractmethod

import numpy as np

from mpest.annotations import Params, Samples
from mpest.models import AModel, AModelWithGenerator


class APDFAble(ABC):
    """Abstract class which represents probability density function method"""

    # pylint: disable=too-few-public-methods

    @abstractmethod
    def pdf(self, x: float) -> float:
        """Probability density function"""


class AWithGenerator(ABC):
    """Abstract class which represents sample generation method"""

    # pylint: disable=too-few-public-methods

    @abstractmethod
    def generate(self, size=1) -> Samples:
        """Generates samples"""


class Distribution(APDFAble, AWithGenerator):
    """Class which represents all needed data about distribution."""

    def __init__(
        self,
        model: AModel | AModelWithGenerator,
        params: Params,
    ) -> None:
        self._model = model
        self._params = params

    @classmethod
    def from_params(
        cls,
        model: type[AModel],
        params: list[float],
    ) -> "Distribution":
        """User friendly Distribution initializer"""

        model_obj = model()
        return cls(model_obj, np.array(params))

    def pdf(self, x: float):
        return self.model.pdf(x, self.model.params_convert_to_model(self.params))

    @property
    def model(self):
        """Model of distribution getter."""
        return self._model

    @property
    def params(self):
        """Params getter."""
        return self._params

    @property
    def has_generator(self):
        """Returns True if contained model has generator"""
        return isinstance(self.model, AModelWithGenerator)

    def generate(self, size=1):
        """
        Generates samples of contained model
        - Raises TypeError if model hasn't got a generator
        """
        if not isinstance(self.model, AModelWithGenerator):
            raise TypeError(f"Model {self.model} hasn't got a generator")
        return self.model.generate(self.model.params_convert_to_model(self.params), size=size)
