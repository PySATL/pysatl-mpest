"""Module which contains abstract сlass for methods estimating number of components in mixture"""

from abc import ABC, abstractmethod

from mpest.types import Samples
from mpest.utils import ANamed


class AComponentsNumber(ANamed, ABC):
    """Abstract сlass for methods estimating number of components in mixture"""

    @abstractmethod
    def estimate(self, samples: Samples) -> float:
        """The function for estimating number of components"""
