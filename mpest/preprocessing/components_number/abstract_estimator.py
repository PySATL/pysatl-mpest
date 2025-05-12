"""Module which contains abstract class for methods estimating number of components in mixture"""

from abc import ABC, abstractmethod

from mpest.annotations import Samples
from mpest.utils import ANamed


class AComponentsNumber(ANamed, ABC):
    """Abstract class for methods estimating number of components in mixture"""

    @abstractmethod
    def estimate(self, samples: Samples) -> int:
        """The function for estimating number of components"""
