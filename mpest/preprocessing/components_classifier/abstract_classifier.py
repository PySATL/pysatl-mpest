"""Module which contains abstract class for classifiers of components in mixture"""

from abc import ABC, abstractmethod

from mpest.annotations import Samples
from mpest.preprocessing.utils import Model
from mpest.utils import ANamed


class AComponentsClassifier(ANamed, ABC):
    """Abstract class for classifiers of components in mixture"""

    @abstractmethod
    def predict(self, samples: Samples, k: int) -> list[Model]:
        """The function for classification of components in mixture"""
