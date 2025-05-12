"""Module which contains Constant Classifier"""

from mpest.annotations import Samples
from mpest.preprocessing.components_classifier.abstract_classifier import AComponentsClassifier
from mpest.preprocessing.utils import Model


class CModel(AComponentsClassifier):
    """
    X-Means method
    -----
    :param model:      AModel     — Constant class for classifier
    """

    def __init__(self, model: Model) -> None:
        self.model = model

    @property
    def name(self) -> str:
        return "Constant Classifier"

    def predict(self, samples: Samples, k: int) -> list[Model]:
        return [self.model for _ in range(k)]
