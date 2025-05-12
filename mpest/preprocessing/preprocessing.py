"""Module wich contains the method of task creation for EM algorithm"""

import numpy as np

from mpest import Problem, Samples
from mpest.preprocessing.components_classifier.abstract_classifier import AComponentsClassifier
from mpest.preprocessing.components_number import AComponentsNumber
from mpest.preprocessing.parameterization import Parameterization


class Preprocessing:
    """
    Preprocessing
    -----
    :param components_number:     AComponentsNumber                      — Number of components estimator
    :param components_classifier: AComponentsClassifier                  — Classifier of components in mixture
    :param random_state:          int | None             default: None   — Determines random generation parameters
    """

    def __init__(
        self,
        components_number: AComponentsNumber,
        components_classifier: AComponentsClassifier,
        random_state: int | None = None
    ) -> None:
        self.components_number = components_number
        self.components_classifier = components_classifier
        self.random_state = random_state

    def get_problem(self, samples: Samples) -> Problem:
        """Function for creating a task for EM algorithm"""

        k = self.components_number.estimate(samples)
        models = self.components_classifier.predict(samples, k)

        np.random.seed(self.random_state)

        return Parameterization().get_problem(models, samples)
