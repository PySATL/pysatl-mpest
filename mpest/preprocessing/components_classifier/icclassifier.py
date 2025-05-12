"""Module which contains IC-based Classifier"""

from itertools import combinations_with_replacement

import numpy as np

from mpest.annotations import Samples
from mpest.em import EM
from mpest.em.breakpointers import ParamDifferBreakpointer, StepCountBreakpointer
from mpest.em.distribution_checkers import FiniteChecker, PriorProbabilityThresholdChecker
from mpest.em.methods.abstract_steps import AExpectation, AMaximization
from mpest.em.methods.method import Method
from mpest.models import ExponentialModel, GaussianModel, WeibullModelExp
from mpest.preprocessing.components_classifier.abstract_classifier import AComponentsClassifier
from mpest.preprocessing.components_classifier.constant import CModel
from mpest.preprocessing.criterions.abstract_criterion import ACriterion
from mpest.preprocessing.parameterization import Parameterization
from mpest.preprocessing.utils import Model


class ICClassifier(AComponentsClassifier):
    """
    IC-based Classifier
    -----
    :param criterion:     ACriterion                      — Information criterion
    :param estep:         AExpectation                    — Selected EStep for EM
    :param mstep:         AMaximization                   — Selected MStep for EM
    :param random_state:  int | None      default: None   — Determines random generation parameters
    """

    def __init__(
        self,
        criterion: ACriterion,
        estep: AExpectation,
        mstep: AMaximization,
        random_state: int | None = None
    ) -> None:
        self.criterion = criterion
        self.estep = estep
        self.mstep = mstep
        self.random_state = random_state
        self.models: list[Model] = [GaussianModel(), WeibullModelExp(), ExponentialModel()]

    @property
    def name(self) -> str:
        return "IC-based Classifier"

    def predict(self, samples: Samples, k: int) -> list[Model]:
        search_limit = 2

        if samples.min() < 0 and (k > search_limit or k == 1):
            return CModel(GaussianModel()).predict(samples, k)

        np.random.seed(self.random_state)

        criterions = []
        distributions = []

        method = Method(self.estep, self.mstep)
        em_algo = EM(
            StepCountBreakpointer(16) + ParamDifferBreakpointer(0.01),
            FiniteChecker() + PriorProbabilityThresholdChecker(),
            method,
        )

        if k <= search_limit and samples.min() >= 0:
            model_combinations = list(combinations_with_replacement(self.models, k))
        elif k <= search_limit and samples.min() < 0:
            model_combinations = [
                (GaussianModel(), *model) for model in combinations_with_replacement(self.models, k-1)
            ]
        else:
            model_combinations = [tuple(CModel(self.models[i]).predict(samples, k)) for i in range(len(self.models))]

        for models in model_combinations:
            problem = Parameterization().get_problem(models, samples)
            result = em_algo.solve(problem).content.distributions
            result = [m for m in result if m.prior_probability]

            criterions.append(self.criterion.estimate(result, samples))
            distributions.append(result)

        return [distribution.model for distribution in distributions[np.argmin(criterions)]]
