"""Module which contains X-Means Method"""

from itertools import combinations_with_replacement

import numpy as np

from mpest.annotations import Samples
from mpest.em import EM
from mpest.em.breakpointers import ParamDifferBreakpointer, StepCountBreakpointer
from mpest.em.distribution_checkers import FiniteChecker, PriorProbabilityThresholdChecker
from mpest.em.methods.abstract_steps import AExpectation, AMaximization
from mpest.em.methods.method import Method
from mpest.models import ExponentialModel, GaussianModel, WeibullModelExp
from mpest.preprocessing.components_number.abstract_estimator import AComponentsNumber
from mpest.preprocessing.criterions.abstract_criterion import ACriterion
from mpest.preprocessing.parameterization import Parameterization
from mpest.preprocessing.utils import Model


class XMeans(AComponentsNumber):
    """
    X-Means method
    -----
    :param kmax:          int                             — Assumed maximum number of components
    :param criterion:     ACriterion                      — Information criterion
    :param estep:         AExpectation                    — Selected EStep for EM
    :param mstep:         AMaximization                   — Selected MStep for EM
    :param random_state:  int | None      default: None   — Determines random generation parameters for mixture model
    """

    def __init__(
        self,
        kmax: int,
        criterion: ACriterion,
        estep: AExpectation,
        mstep: AMaximization,
        random_state: int | None = None
    ) -> None:
        self.kmax = kmax
        self.criterion = criterion
        self.estep = estep
        self.mstep = mstep
        self.random_state = random_state
        self.models: list[Model] = [GaussianModel(), WeibullModelExp(), ExponentialModel()]

    @property
    def name(self) -> str:
        return "X-Means"

    def estimate(self, samples: Samples) -> int:
        np.random.seed(self.random_state)
        search_limit = 2

        negative = samples.min() < 0

        method = Method(self.estep, self.mstep)
        em_algo = EM(
            StepCountBreakpointer(16) + ParamDifferBreakpointer(0.01),
            FiniteChecker() + PriorProbabilityThresholdChecker(),
            method,
        )

        criterions = []
        mixtures = []

        for k in range(1, self.kmax + 1):
            if k <= search_limit:
                model_combinations = list(combinations_with_replacement(self.models, k))
            else:
                model_combinations = [tuple([self.models[i] for _ in range(k)]) for i in range(len(self.models))]

            for models in model_combinations:
                if negative and not any([isinstance(model, GaussianModel) for model in models]):
                    continue
                problem = Parameterization().get_problem(models, samples)
                result = em_algo.solve(problem).content.distributions
                result = [m for m in result if m.prior_probability]

                criterions.append(self.criterion.estimate(result, samples))
                mixtures.append(result)

        return len(mixtures[np.argmin(criterions)])
