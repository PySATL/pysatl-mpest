from itertools import combinations_with_replacement

import numpy as np

from mpest import Distribution, MixtureDistribution, Problem, Samples
from mpest.components_number.criterions.abstract_criterion import ACriterion
from mpest.components_number.methods.abstract_estimator import AComponentsNumber
from mpest.em import EM
from mpest.em.breakpointers import ParamDifferBreakpointer, StepCountBreakpointer
from mpest.em.distribution_checkers import (
    FiniteChecker,
    PriorProbabilityThresholdChecker,
)
from mpest.em.methods.abstract_steps import AExpectation, AMaximization
from mpest.em.methods.method import Method
from mpest.models import AModel, ExponentialModel, GaussianModel, WeibullModelExp


class XMeans(AComponentsNumber):
    def __init__(
        self,
        kmax: int,
        criterion: ACriterion,
        estep: AExpectation,
        mstep: AMaximization,
        random_state: int | None = None,
    ) -> None:
        self.kmax = kmax
        self.criterion = criterion
        self.estep = estep
        self.mstep = mstep
        self.random_state = random_state
        self.models = (GaussianModel, WeibullModelExp, ExponentialModel)

    @property
    def name(self) -> str:
        return "X-Means"

    @staticmethod
    def _generate_params(model: AModel, samples: Samples) -> list[float]:
        if model is GaussianModel:
            m = np.mean(samples) + np.random.normal(0, 0.1 * np.std(samples))
            sd = np.abs(np.random.normal(0.5 * np.std(samples), 0.25 * np.std(samples)))
            return [m, sd]
        if model is WeibullModelExp:
            k = np.random.uniform(0.5, 5)
            l = np.random.uniform(0.1, 10)
            return [k, l]
        if model is ExponentialModel:
            l = np.random.uniform(0.1, 10)
            return [l]

    def _generate_problem(self, models: list[AModel], samples: Samples) -> Problem:
        params = []
        for model in models:
            params.append(self._generate_params(model, samples))
        problem = Problem(
            samples=samples,
            distributions=MixtureDistribution.from_distributions(
                [Distribution(model(), param) for model, param in zip(models, params)]
            ),
        )
        return problem

    def estimate(self, samples: Samples) -> float:
        negative = samples.min() < 0
        np.random.seed(self.random_state)

        method = Method(self.estep, self.mstep)
        em_algo = EM(
            StepCountBreakpointer(16) + ParamDifferBreakpointer(0.01),
            FiniteChecker() + PriorProbabilityThresholdChecker(),
            method,
        )

        criterions = []
        distributions = []

        for k in range(1, self.kmax + 1):
            if k == 2:
                model_combinations = combinations_with_replacement(self.models, 2)
            else:
                model_combinations = [
                    [self.models[i] for _ in range(k)] for i in range(len(self.models))
                ]

            for models in model_combinations:
                if negative and GaussianModel not in models:
                    continue
                problem = self._generate_problem(models, samples)
                result = em_algo.solve(problem).content.distributions
                result = [m for m in result if m.prior_probability]

                criterions.append(self.criterion.estimate(result, samples))
                distributions.append(result)

        return len(distributions[np.argmin(criterions)])
