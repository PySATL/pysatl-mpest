"""Module which contains X-Means Method"""

from itertools import combinations_with_replacement

import numpy as np

from mpest import Distribution, MixtureDistribution, Problem, Samples
from mpest.em import EM
from mpest.em.breakpointers import ParamDifferBreakpointer, StepCountBreakpointer
from mpest.em.distribution_checkers import FiniteChecker, PriorProbabilityThresholdChecker
from mpest.em.methods.abstract_steps import AExpectation, AMaximization
from mpest.em.methods.method import Method
from mpest.models import AModel, AModelDifferentiable, ExponentialModel, GaussianModel, WeibullModelExp
from mpest.preprocessing.components_number.criterions.abstract_criterion import ACriterion
from mpest.preprocessing.components_number.methods.abstract_estimator import AComponentsNumber


class XMeans(AComponentsNumber):
    """
    X-Means method
    -----
    :param kmax:       int                       — Assumed maximum number of components
    :criterion:        ACriterion                — Information criterion
    :estep:            AExpectation              — Selected EStep for EM
    :mstep:            AMaximization             — Selected MStep for EM
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
        self.models = (GaussianModel(), WeibullModelExp(), ExponentialModel())

    @property
    def name(self) -> str:
        return "X-Means"

    def _generate_params(self, model: AModel, samples: Samples) -> np.ndarray:

        if isinstance(model, GaussianModel):
            m = np.mean(samples) + np.random.normal(0, 0.1 * np.std(samples))
            sd = np.abs(np.random.normal(0.5 * np.std(samples), 0.25 * np.std(samples)))
            return np.array([m, sd])
        if isinstance(model, WeibullModelExp):
            k = np.random.uniform(0.5, 5)
            lm = np.random.uniform(0.1, 10)
            return np.array([k, lm])
        else:
            lm = np.random.uniform(0.1, 10)
            return np.array([lm])

    def _generate_problem(self, models: tuple[AModelDifferentiable, ...], samples: Samples) -> Problem:

        params = []
        for model in models:
            params.append(self._generate_params(model, samples))
        problem = Problem(
            samples=samples,
            distributions=MixtureDistribution.from_distributions(
                [Distribution(model, param) for model, param in zip(models, params)]
            ),
        )
        return problem

    def estimate(self, samples: Samples) -> float:
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
        distributions = []

        for k in range(1, self.kmax + 1):
            if k <= search_limit:
                model_combinations = list(combinations_with_replacement(self.models, k))
            else:
                model_combinations = [tuple([self.models[i] for _ in range(k)]) for i in range(len(self.models))]

            for models in model_combinations:
                if negative and not any([isinstance(model, GaussianModel) for model in models]):
                    continue
                problem = self._generate_problem(models, samples)
                result = em_algo.solve(problem).content.distributions
                result = [m for m in result if m.prior_probability]

                criterions.append(self.criterion.estimate(result, samples))
                distributions.append(result)

        return len(distributions[np.argmin(criterions)])
