"""Module wich contains a method for random parameterization of distributions"""

import numpy as np

from mpest import Distribution, MixtureDistribution, Problem, Samples
from mpest.models import GaussianModel, WeibullModelExp
from mpest.preprocessing.utils import Model


class Parameterization:
    """Class for random parameterization"""

    @staticmethod
    def get_componet_parameters(model: Model, samples: Samples) -> np.ndarray:
        """Function for random parametrication of distribution"""

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

    def get_components_parameters(self, models: list[Model] | tuple[Model, ...], samples: Samples) -> list[np.ndarray]:
        """Function for random parametrication of distribution"""

        return [self.get_componet_parameters(model, samples) for model in models]

    def get_problem(self, models: list[Model] | tuple[Model, ...], samples: Samples) -> Problem:
        """Function for creating random problem for EM algorithm"""

        params = self.get_components_parameters(models, samples)

        problem = Problem(
            samples=samples,
            distributions=MixtureDistribution.from_distributions(
                [Distribution(model, param) for model, param in zip(models, params)]
            ),
        )
        return problem
