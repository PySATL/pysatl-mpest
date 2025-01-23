""" A module that provides a class for generating initial mixtures with standart parameters """

from experimental_env.experiment.experiment_executors.abstract_executor import AExecutor
from experimental_env.mixture_generators.standart_mixture_generator import (
    StandartMixtureGenerator,
)
from mpest import Problem


class StandartExperimentExecutor(AExecutor):
    """
    A performer who generates standart params for init mixture.
    """

    def __init__(self, path, cpu_count, seed=42):
        super().__init__(path, cpu_count, seed)

    def init_problems(self, ds_descriptions, models):
        return [
            Problem(
                descr.samples,
                StandartMixtureGenerator().create_mixture(models, self._seed + i),
            )
            for i, descr in enumerate(ds_descriptions)
        ]
