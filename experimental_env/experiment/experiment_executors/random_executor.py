"""A module that provides a class for generating initial mixtures with uniform distribution."""

from experimental_env.experiment.experiment_executors.abstract_executor import AExecutor
from experimental_env.mixture_generators.random_mixture_generator import (
    RandomMixtureGenerator,
)
from mpest import Problem


class RandomExperimentExecutor(AExecutor):
    """
    A performer who randomly generates the initial conditions for the algorithm.
    """

    def __init__(self, path, cpu_count, seed=42):
        super().__init__(path, cpu_count, seed)

    def init_problems(self, ds_descriptions, models):
        return [
            Problem(
                descr.samples,
                RandomMixtureGenerator(self._seed).create_mixture(models),
            )
            for i, descr in enumerate(ds_descriptions)
        ]
