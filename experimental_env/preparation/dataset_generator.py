"""Module from generating datasets with the given sample size, experimental counts and mixture"""

import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from experimental_env.mixture_generators.dataset_mixture_generator import (
    DatasetMixtureGenerator,
)
from experimental_env.preparation.dataset_saver import DatasetDescrciption, DatasetSaver
from mpest.core.distribution import Distribution
from mpest.core.mixture_distribution import MixtureDistribution
from mpest.models.abstract_model import AModel


class RandomDatasetGenerator:
    """
    Class that generates datasets from mixture.
    Randomize params of base mixture at each experiment.
    You can select the sample size and the number of experiments.
    """

    def __init__(self, seed: int = 42):
        """
        Setting seed for determined result.
        """
        random.seed(seed)
        self._seed = seed

    def generate(
        self,
        samples_size: int,
        models: list[type[AModel]],
        working_path: Path,
        exp_count: int = 10,
    ):
        """
        A function that generates datasets based on random mixture.
        """

        with tqdm(total=exp_count) as tbar:
            for i in range(exp_count):
                tbar.update()
                mixture = DatasetMixtureGenerator().create_mixture(models)
                samples = mixture.generate(samples_size)
                descr = DatasetDescrciption(samples_size, samples, mixture, i + 1)

                mixture_name_dir: Path = working_path.joinpath(descr.get_dataset_name())
                exp_dir: Path = mixture_name_dir.joinpath(f"experiment_{i + 1}")
                saver = DatasetSaver(exp_dir)
                saver.save_dataset(descr)


class ConcreteDatasetGenerator:
    """
    A preparation class that allows you to generate datasets based on user-selected mixtures
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self._dists: list[Distribution] = []
        self._priors: list[float | None] = []

    def add_distribution(self, model: type[AModel], params: list[float], prior: float) -> None:
        """
        Add distribution with params and prior to mixture.
        """
        self._dists.append(Distribution.from_params(model, params))
        self._priors.append(prior)

    def generate(self, samples_size: int, working_path: Path, exp_count: int):
        """
        A function that generates a dataset based on a user's mixture.

        TODO: Figure out how to implement overwriting or adding additional experiments to existing ones.
        """

        saved_exp_count = 0
        for i in range(1000):
            if saved_exp_count >= exp_count:
                break

            mixture = MixtureDistribution.from_distributions(self._dists, self._priors)
            samples = mixture.generate(samples_size)

            descr = DatasetDescrciption(samples_size, samples, mixture, i + 1)
            mixture_name_dir: Path = working_path.joinpath(descr.get_dataset_name())
            exp_dir: Path = mixture_name_dir.joinpath(f"experiment_{i + 1}")

            if exp_dir.exists():
                continue

            saver = DatasetSaver(exp_dir)
            saver.save_dataset(descr)
            saved_exp_count += 1
