"""A module that provides an abstract class for performing the 2nd stage of the experiment"""

import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from experimental_env.experiment.estimators import AEstimator
from experimental_env.experiment.experiment_description import ExperimentDescription
from experimental_env.experiment.experiment_saver import ExperimentSaver
from experimental_env.preparation.dataset_description import DatasetDescrciption
from mpest import Problem
from mpest.models import ALL_MODELS, AModel


class AExecutor(ABC):
    """
    An abstract class that provides an interface for generating a mixture,
    as well as the implementation of the execute method, to implement the 2nd stage of the experiment.
    """

    def __init__(self, path: Path, cpu_count: int, seed):
        """
        Class constructor

        :param path: The path in which the results of the second stage of the experiment will lie
        :param seed: Seed for determined results.
        """

        self._out_dir = path
        self._cpu_count = cpu_count
        self._seed = seed
        np.random.seed(self._seed)

    @abstractmethod
    def init_problems(self, ds_descriptions: list[DatasetDescrciption], models: list[type[AModel]]) -> list[Problem]:
        """
        Function for generate problem any method user want.
        """

    def execute(self, preparation_results: dict, estimator: AEstimator) -> None:
        """
        Function for the execution of the second stage

        :param preparation_results: Data from the first stage received from the parser.
        :param estimator: Estimator
        """
        method_dir: Path = self._out_dir.joinpath(estimator.name)

        for mixture_name, ds_descriptions in preparation_results.items():
            mixture_name_dir: Path = method_dir.joinpath(mixture_name)
            models = [ALL_MODELS[model_name] for model_name in mixture_name.split("_")]

            problems = self.init_problems(ds_descriptions, models)

            # Disable warnings and estimating params.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = estimator.estimate(problems, self._cpu_count, self._seed)

            # Saving results
            for i, ds_descr in enumerate(ds_descriptions):
                exp_dir: Path = mixture_name_dir.joinpath(f"experiment_{ds_descr.exp_num}")
                result = results[i]

                result_descr = ExperimentDescription.from_result(problems[i].distributions, result, ds_descr)
                ExperimentSaver(exp_dir).save(result_descr)
